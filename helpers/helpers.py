import torch
import re
import numpy as np
import time
import joblib
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from evaluate import load
from tqdm import tqdm
import sacrebleu
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import StandardScaler

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# translators
hs_translator = pipeline("translation", model = "Helsinki-NLP/opus-mt-en-he", device=device)
en_zh_translator = pipeline("translation", model = "Helsinki-NLP/opus-mt-en-zh", device=device)
zh_he_translator = pipeline("translation", model = "Helsinki-NLP/opus-mt-zh-he", device=device)
goog_translator = GoogleTranslator(source='en', target='iw')

# scoring pipelines
_bertscore = load("bertscore")
xnli = pipeline("text-classification", model="joeddav/xlm-roberta-large-xnli")
labse = SentenceTransformer("sentence-transformers/LaBSE")
e5 = SentenceTransformer("intfloat/multilingual-e5-base")
tok_he = AutoTokenizer.from_pretrained("onlplab/alephbert-base")
mlm_he = AutoModelForMaskedLM.from_pretrained("onlplab/alephbert-base").to(device).eval()

def batch_translate_hs(sentences, batch_size=64):
    translations = []
    
    for i in tqdm(range(0, len(sentences), batch_size), desc="Translating"):
        batch = sentences[i:i+batch_size]
        results = hs_translator(batch, batch_size=batch_size)
        translations.extend([r["translation_text"] for r in results])
    
    return translations

def batch_translate_hs_pv(sentences, batch_size=64):
    """translate enlish-chinese-hebrew"""
    chinese = []
    hebrew = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Translating English -> Chinese"):
        batch = sentences[i:i+batch_size]
        results = en_zh_translator(batch, batch_size=batch_size)
        chinese.extend([r["translation_text"] for r in results])

    for i in tqdm(range(0, len(chinese), batch_size), desc="Translating Chinese -> Hebrew"):
        batch = chinese[i:i+batch_size]
        results = zh_he_translator(batch, batch_size=batch_size)
        hebrew.extend([r["translation_text"] for r in results])

    return hebrew

def batch_translate_wbw_hs(sentences, batch_size=64):
    """word-by-word translation"""
    tokens = [s.split() for s in sentences] # list of list of tokens in sentence
    eng_vocab = list(dict.fromkeys(t for sent in tokens for t in sent)) # list of unique words
    heb_vocab = batch_translate_hs(eng_vocab, batch_size=batch_size)
    lex = dict(zip(eng_vocab, heb_vocab))
    translations = [" ".join(lex.get(t, t) for t in sent) for sent in tokens]

    return translations

def translate_gt(sentences):
    translations = []

    for s in tqdm(sentences, desc="Translating"):
        for i in range(3):
            try:
                translations.append(goog_translator.translate(s))
                break
            except Exception as e:
                if i < 2:
                    time.sleep(3)
                    continue
                print(e)
                translations.append("None")
    
    return translations

def batch_scores(sentences, translations, references, batch_size=64):
    bleu_scores = [
        sacrebleu.sentence_bleu(t, [r]).score
        for t, r in tqdm(zip(translations, references), desc="Scoring BLEU")
    ]
    chrf_scores = [
        sacrebleu.sentence_chrf(t, [r]).score
        for t, r in tqdm(zip(translations, references), desc="Scoring CHRF")
    ]

    print("Scoring BERT scores")
    bert_scores = _bertscore.compute(predictions=translations, references=[[r] for r in references], lang="he", batch_size=batch_size, device=device)["f1"]

    # Entailment    
    params = dict(
        top_k=None,
        batch_size=batch_size,
        padding=True, 
        truncation=True,
        max_length=256,
    )

    print("Scoring Bi-Di-Entailment scores")
    f_inputs = [{"text": p, "text_pair": h} for p, h in zip(sentences, translations)]
    b_inputs = [{"text": h, "text_pair": p} for p, h in zip(sentences, translations)]

    f = xnli(f_inputs, **params)
    b = xnli(b_inputs, **params)
    def pE(out): return next(d["score"] for d in out if d["label"].lower().startswith("entail"))
    ef = np.array([pE(o) for o in f])
    eb = np.array([pE(o) for o in b])
    of = ef / (1-ef+1e-12)
    ob = eb / (1-eb+1e-12)
    dot = of * ob
    bde_scores = (dot - dot.min()) / (dot.max() - dot.min() + 1e-12)

    print("Computing Cosine Similarity")
    tau = 0.07
    p = labse.encode(sentences, batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True)
    h = labse.encode(translations, batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True)
    s = util.cos_sim(p, h)
    cos_sim = ((torch.diagonal(s) + 1.0) / 2.0).cpu().numpy().astype(np.float32)

    print("Computing Penalties")
    penalties = length_penalty(sentences, translations) * latin_penalty(translations)

    print("Computing Hebrew Perplexity")
    heb_per = hebrew_perplexity(translations)

    print("Computing scores")
    features = [bde_scores, cos_sim, penalties, heb_per]
    scores = compute_score(features)

    return [bleu_scores, chrf_scores, bert_scores, bde_scores, cos_sim.tolist(), penalties, heb_per.tolist(), scores.tolist()]

def length_penalty(src_list, hyp_list):
    """
    Symmetric length ratio: min(|hyp|/|src|, |src|/|hyp|) in (0,1].
    """
    out = []
    for s, h in zip(src_list, hyp_list):
        ls = max(1, len(s.split()))
        lh = max(1, len(h.split()))
        r = lh / ls
        out.append(min(r, 1/r))
    return np.asarray(out, dtype=np.float32)

_lat_pat = re.compile(r"[A-Za-z]")

def latin_penalty(hyp_list, cap=0.30):
    """
    Penalize untranslated Latin tokens in the Hebrew hypothesis.
    """
    vals = []
    for h in hyp_list:
        toks = h.split()
        if not toks:
            vals.append(1.0)
            continue
        frac = sum(bool(_lat_pat.search(t)) for t in toks) / len(toks)
        vals.append(1.0 - min(frac, cap))
    return np.asarray(vals, dtype=np.float32)

@torch.inference_mode()
def hebrew_perplexity(translations, max_len=256):
    """
    texts: list[str] Hebrew sentences
    returns: np.array of per-sentence NLL
    """
    nlls = []
    mask_id = tok_he.mask_token_id
    for t in translations:
        enc = tok_he(t, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]
        # exclude specials from scoring
        specials = tok_he.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True)
        positions = [i for i, sp in enumerate(specials) if sp == 0 and attn[i].item() == 1]
        if not positions: #if empty
            nlls.append(10.0)
            continue
        # build a batch: each row has one masked position
        masked = input_ids.unsqueeze(0).repeat(len(positions), 1)
        masked[range(len(positions)), positions] = mask_id
        attn_b = attn.unsqueeze(0).repeat(len(positions), 1)

        out = mlm_he(masked.to(device), attention_mask=attn_b.to(device)).logits
        logp = torch.log_softmax(out[range(len(positions)), positions, :], dim=-1)
        true_ids = input_ids[positions].to(device)

        nll = (-logp[range(len(positions)), true_ids]).mean().item()
        nlls.append(nll)

    nll = np.array(nlls, dtype=np.float32)
    nmin, nmax = float(np.min(nll)), float(np.max(nll))

    return (nmax - nll) / (nmax - nmin + 1e-12)

def compute_score(features):
    ART = joblib.load("models/mt_score.joblib")
    FEATURES = ART["feature_names"]; W = ART["w"]; B = ART["b"]; SCALER = ART["scaler"]; ISO = ART["iso"]
    X = np.vstack(features).T
    Xs = SCALER.transform(X)
    s = Xs @ W + B
    y_hat = ISO.predict(s)
    return y_hat