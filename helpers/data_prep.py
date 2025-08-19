import os
import pandas as pd
from pathlib import Path

# flores101
def load_flores101():
    parent_path = Path(__file__).resolve().parents[1]
    flores_path = os.path.join(parent_path, "datasets", "flores101_dataset")
    eng_file = os.path.join(flores_path, "dev", "eng.dev")
    heb_file = os.path.join(flores_path, "dev", "heb.dev")

    with open(eng_file, encoding="utf-8") as f:
        eng_sentences = f.read().splitlines()

    with open(heb_file, encoding="utf-8") as f:
        heb_sentences = f.read().splitlines()

    df = pd.DataFrame({
        "eng": eng_sentences,
        "heb": heb_sentences
    })

    df.to_pickle("datasets/pkl_data/flores101.pkl")
    return df

# HebNLI
def load_hebNLI():
    splits = {'train': 'HebNLI_train.jsonl', 'dev': 'HebNLI_val.jsonl', 'test': 'HebNLI_test.jsonl'}
    df = pd.read_json("hf://datasets/HebArabNlpProject/HebNLI/" + splits["train"], lines=True)
    df.to_pickle("datasets/pkl_data/hebnli.pkl")
    return df

def load_datasets():
    flores_pkl_path = "datasets/pkl_data/flores101.pkl"
    hebnli_pkl_path = "datasets/pkl_data/hebnli.pkl"
    
    if os.path.exists(flores_pkl_path):
        flores101 = pd.read_pickle(flores_pkl_path)
    else:
        flores101 = load_flores101()
    
    if os.path.exists(hebnli_pkl_path):
        hebnli = pd.read_pickle(hebnli_pkl_path)
    else:
        hebnli = load_hebNLI()

    return flores101, hebnli

def create_translated_df(sentences, translations, references, scores, method, dataset):
    df = pd.DataFrame({
        "eng": sentences,
        "heb_reference": references,
        "heb_translation": translations,
        "bleu_score": scores[0],
        "chrf_score": scores[1],
        "bert_score": scores[2],
        "bde_score": scores[3],
        "cos_sim": scores[4],
        "penalties": scores[5],
        "heb_fluency": scores[6],
        "mt_score": scores[7],
    })

    df.to_pickle(f"datasets/pkl_data/{method}_{dataset}.pkl")
    return df