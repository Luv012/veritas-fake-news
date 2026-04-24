"""Train your own lightweight fake news classifier on the ISOT dataset.

This is an *optional* extra for your project report — shows a classical ML
baseline (TF-IDF + Logistic Regression) alongside the transformer approach
already wired into src/classifier.py.

1. Download the ISOT Fake News dataset: https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/
   You should get two CSVs: Fake.csv and True.csv. Put them in data/isot/.
2. Run:  python train/train_tfidf_baseline.py
3. The trained model is saved to data/tfidf_baseline.joblib and a confusion
   matrix + classification report are printed.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "isot"
FAKE = DATA / "Fake.csv"
TRUE = DATA / "True.csv"
OUT = ROOT / "data" / "tfidf_baseline.joblib"


def load():
    if not FAKE.exists() or not TRUE.exists():
        print(f"Put Fake.csv and True.csv in {DATA} first.")
        sys.exit(1)
    fake = pd.read_csv(FAKE); fake["label"] = 1
    real = pd.read_csv(TRUE); real["label"] = 0
    df = pd.concat([fake, real], ignore_index=True).sample(frac=1, random_state=42)
    df["text"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.strip()
    return df[["text", "label"]]


def main():
    df = load()
    X_tr, X_te, y_tr, y_te = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                                  stop_words="english", min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=-1, C=4.0)),
    ])
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    print("\n=== Classification report ===")
    print(classification_report(y_te, pred, target_names=["REAL", "FAKE"]))
    print("Confusion matrix:\n", confusion_matrix(y_te, pred))

    joblib.dump(pipe, OUT)
    print(f"\nSaved model to {OUT}")


if __name__ == "__main__":
    main()
