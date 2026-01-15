import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

DATA_PATH = "data/dataset.csv"
OUT_PATH = "artifacts/model.joblib"

def main():
    df = pd.read_csv(DATA_PATH)

    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df = df.dropna(subset=["queue"]).copy()

    df["text"] = (df["subject"].astype(str) + "\n" + df["body"].astype(str)).str.strip()
    df = df[df["text"].str.len() >= 5].copy()

    X = df["text"].values
    y = df["queue"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print("macro-F1:", f1_score(y_test, pred, average="macro"))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    joblib.dump(pipe, OUT_PATH)

if __name__ == "__main__":
    main()
