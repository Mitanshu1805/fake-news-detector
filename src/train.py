import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    fake_path = DATA_DIR / "Fake.csv"
    real_path = DATA_DIR / "Real.csv"
    short_path = DATA_DIR / "short_dataset.csv"

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)
    short_df = pd.read_csv(short_path)

    # Label the data
    fake_df["label"] = "FAKE"
    real_df["label"] = "REAL"

    def build_text(df):
        cols = [c.lower() for c in df.columns]
        title_col = "title" if "title" in cols else None
        text_col = "text" if "text" in cols else None

        if title_col and text_col:
            df["input_text"] = df["title"].astype(str) + " " + df["text"].astype(str)
        elif title_col:
            df["input_text"] = df["title"].astype(str)
        elif text_col:
            df["input_text"] = df["text"].astype(str)
        else:
            raise ValueError("Dataset must have either 'title' or 'text' column")
        return df

    fake_df = build_text(fake_df)
    real_df = build_text(real_df)
    short_df["input_text"] = short_df["text"]

    df = pd.concat([fake_df, real_df, short_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def main():
    df = load_dataset()
    print("Dataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    X = df["input_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams
            max_df=0.9,
            min_df=3,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    out_path = MODEL_DIR / "news_classifier.joblib"
    joblib.dump(pipeline, out_path)
    print(f"\n✅ Saved model to {out_path}")

if __name__ == "__main__":
    main()
 