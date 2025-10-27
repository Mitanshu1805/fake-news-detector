import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Dataset
fake_path = DATA_DIR / "Fake.csv"
real_path = DATA_DIR / "Real.csv"

fake_df = pd.read_csv(fake_path)
real_df = pd.read_csv(real_path)

fake_df["label"] = "FAKE"
real_df["label"] = "REAL"

# combine title + text for better generalization
def combine_text(df):
    title = df["title"].fillna("")
    text = df["text"].fillna("")
    return (title + " " + text).str.strip()

fake_df["input_text"] = combine_text(fake_df)
real_df["input_text"] = combine_text(real_df)

df = pd.concat([fake_df, real_df], ignore_index=True)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df["input_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 3. Load pre-trained BERT model (mini version)
print("Loading sentence-transformer model...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # light and fast

# 4. Convert sentences to embeddings
print("Encoding text (this may take a few minutes)...")
X_train_emb = bert_model.encode(X_train.tolist(), batch_size=32, show_progress_bar=True)
X_test_emb = bert_model.encode(X_test.tolist(), batch_size=32, show_progress_bar=True)

# 5. Train Logistic Regression on embeddings
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
print("Training classifier...")
clf.fit(X_train_emb, y_train)

# 6. Evaluate
y_pred = clf.predict(X_test_emb)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save model + embedding model
out_path = MODEL_DIR / "news_classifier_bert.joblib"
joblib.dump({"bert": bert_model, "clf": clf}, out_path)
print(f"\n✅ Saved model to {out_path}")
