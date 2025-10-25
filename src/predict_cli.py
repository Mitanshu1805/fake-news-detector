from pathlib import Path
import joblib
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
model_path = MODEL_DIR / "news_classifier.joblib"

if not model_path.exists():
    raise SystemExit(f"⚠️ Model file not found at {model_path}. Run 'python src/train.py' first.")

print(f"Loading model from: {model_path}")
pipeline = joblib.load(model_path)

THRESHOLD = 0.60  # 60% confidence required

def classify(text: str):
    clf = pipeline.named_steps.get("clf", None)
    pred = pipeline.predict([text])[0]

    probs = {}
    if clf is not None and hasattr(clf, "predict_proba"):
        raw_probs = pipeline.predict_proba([text])[0]
        classes = list(clf.classes_)
        probs = {cls: float(prob)*100 for cls, prob in zip(classes, raw_probs)}

        max_conf = max(raw_probs)
        pred = classes[np.argmax(raw_probs)]
        if max_conf < THRESHOLD:
            pred = "UNCERTAIN / Needs more context"

    return pred, probs

print("\n🔍 Fake News Detector — CLI")
print("Type some news text and press Enter. Type 'exit' to quit.\n")

while True:
    user_input = input("> ")
    if user_input.strip().lower() == "exit":
        break
    if not user_input.strip():
        print("Please type something (or 'exit').")
        continue

    pred, probs = classify(user_input.strip())
    print(f"🧾 Prediction: {pred}")
    if probs:
        for label, val in probs.items():
            print(f"   - {label}: {val:.1f}%")
    print()
