import streamlit as st
from pathlib import Path
import joblib
import numpy as np

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.write("Paste a headline or article below to classify it as **FAKE**, **REAL**, or **UNCERTAIN**.")

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
model_path = MODEL_DIR / "news_classifier.joblib"

if not model_path.exists():
    st.warning("⚠️ Model not found. Please run `python src/train.py` first to train and save the model.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(model_path)

pipeline = load_model()

THRESHOLD = 0.60  # 60% confidence required

text = st.text_area("News text", height=200, placeholder="e.g., Government announces new policy to ...")

if st.button("Classify"):
    if not text.strip():
        st.error("Please enter some text.")
    else:
        pred = pipeline.predict([text])[0]
        conf = None
        probs = None

        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            probs = pipeline.predict_proba([text])[0]
            classes = pipeline.classes_
            max_conf = np.max(probs)
            pred = classes[np.argmax(probs)]
            conf = float(max_conf) * 100.0

            if max_conf < THRESHOLD:
                pred = "UNCERTAIN / Needs more context"

        st.subheader("Result")
        if conf is not None:
            st.success(f"**{pred}** (confidence ~{conf:.1f}%)")
        else:
            st.success(f"**{pred}**")
