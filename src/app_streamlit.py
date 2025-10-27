# import streamlit as st
# from pathlib import Path
# import joblib
# import numpy as np

# st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# st.title("📰 Fake News Detector")
# st.write("Paste a headline or article below to classify it as **FAKE**, **REAL**, or **UNCERTAIN**.")

# MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
# model_path = MODEL_DIR / "news_classifier.joblib"

# if not model_path.exists():
#     st.warning("⚠️ Model not found. Please run `python src/train.py` first to train and save the model.")
#     st.stop()

# @st.cache_resource
# def load_model():
#     return joblib.load(model_path)

# pipeline = load_model()

# THRESHOLD = 0.60  # 60% confidence required

# text = st.text_area("News text", height=200, placeholder="e.g., Government announces new policy to ...")

# if st.button("Classify"):
#     if not text.strip():
#         st.error("Please enter some text.")
#     else:
#         pred = pipeline.predict([text])[0]
#         conf = None
#         probs = None

#         if hasattr(pipeline.named_steps["clf"], "predict_proba"):
#             probs = pipeline.predict_proba([text])[0]
#             classes = pipeline.classes_
#             max_conf = np.max(probs)
#             pred = classes[np.argmax(probs)]
#             conf = float(max_conf) * 100.0

#             if max_conf < THRESHOLD:
#                 pred = "UNCERTAIN / Needs more context"

#         st.subheader("Result")
#         if conf is not None:
#             st.success(f"**{pred}** (confidence ~{conf:.1f}%)")
#         else:
#             st.success(f"**{pred}**")


import streamlit as st
from pathlib import Path
import joblib
import numpy as np

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector (BERT Enhanced)")
st.write("Paste a headline or article below to classify it as **FAKE**, **REAL**, or **UNCERTAIN**.")

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
model_path = MODEL_DIR / "news_classifier_bert.joblib"   # 👉 use the new BERT model file

if not model_path.exists():
    st.warning("⚠️ BERT model not found. Please run `python src/train_bert.py` first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model_data = load_model()
bert = model_data["bert"]
clf = model_data["clf"]

THRESHOLD = 0.65  # 65% confidence threshold

text = st.text_area("News text", height=200, placeholder="e.g., Government announces new policy to ...")

if st.button("Classify"):
    if not text.strip():
        st.error("Please enter some text.")
    else:
        # Encode input text
        emb = bert.encode([text])
        probs = clf.predict_proba(emb)[0]
        classes = clf.classes_
        pred = classes[np.argmax(probs)]
        conf = np.max(probs) * 100

        if conf < (THRESHOLD * 100):
            pred = "UNCERTAIN / Needs more context"

        # --- UI Output ---
        st.subheader("Result")

        if "FAKE" in pred.upper():
            st.error(f"🛑 **{pred}** (confidence ~{conf:.1f}%)")
        elif "REAL" in pred.upper():
            st.success(f"✅ **{pred}** (confidence ~{conf:.1f}%)")
        else:
            st.warning(f"⚠️ **{pred}** (confidence ~{conf:.1f}%)")

        st.write("")  # spacing

        # Confidence progress bar
        st.write("Confidence Level:")
        st.progress(int(conf))
