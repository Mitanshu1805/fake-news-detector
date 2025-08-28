# Fake News Detector — Mini Project (Step-by-Step)

This is a beginner-friendly, **ready-to-run** Fake News Detection project using Python, NLP, and Machine Learning.
You'll train a model on a dataset (Fake.csv, Real.csv) and then classify any news text as **FAKE** or **REAL**.

---

## 0) Project Structure
```
fake-news-detector/
├── dataset/
│   ├── Fake.csv        # put here
│   └── Real.csv        # put here
├── models/
│   └── news_classifier.joblib  # auto-created after training
├── src/
│   ├── train.py
│   ├── predict_cli.py
│   └── app_streamlit.py
├── requirements.txt
└── README.md
```

---

## 1) Setup (Windows PowerShell)
```powershell
cd fake-news-detector

# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Mac/Linux (bash/zsh):**
```bash
cd fake-news-detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2) Add Dataset
Download the *Fake and Real News* dataset (or any similar dataset) and place the two files in `dataset/` as:
- `dataset/Fake.csv`
- `dataset/Real.csv`

> Expected columns: either a `text` column, or `title` + `text`. If your CSVs only have `title`, it's still fine—
the training script will combine `title` and `text` when available.

---

## 3) Train the Model
```bash
python src/train.py
```
This will:
- Load and combine the two CSVs
- Split into train/test (stratified)
- Build a TF-IDF + Logistic Regression pipeline
- Print metrics (accuracy, classification report, confusion matrix)
- Save the trained pipeline to `models/news_classifier.joblib`

---

## 4) Test in the Terminal (CLI)
```bash
python src/predict_cli.py
```
Type/paste a news headline or short article. Type `exit` to quit.

---

## 5) Optional: Simple Web UI (Streamlit)
```bash
streamlit run src/app_streamlit.py
```
Then open the local URL in your browser. Paste text and see the prediction + confidence.

---

## Notes
- The model uses class balancing and bi-grams to avoid predicting the same label always.
- If you're getting poor accuracy, try adding more data, or switching to a different algorithm.
- This project keeps preprocessing simple (no NLTK required). TF-IDF handles tokenization and stop-words.

Good luck with your mini-project! 💪
