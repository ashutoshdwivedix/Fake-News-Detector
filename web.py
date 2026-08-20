"""
app.py — Streamlit frontend for the Fake News Detection model.

Setup:
    pip install streamlit joblib scikit-learn xgboost nltk

Run:
    streamlit run app.py

Make sure 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl'
are in the same folder as this script.
"""

import re
import string

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- NLTK setup (cached so it only runs once per session) ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Same cleaning function used during training — must match exactly."""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)


@st.cache_resource
def load_model():
    """Load model + vectorizer once and cache across reruns."""
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer


def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    label = "Real" if pred == 1 else "Fake"

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]

    return label, proba


# ---------------- UI ----------------

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.write(
    "Paste a news headline and/or article text below. "
    "The model will predict whether it's likely **Fake** or **Real**."
)

try:
    model, vectorizer = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error(
        "Model files not found. Make sure `fake_news_model.pkl` and "
        "`tfidf_vectorizer.pkl` are in the same folder as `app.py`."
    )

headline = st.text_input("Headline (optional)")
body = st.text_area("Article text", height=250, placeholder="Paste the article content here...")

analyze = st.button("Analyze", type="primary", disabled=not model_loaded)

if analyze:
    combined = f"{headline} {body}".strip()
    if not combined:
        st.warning("Please enter a headline or article text first.")
    else:
        with st.spinner("Analyzing..."):
            label, proba = predict(combined, model, vectorizer)

        if label == "Real":
            st.success(f"### Prediction: {label} ✅")
        else:
            st.error(f"### Prediction: {label} ❌")

        if proba is not None:
            col1, col2 = st.columns(2)
            col1.metric("Confidence: Fake", f"{proba[0]:.1%}")
            col2.metric("Confidence: Real", f"{proba[1]:.1%}")
            st.progress(float(proba[1]))

st.divider()
st.caption(
    "This tool uses a TF-IDF + XGBoost model trained on the "
    "Fake and Real News dataset. Predictions are statistical, not verified fact-checks — "
    "always cross-check important claims with trusted sources."
)