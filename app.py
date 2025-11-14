# app.py
import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------------
# Ensure necessary NLTK data is present (will be no-op if already downloaded)
import nltk

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")

nltk_download_done = False
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk_download_done = True
except Exception:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk_download_done = True

# -------------------------
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# -------------------------
ARTIFACT_DIR = Path(".")
VECT_PATH = ARTIFACT_DIR / "vectorizer.joblib"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
LABELS_PATH = ARTIFACT_DIR / "label_columns.joblib"  # optional file if you saved labels


@st.cache_resource
def load_vectorizer(path):
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_resource
def load_model(path):
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_resource
def load_label_columns(path):
    if path.exists():
        return joblib.load(path)
    return None

vectorizer = load_vectorizer(VECT_PATH)
model = load_model(MODEL_PATH)
label_columns = load_label_columns(LABELS_PATH)

st.title("Abstract → Topic(s) Predictor")
st.write("Paste an abstract below and press Predict. (Simple UI — top-K predictions shown.)")

if vectorizer is None or model is None:
    st.error("Model or vectorizer not found. Make sure `vectorizer.joblib` and `model.joblib` exist in the app folder.")
    st.stop()

# Try to infer label names
inferred_labels = None
# 1) If you provided saved label_columns list
if label_columns is not None:
    inferred_labels = list(label_columns)
else:
    # 2) Try to get something useful from the model
    # For multi-label OneVsRestClassifier trained with DataFrame, sometimes `classes_` exists on estimator or model
    try:
        if hasattr(model, "classes_"):
            inferred_labels = list(model.classes_)
    except Exception:
        inferred_labels = None

if inferred_labels is None:
    # Fallback to numeric labels
    # Try to infer number of labels from predict_proba shape
    try:
        sample_probs = model.predict_proba(vectorizer.transform(["test"]))  # shape depends on model
        if isinstance(sample_probs, np.ndarray):
            n_labels = sample_probs.shape[-1]
            inferred_labels = [f"label_{i}" for i in range(n_labels)]
        else:
            inferred_labels = ["label_0"]
    except Exception:
        inferred_labels = ["label_0"]

top_k = st.sidebar.slider("Top K", min_value=1, max_value=min(10, len(inferred_labels)), value=3)

summary = st.text_area("Enter abstract / summary here", height=200)

if st.button("Predict"):
    if not summary.strip():
        st.warning("Please enter some text.")
    else:
        transformed = transform_text(summary)
        vect = vectorizer.transform([transformed])

        # If model supports predict_proba, use probabilities and show top-k
        try:
            probs = model.predict_proba(vect)
            # predict_proba may return list / array depending on model; handle common cases
            if isinstance(probs, list):
                # some multilabel wrappers return a list of arrays (one per class); convert
                probs = np.array([p[0] if p.shape[0] == 1 else p for p in probs]).T
            probs = np.asarray(probs)
            # If probs is 2D array: (n_samples, n_labels) — take first sample
            if probs.ndim == 2:
                probs = probs[0]
            # get top-k indices
            top_indices = probs.argsort()[-top_k:][::-1]
            top_labels = [inferred_labels[i] if i < len(inferred_labels) else f"label_{i}" for i in top_indices]
            top_probs = [float(probs[i]) for i in top_indices]

            df = pd.DataFrame({"label": top_labels, "probability": top_probs})
            st.subheader("Top predictions")
            st.table(df)
        except Exception:
            # fallback to predict()
            preds = model.predict(vect)
            # If preds is 2D multilabel binary matrix, pick indices with 1
            preds = np.asarray(preds)
            if preds.ndim == 2:
                present = np.where(preds[0] == 1)[0]
                if present.size == 0:
                    st.info("No labels predicted (model predicted all zeros).")
                else:
                    labels = [inferred_labels[i] for i in present]
                    st.write("Predicted labels:", labels)
            else:
                # single-label classifier
                label_idx = int(preds[0])
                label = inferred_labels[label_idx] if label_idx < len(inferred_labels) else str(label_idx)
                st.write("Predicted label:", label)


st.markdown("---")
st.caption("Note: If you saved a file named `label_columns.joblib` containing the list(y.columns) used during training, the app will display those human-readable labels. Otherwise labels may be numeric.")
