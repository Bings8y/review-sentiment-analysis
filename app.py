"""Streamlit demo: type a product review, get its sentiment and a confidence score.

    streamlit run app.py
"""
import json
import os

import pandas as pd
import streamlit as st

from sentiment.predict import ROOT, Predictor, available_models

COLORS = {"positive": "#1a7f4b", "neutral": "#9a6b00", "negative": "#b42318"}
EXAMPLES = [
    "Absolutely love this serum, my skin feels so soft and it arrived two days early!",
    "It's okay. Does the job but the scent is a bit strong and it's overpriced for the size.",
    "Not worth the money at all. The pump broke after a week and customer service never replied.",
    "I wouldn't say it's bad, but it's not great either.",
]

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load(name):
    return Predictor(name)


def test_scores():
    out = {}
    path = os.path.join(ROOT, "results", "classical.json")
    if os.path.exists(path):
        for k, v in json.load(open(path))["models"].items():
            out[k] = v
    path = os.path.join(ROOT, "results", "distilbert.json")
    if os.path.exists(path):
        out["DistilBERT"] = json.load(open(path))["metrics"]
    return out


st.title("💬 Review Sentiment Analyzer")
st.caption("Classifies an Amazon-style product review as positive, neutral or negative. "
           "23CS2028 Machine Learning Techniques · micro project")

models = available_models()
if not models:
    st.error("No trained model found. Run `python train_classical.py` (and optionally `python train_distilbert.py`).")
    st.stop()
name = st.sidebar.selectbox("Model", models)
st.sidebar.markdown("**Labels** come from star ratings: 1–2 ★ negative, 3 ★ neutral, 4–5 ★ positive.")

if "text" not in st.session_state:  # ?example=N opens with that example (handy for screenshots)
    st.session_state.text = EXAMPLES[int(st.query_params.get("example", 1)) - 1]
st.write("Try an example:")
cols = st.columns(len(EXAMPLES))
for i, (c, ex) in enumerate(zip(cols, EXAMPLES)):
    if c.button(f"Example {i + 1}", use_container_width=True):
        st.session_state.text = ex
text = st.text_area("Review text (Ctrl/⌘ + Enter to analyse)", key="text", height=130)

if text.strip():
    label, conf, probs = load(name).predict(text)
    st.markdown(f"<div style='padding:14px 18px;border-radius:10px;background:{COLORS[label]}1a;"
                f"border-left:6px solid {COLORS[label]}'><span style='font-size:1.6rem;font-weight:700;"
                f"color:{COLORS[label]}'>{label.upper()}</span>&nbsp;&nbsp;confidence {conf:.1%}</div>",
                unsafe_allow_html=True)
    st.write("")
    for k in ("positive", "neutral", "negative"):
        st.progress(probs[k], text=f"{k}: {probs[k]:.1%}")

scores = test_scores()
if scores:
    with st.expander("Model comparison on the held-out test set (3,000 reviews)"):
        st.dataframe(pd.DataFrame({k: {"Accuracy": v["accuracy"], "Macro F1": v["macro_f1"],
                                       "Neutral F1": v["per_class"]["neutral"]["f1"]}
                                   for k, v in scores.items()}).T.style.format("{:.3f}"))
