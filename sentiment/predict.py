"""Load a trained model and predict the sentiment of raw review text (used by app.py)."""
import os

import joblib
import numpy as np

from sentiment.metrics import LABELS
from sentiment.preprocess import clean_text, light_clean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {
    "DistilBERT (fine-tuned)": os.path.join(ROOT, "models", "distilbert-sentiment"),
    "TF-IDF + Logistic Regression (balanced)": os.path.join(ROOT, "models", "tfidf_logreg_balanced.joblib"),
    "TF-IDF + Linear SVM": os.path.join(ROOT, "models", "tfidf_svm.joblib"),
}


def available_models():
    return [name for name, path in MODELS.items() if os.path.exists(path)]


class Predictor:
    def __init__(self, name):
        self.name, path = name, MODELS[name]
        self.is_bert = os.path.isdir(path)
        if self.is_bert:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.torch = torch
            self.tok = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()  # CPU is plenty for one review
        else:
            self.pipe = joblib.load(path)

    def predict_proba(self, texts):
        """Probabilities for each text, columns in LABELS order."""
        if self.is_bert:
            enc = self.tok([light_clean(t) for t in texts], truncation=True, max_length=128, padding=True,
                           return_tensors="pt")
            with self.torch.no_grad():
                return self.torch.softmax(self.model(**enc).logits, -1).numpy()
        proba = self.pipe.predict_proba([clean_text(t) for t in texts])
        order = [list(self.pipe.classes_).index(c) for c in LABELS]
        return proba[:, order]

    def predict(self, text):
        p = self.predict_proba([text])[0]
        k = int(np.argmax(p))
        return LABELS[k], float(p[k]), dict(zip(LABELS, p.round(4).tolist()))
