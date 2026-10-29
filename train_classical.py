"""TF-IDF + classical classifiers: Multinomial Naive Bayes, Logistic Regression and linear SVM.

Hyperparameters are chosen on the validation split by macro-F1; the test split is used once at the end.
Logistic Regression and SVM are trained with and without class weighting to study class imbalance.

    python train_classical.py         # writes results/classical.json, results/test_predictions.csv, models/
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sentiment.metrics import evaluate
from sentiment.preprocess import clean_standard, clean_text

ROOT = os.path.dirname(os.path.abspath(__file__))
LABELS = ["negative", "neutral", "positive"]


def load_clean():
    cache = os.path.join(ROOT, "data", "reviews_clean.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, keep_default_na=False)
    df = pd.read_csv(os.path.join(ROOT, "data", "reviews.csv"))
    t = time.time()
    df["clean"] = df["review"].map(clean_text)
    df["clean_std"] = df["review"].map(clean_standard)
    print(f"preprocessed {len(df):,} reviews in {time.time() - t:.1f}s")
    df.to_csv(cache, index=False)
    return df


def tfidf(ngram=(1, 2)):
    return TfidfVectorizer(ngram_range=ngram, min_df=2, max_df=0.9, sublinear_tf=True, max_features=100_000)


def candidates():
    for a in (0.01, 0.03, 0.05, 0.1, 0.3, 1.0):
        yield "Multinomial NB", dict(alpha=a), lambda a=a: MultinomialNB(alpha=a)
    for cw in (None, "balanced"):
        name = "Logistic Regression" + (" (balanced)" if cw else "")
        for c in (1.0, 4.0, 10.0, 30.0):
            yield name, dict(C=c), lambda c=c, cw=cw: LogisticRegression(C=c, max_iter=2000, class_weight=cw)
        name = "Linear SVM" + (" (balanced)" if cw else "")
        for c in (0.1, 0.3, 1.0):
            yield name, dict(C=c), lambda c=c, cw=cw: LinearSVC(C=c, class_weight=cw)


def main():
    df = load_clean()
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)

    vec = tfidf()
    t = time.time()
    Xtr = vec.fit_transform(tr["clean"])
    vec_time = time.time() - t
    Xva, Xte = vec.transform(va["clean"]), vec.transform(te["clean"])
    print(f"TF-IDF: {Xtr.shape[1]:,} features (uni+bigrams), fit in {vec_time:.1f}s")

    # ---- model selection on the validation split
    best = {}
    for name, params, make in candidates():
        clf = make().fit(Xtr, tr["label"])
        f1 = evaluate(va["label"], clf.predict(Xva))["macro_f1"]
        print(f"  {name:32s} {params}  val macro-F1 = {f1:.4f}")
        if name not in best or f1 > best[name][0]:
            best[name] = (f1, params, make)

    # ---- refit the chosen configuration of each family and score it once on test
    results, preds = {}, pd.DataFrame({"review": te["review"].values, "label": te["label"].values})
    for name, (val_f1, params, make) in best.items():
        t = time.time()
        clf = make().fit(Xtr, tr["label"])
        train_s = time.time() - t
        t = time.time()
        yp = clf.predict(Xte)
        infer_ms = (time.time() - t) / len(te) * 1000
        m = evaluate(te["label"], yp)
        m.update(params=params, val_macro_f1=val_f1, train_seconds=round(train_s, 2),
                 infer_ms_per_review=round(infer_ms, 4))
        results[name] = m
        preds[name] = yp
        print(f"TEST {name:32s} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")
        if name == "Logistic Regression":
            joblib.dump(Pipeline([("tfidf", vec), ("clf", clf)]), os.path.join(ROOT, "models", "tfidf_logreg.joblib"))
        if name == "Logistic Regression (balanced)":
            joblib.dump(Pipeline([("tfidf", vec), ("clf", clf)]),
                        os.path.join(ROOT, "models", "tfidf_logreg_balanced.joblib"))
        if name == "Linear SVM":
            # LinearSVC has no probabilities; calibrate it so the app can report a confidence score
            cal = CalibratedClassifierCV(LinearSVC(**{**params, "class_weight": None}), cv=3).fit(Xtr, tr["label"])
            joblib.dump(Pipeline([("tfidf", vec), ("clf", cal)]), os.path.join(ROOT, "models", "tfidf_svm.joblib"))

    # ---- ablations with Logistic Regression: stopword handling and n-gram range
    lr_c = best["Logistic Regression"][1]["C"]
    ablation = {}
    for label, col, ngram in [("Standard stopwords, unigrams", "clean_std", (1, 1)),
                              ("Standard stopwords, uni+bigrams", "clean_std", (1, 2)),
                              ("Negation-aware, unigrams", "clean", (1, 1)),
                              ("Negation-aware, uni+bigrams", "clean", (1, 2))]:
        v = tfidf(ngram)
        clf = LogisticRegression(C=lr_c, max_iter=2000).fit(v.fit_transform(tr[col]), tr["label"])
        m = evaluate(te["label"], clf.predict(v.transform(te[col])))
        neg_mask = te["review"].str.lower().str.contains(r"\bnot\b|n't|\bnever\b|\bno\b", regex=True).values
        m_neg = evaluate(te["label"][neg_mask], clf.predict(v.transform(te[col][neg_mask])))
        ablation[label] = dict(accuracy=m["accuracy"], macro_f1=m["macro_f1"], negation_subset_acc=m_neg["accuracy"],
                               negation_subset_f1=m_neg["macro_f1"], n_negation=int(neg_mask.sum()),
                               n_features=len(v.vocabulary_))
        print(f"ABLATION {label:34s} acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f} "
              f"| negation subset ({neg_mask.sum()}) F1={m_neg['macro_f1']:.4f}")

    # top weighted n-grams per class (Logistic Regression)
    lr = joblib.load(os.path.join(ROOT, "models", "tfidf_logreg.joblib"))
    names = np.array(lr["tfidf"].get_feature_names_out())
    coef = lr["clf"].coef_
    top = {c: [(names[i], round(float(coef[k, i]), 3)) for i in np.argsort(coef[k])[::-1][:15]]
           for k, c in enumerate(lr["clf"].classes_)}

    out = dict(n_features=int(Xtr.shape[1]), vectorizer_fit_seconds=round(vec_time, 2),
               split_sizes={s: int((df.split == s).sum()) for s in ("train", "val", "test")},
               models=results, ablation=ablation, top_features=top)
    json.dump(out, open(os.path.join(ROOT, "results", "classical.json"), "w"), indent=1)
    preds.to_csv(os.path.join(ROOT, "results", "test_predictions.csv"), index=False)
    print("wrote results/classical.json")


if __name__ == "__main__":
    main()
