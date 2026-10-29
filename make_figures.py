"""Figures for the report, paper and slides, plus the error analysis (negation, sarcasm, imbalance).

    python make_figures.py            # writes results/figures/*.png and results/analysis.json
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from sentiment.metrics import LABELS, evaluate

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "results", "figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLASS_COLORS = {"negative": "#eb6834", "neutral": "#8a8984", "positive": "#2a78d6"}
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SHORT = {"Multinomial NB": "Naive Bayes", "Logistic Regression": "Log. Reg.", "Linear SVM": "Linear SVM",
         "Logistic Regression (balanced)": "Log. Reg.\n(balanced)", "Linear SVM (balanced)": "Linear SVM\n(balanced)",
         "DistilBERT": "DistilBERT"}
NEG_RE = r"\bnot\b|n't|\bnever\b|\bno\b"
# Sarcasm has no reliable surface marker, so use its symptom as a proxy: negative reviews that use praise words
PRAISE_RE = r"\b(?:great|love|loved|perfect|amazing|awesome|excellent|wonderful|best)\b"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


def bar_labels(ax, bars, fmt="{:.3f}", dy=0.006):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=8, color=INK)


def fig_dataset(df):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.3]})
    counts = df["label"].value_counts().reindex(LABELS)
    bars = a1.bar(LABELS, counts.values, color=[CLASS_COLORS[c] for c in LABELS], width=0.6)
    for b, v in zip(bars, counts.values):
        a1.text(b.get_x() + b.get_width() / 2, v + 300, f"{v:,}\n({v / len(df):.1%})", ha="center", fontsize=9)
    a1.set_ylim(0, counts.max() * 1.22)
    a1.set_ylabel("Reviews")
    a1.set_title("Sentiment classes (30,000 reviews)", fontsize=11, loc="left")
    a1.grid(axis="x", visible=False)
    words = df["review"].str.split().str.len().clip(upper=300)
    for c in LABELS:
        a2.hist(words[df.label == c], bins=np.arange(0, 305, 10), histtype="step", lw=2, color=CLASS_COLORS[c],
                density=True, label=f"{c} (median {int(words[df.label == c].median())})")
    a2.set_xlabel("Words per review (clipped at 300)")
    a2.set_ylabel("Density")
    a2.set_title("Review length", fontsize=11, loc="left")
    a2.legend(frameon=False, fontsize=9)
    save(fig, "dataset.png")


def fig_models(scores):
    names = list(scores)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.38
    b1 = ax.bar(x - w / 2, [scores[n]["accuracy"] for n in names], w - 0.03, color=BLUE, label="Accuracy")
    b2 = ax.bar(x + w / 2, [scores[n]["macro_f1"] for n in names], w - 0.03, color=ORANGE, label="Macro F1")
    bar_labels(ax, b1)
    bar_labels(ax, b2)
    ax.set_xticks(x, [SHORT[n] for n in names])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test score")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title("Accuracy hides the minority classes: macro F1 is the score that separates the models",
                 fontsize=11, loc="left")
    save(fig, "model_comparison.png")


def fig_per_class(scores):
    names = list(scores)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.27
    for k, c in enumerate(LABELS):
        ax.bar(x + (k - 1) * w, [scores[n]["per_class"][c]["f1"] for n in names], w - 0.03,
               color=CLASS_COLORS[c], label=c)
    ax.set_xticks(x, [SHORT[n] for n in names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.12))
    save(fig, "per_class_f1.png")


def fig_confusions(scores, pick):
    fig, axes = plt.subplots(1, len(pick), figsize=(4.1 * len(pick), 3.6))
    for ax, n in zip(np.atleast_1d(axes), pick):
        cm = np.array(scores[n]["confusion"])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
        ax.set_title(SHORT[n].replace("\n", " "), fontsize=11)
    fig.tight_layout()
    save(fig, "confusion_matrices.png")


def fig_ablation(ablation):
    names = list(ablation)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.38
    b1 = ax.bar(x - w / 2, [ablation[n]["macro_f1"] for n in names], w - 0.03, color=BLUE, label="All test reviews")
    b2 = ax.bar(x + w / 2, [ablation[n]["negation_subset_f1"] for n in names], w - 0.03, color=YELLOW,
                label=f"Reviews with a negation ({ablation[names[0]]['n_negation']:,})")
    bar_labels(ax, b1)
    bar_labels(ax, b2)
    ax.set_xticks(x, [n.replace(", ", "\n") for n in names])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Macro F1 (Log. Reg.)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "negation_ablation.png")


def fig_top_features(top):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, c in zip(axes, LABELS):
        feats = top[c][:10][::-1]
        ax.barh([f for f, _ in feats], [w for _, w in feats], color=CLASS_COLORS[c], height=0.65)
        ax.set_title(c, fontsize=11, loc="left")
        ax.set_xlabel("LR coefficient")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    save(fig, "top_features.png")


def fig_bert_curve(log):
    fig, ax = plt.subplots(figsize=(7, 3.3))
    s, l = np.array(log["step"]), np.array(log["loss"])
    top = l.max()
    ax.plot(s, l, color=GRID, lw=1)
    k = 10
    ax.plot(s[k - 1:], np.convolve(l, np.ones(k) / k, mode="valid"), color=BLUE, lw=2, label="training loss (moving avg.)")
    ax.set_ylim(0, top * 1.02)
    for e in log["epoch_val"]:
        ax.axvline(e["step"], color=INK2, lw=1, ls=":")
        ax.text(e["step"] - 8, top * 0.97, f"end of epoch {e['epoch']}\nval macro F1 {e['macro_f1']:.3f}",
                fontsize=8, va="top", ha="right", color=INK2)
    ax.set_xlabel("Optimiser step")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "distilbert_training.png")


def fig_pipeline():
    steps = [("1  Amazon reviews", "30k Beauty reviews,\nstar rating → label"),
             ("2  Preprocess", "clean, tokenise, lemmatise;\nkeep negation words"),
             ("3  Features", "TF-IDF, unigrams + bigrams\n60k terms"),
             ("4  Classifiers", "NB · Log. Reg. · Linear SVM\n+ fine-tuned DistilBERT"),
             ("5  Evaluate", "accuracy, macro F1, confusion\nmatrix, error analysis"),
             ("6  Deploy", "Streamlit app:\nlabel + confidence")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    pos = [(0.1, 2.3), (3.45, 2.3), (6.8, 2.3), (6.8, 0.1), (3.45, 0.1), (0.1, 0.1)]  # snake: row 1 L→R, row 2 R→L
    for (h, b), (x, y) in zip(steps, pos):
        ax.add_patch(FancyBboxPatch((x, y), 3.1, 1.75, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="#eef4fc", ec=BLUE, lw=1.6))
        ax.text(x + 1.55, y + 1.3, h, ha="center", va="center", fontsize=13, weight="bold", color=INK)
        ax.text(x + 1.55, y + 0.6, b, ha="center", va="center", fontsize=11, color=INK2)
    arrow = dict(arrowstyle="-|>", color=INK2, lw=1.8, mutation_scale=16)
    for (x0, y0), (x1, y1) in zip(pos[:-1], pos[1:]):
        if y0 == y1:
            a, b = (x0 + 3.15, y0 + 0.88), (x1 - 0.05, y1 + 0.88)
            if x1 < x0:
                a, b = (x0 - 0.05, y0 + 0.88), (x1 + 3.15, y1 + 0.88)
        else:
            a, b = (x0 + 1.55, y0 - 0.03), (x1 + 1.55, y1 + 1.8)
        ax.annotate("", b, a, arrowprops=arrow)
    save(fig, "pipeline.png")


def error_analysis(preds, models):
    neg = preds["review"].str.lower().str.contains(NEG_RE, regex=True)
    sar = (preds.label == "negative") & preds["review"].str.lower().str.contains(PRAISE_RE, regex=True)
    out = {"n_test": len(preds), "n_negation": int(neg.sum()), "n_praise_negative": int(sar.sum()), "models": {}}
    for m in models:
        out["models"][m] = {
            "all": evaluate(preds["label"], preds[m])["macro_f1"],
            "negation": evaluate(preds.label[neg], preds[m][neg])["macro_f1"],
            "negation_acc": round(float((preds.label[neg] == preds[m][neg]).mean()), 4),
            "no_negation_acc": round(float((preds.label[~neg] == preds[m][~neg]).mean()), 4),
            "praise_negative_acc": round(float((preds.label[sar] == preds[m][sar]).mean()), 4),
            "negative_acc": round(float((preds[m][preds.label == "negative"] == "negative").mean()), 4),
            "neutral_recall": evaluate(preds["label"], preds[m])["per_class"]["neutral"]["recall"],
        }
    # Illustrative failures: short reviews that the best classical model gets wrong
    best_cls = "Logistic Regression (balanced)"
    wrong = preds[(preds[best_cls] != preds.label) & (preds.review.str.len() < 160)]
    ex = []
    for kind, mask in (("negation", wrong.review.str.lower().str.contains(NEG_RE, regex=True)),
                       ("praise-in-negative", (wrong.label == "negative") & wrong.review.str.lower().str.contains(PRAISE_RE, regex=True)),
                       ("neutral", wrong.label == "neutral")):
        for _, r in wrong[mask].head(4).iterrows():
            ex.append(dict(kind=kind, review=r.review, label=r.label, lr=r[best_cls],
                           bert=r.get("DistilBERT", None)))
    out["examples"] = ex
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(os.path.join(ROOT, "data", "reviews.csv"))
    cl = json.load(open(os.path.join(ROOT, "results", "classical.json")))
    scores = dict(cl["models"])
    bert_path = os.path.join(ROOT, "results", "distilbert.json")
    bert = json.load(open(bert_path)) if os.path.exists(bert_path) else None
    if bert:
        scores["DistilBERT"] = bert["metrics"]

    fig_dataset(df)
    fig_models(scores)
    fig_per_class(scores)
    fig_confusions(scores, ["Logistic Regression (balanced)"] + (["DistilBERT"] if bert else ["Linear SVM"]))
    fig_ablation(cl["ablation"])
    fig_top_features(cl["top_features"])
    fig_pipeline()
    if bert:
        fig_bert_curve(bert["log"])

    preds = pd.read_csv(os.path.join(ROOT, "results", "test_predictions.csv"), keep_default_na=False)
    analysis = error_analysis(preds, [m for m in scores if m in preds.columns])
    json.dump(analysis, open(os.path.join(ROOT, "results", "analysis.json"), "w"), indent=1)
    for m, v in analysis["models"].items():
        print(f"{m:32s} {v}")


if __name__ == "__main__":
    main()
