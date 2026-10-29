"""Shared evaluation: accuracy, macro / weighted precision-recall-F1, per-class scores, confusion matrix."""
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

LABELS = ["negative", "neutral", "positive"]


def evaluate(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    wf = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)[2]
    return dict(
        accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
        macro_precision=round(float(mp), 4), macro_recall=round(float(mr), 4), macro_f1=round(float(mf), 4),
        weighted_f1=round(float(wf), 4),
        per_class={c: dict(precision=round(float(p[i]), 4), recall=round(float(r[i]), 4), f1=round(float(f[i]), 4),
                           support=int(s[i])) for i, c in enumerate(LABELS)},
        confusion=confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    )
