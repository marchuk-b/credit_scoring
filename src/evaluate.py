import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from . import config as C


def score_from_proba(proba: np.ndarray) -> np.ndarray:
    # Чим більша ймовірність Good, тим вищий скор.
    p = np.asarray(proba)
    return C.SCORE_MIN + C.SCORE_RANGE * (p[:, 2] + 0.5 * p[:, 1])


def ks_poor_vs_rest(y_true, p_poor) -> float:
    # KS-статистика для виявлення клієнтів класу Poor.
    y = (np.asarray(y_true) == 0).astype(int)
    order = np.argsort(-np.asarray(p_poor))
    y = y[order]
    tpr = np.cumsum(y) / max(y.sum(), 1)
    fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(tpr - fpr)))


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)
    pred = proba.argmax(axis=1)
    auc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "roc_auc_ovr": float(auc),
        "gini_poor_vs_rest": float(2 * roc_auc_score((y == 0).astype(int), proba[:, 0]) - 1),
        "ks_poor_vs_rest": ks_poor_vs_rest(y, proba[:, 0]),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
