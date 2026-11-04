"""Звіт для вебзастосунку: статистика даних, порівняння моделей, важливість ознак."""
from datetime import datetime

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report

from . import config as C
from .evaluate import score_from_proba

SCORE_EDGES = list(range(300, 851, 50))  # 11 інтервалів


def dataset_stats(X, y, groups) -> dict:
    counts = y.value_counts().reindex(range(len(C.CLASSES))).fillna(0).astype(int)
    labels_per_customer = y.groupby(groups).nunique()
    quality = sorted(
        ({"feature": c, "valid_pct": round(float(100 * X[c].notna().mean()), 1)} for c in C.VALID_RANGES),
        key=lambda d: d["valid_pct"],
    )
    return {
        "rows": int(len(X)),
        "customers": int(groups.nunique()),
        "rows_per_customer": int(len(X) / groups.nunique()),
        "classes": [{"name": n, "rows": int(counts[i]), "pct": round(100 * counts[i] / len(y), 1)}
                    for i, n in enumerate(C.CLASSES)],
        "changing_label_pct": round(float(100 * (labels_per_customer > 1).mean()), 1),
        "quality": quality,
    }


def best_model_stats(model, X_te, y_te) -> dict:
    proba = model.predict_proba(X_te)
    pred = proba.argmax(axis=1)
    rep = classification_report(y_te, pred, target_names=C.CLASSES, output_dict=True, zero_division=0)
    per_class = [{"name": n, **{k: round(float(rep[n][k]), 3) for k in ("precision", "recall", "f1-score")},
                  "support": int(rep[n]["support"])} for n in C.CLASSES]

    scores = score_from_proba(proba)
    y_arr = np.asarray(y_te)
    hist = {n: np.histogram(scores[y_arr == i], bins=SCORE_EDGES)[0].astype(int).tolist()
            for i, n in enumerate(C.CLASSES)}

    sample = np.random.default_rng(C.RANDOM_STATE).choice(len(X_te), min(2500, len(X_te)), replace=False)
    imp = permutation_importance(model, X_te.iloc[sample], y_te.iloc[sample], scoring="f1_macro",
                                 n_repeats=3, random_state=C.RANDOM_STATE)
    importance = sorted(({"feature": f, "drop": round(float(m), 4)}
                         for f, m in zip(X_te.columns, imp.importances_mean)),
                        key=lambda d: -d["drop"])[:12]
    return {"per_class": per_class, "score_hist": {"edges": SCORE_EDGES, "counts": hist},
            "importance": importance}


def build(results, best, model, X, y, groups, X_te, y_te, n_train) -> dict:
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "best_model": best, "train_rows": int(n_train), "test_rows": int(len(X_te)),
        "metrics": results,
        "dataset": dataset_stats(X, y, groups),
        "best": best_model_stats(model, X_te, y_te),
    }
