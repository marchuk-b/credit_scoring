import json
import time

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from . import config as C
from . import data_processing as dp
from .pipeline import _preprocessor, build


def candidates() -> dict:
    models = {"logreg": build("logreg"), "gboost": build("gboost"),
              "rf": Pipeline([("prep", _preprocessor()),
                              ("model", RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                               random_state=C.RANDOM_STATE))])}
    try:
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
        models["lgb"] = Pipeline([("prep", _preprocessor()),
                                  ("model", LGBMClassifier(n_estimators=1000, random_state=C.RANDOM_STATE, verbose=-1))])
        models["xgb"] = Pipeline([("prep", _preprocessor()),
                                  ("model", XGBClassifier(n_estimators=1000, random_state=C.RANDOM_STATE, n_jobs=-1))])
    except ImportError:
        print("lightgbm/xgboost не встановлені — ці моделі пропущено")
    return models


def cross_validate(X_tr, y_tr, g_tr, n_splits: int = 5) -> dict:
    # Повертає {модель: {auc_mean, auc_std, f1_mean, f1_std, seconds}} за навчальною частиною
    folds = list(GroupKFold(n_splits).split(X_tr, y_tr, g_tr))
    report = {}
    for name, model in candidates().items():
        aucs, f1s, t0 = [], [], time.time()
        for tr, va in folds:
            est, ytr = clone(model), y_tr.iloc[tr]
            fit_params = {}
            if name in ("rf", "lgb", "xgb"):  # баланс класів через sample_weight, як в ансамблі
                fit_params["model__sample_weight"] = compute_sample_weight("balanced", ytr)
            est.fit(X_tr.iloc[tr], ytr, **fit_params)
            p = est.predict_proba(X_tr.iloc[va])
            aucs.append(roc_auc_score(y_tr.iloc[va], p, multi_class="ovr", average="macro"))
            f1s.append(f1_score(y_tr.iloc[va], p.argmax(1), average="macro"))
        report[name] = {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                        "seconds": round(time.time() - t0)}
        r = report[name]
        print(f"  CV {name:7s} AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f}  F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}  ({r['seconds']} с)")
    return report


def save(report: dict) -> None:
    C.MODELS_DIR.mkdir(exist_ok=True)
    (C.MODELS_DIR / "cv_report.json").write_text(json.dumps(report, indent=2))


def main(n_splits: int = 5):
    X, y, groups = dp.load_train()
    X_tr, _, y_tr, _, g_tr = dp.split(X, y, groups)  # та сама навчальна частина, що й у train.py
    save(cross_validate(X_tr, y_tr, g_tr, n_splits))


if __name__ == "__main__":
    main()