import argparse
import json

import joblib

from . import config as C
from . import data_processing as dp
from .evaluate import evaluate
from .report import build as build_report
from .pipeline import build


def main(use_ensemble: bool = True):
    X, y, groups = dp.load_train()
    X_tr, X_te, y_tr, y_te, g_tr = dp.split(X, y, groups)
    print(f"train={len(X_tr):,} test={len(X_te):,} (розбиття за Customer_ID)")

    results, models = {}, {}
    names = ["logreg", "gboost"] + (["ensemble"] if use_ensemble else [])
    for name in names:
        if name == "ensemble":  # групи потрібні для крос-валідації ваг
            models[name] = build(name).fit(X_tr, y_tr, model__groups=g_tr)
        else:
            models[name] = build(name).fit(X_tr, y_tr)
        results[name] = evaluate(models[name], X_te, y_te)
        if name == "ensemble":
            est = models[name].named_steps["model"]
            results[name]["weights"], results[name]["cv_auc"] = est.weights_, est.cv_auc_
            print("  ваги ансамблю:", {k: round(v, 3) for k, v in est.weights_.items()})
        r = results[name]
        print(f"{name:8s} acc={r['accuracy']:.3f} f1={r['f1_macro']:.3f} "
              f"auc={r['roc_auc_ovr']:.3f} gini={r['gini_poor_vs_rest']:.3f} ks={r['ks_poor_vs_rest']:.3f}")

    best = max(results, key=lambda n: results[n]["roc_auc_ovr"])
    print("Найкраща модель:", best)

    # Базові значення для пояснень: медіана / мода по навчальній вибірці
    baseline = {c: X_tr[c].median() for c in C.NUMERIC}
    baseline.update({c: X_tr[c].mode().iloc[0] for c in C.CATEGORICAL})

    C.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": models[best], "name": best, "baseline": baseline}, C.MODEL_PATH)
    report = build_report(results, best, models[best], X, y, groups, X_te, y_te, len(X_tr))
    C.REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ensemble", action="store_true", help="пропустити довге навчання ансамблю")
    main(use_ensemble=not parser.parse_args().no_ensemble)
