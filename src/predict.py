from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import data_processing as dp
from .evaluate import score_from_proba

DECISIONS = {"Good": "Схвалено", "Standard": "На ручний розгляд", "Poor": "Відмова"}

N_PERMUTATIONS = 16
_UNITS = [c for c in C.NUMERIC if c not in ("Debt_to_Income", "EMI_to_Salary")] + C.CATEGORICAL


@lru_cache(maxsize=1)
def load():
    if not C.MODEL_PATH.exists():
        raise FileNotFoundError("Модель не знайдено. Спочатку виконайте: python -m src.train")
    return joblib.load(C.MODEL_PATH)


def _score(model, X):
    return score_from_proba(model.predict_proba(X))


def explain(row: pd.DataFrame, bundle, top: int = 5):
    model, base = bundle["model"], bundle["baseline"]
    rng = np.random.default_rng(C.RANDOM_STATE)
    client = row.iloc[0]
    units = [u for u in _UNITS if pd.notna(client[u])]  # невідомі поля = типове значення
    base_row = pd.DataFrame([{c: base[c] for c in C.NUMERIC + C.CATEGORICAL}])

    rows, plan = [], []
    for _ in range(N_PERMUTATIONS):
        order = rng.permutation(len(units))
        cur = base_row.iloc[0].copy()
        rows.append(cur.copy())
        for idx in order:
            cur[units[idx]] = client[units[idx]]
            rows.append(cur.copy())
        plan.append(order)
    frame = dp.add_derived(pd.DataFrame(rows).reset_index(drop=True))[C.NUMERIC + C.CATEGORICAL]
    for c in C.NUMERIC:
        frame[c] = frame[c].astype(float)
    scores = _score(model, frame).reshape(N_PERMUTATIONS, len(units) + 1)

    contrib = np.zeros(len(units))
    for scores_p, order in zip(scores, plan):
        contrib[order] += np.diff(scores_p)
    contrib /= N_PERMUTATIONS

    out = [{"feature": u, "value": _py(client[u]), "impact": float(c)}
           for u, c in zip(units, contrib) if abs(c) >= 1]
    pos = sorted((o for o in out if o["impact"] > 0), key=lambda o: -o["impact"])[:top]
    neg = sorted((o for o in out if o["impact"] < 0), key=lambda o: o["impact"])[:top]
    return {"base_score": round(float(scores[0, 0])), "raise_score": pos, "lower_score": neg}


def _py(v):
    return v.item() if isinstance(v, np.generic) else v


def score_client(payload: dict) -> dict:
    bundle = load()
    row = dp.clean(pd.DataFrame([payload]))
    proba = bundle["model"].predict_proba(row)[0]
    label = C.CLASSES[int(proba.argmax())]
    return {
        "score": round(float(score_from_proba(proba[None, :])[0])),
        "class": label,
        "decision": DECISIONS[label],
        "probabilities": {c: round(float(p), 4) for c, p in zip(C.CLASSES, proba)},
        "explanation": explain(row, bundle),
    }
