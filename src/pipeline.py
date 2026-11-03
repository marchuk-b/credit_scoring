from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .ensemble import CreditScoringEnsemble


def _preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, C.NUMERIC), ("cat", cat, C.CATEGORICAL)])


def build(name: str, ensemble_models=None) -> Pipeline:
    if name == "logreg":  # інтерпретований бейзлайн
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif name == "gboost":
        model = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=300, max_leaf_nodes=31, l2_regularization=1.0,
            class_weight="balanced", random_state=C.RANDOM_STATE,
        )
    elif name == "ensemble":  # RF + LightGBM + XGBoost, ваги за крос-валідацією
        model = CreditScoringEnsemble(models=ensemble_models)
    else:
        raise ValueError(name)
    return Pipeline([("prep", _preprocessor()), ("model", model)])
