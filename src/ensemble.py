import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from . import config as C


def default_models() -> dict:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "rf": RandomForestClassifier(n_estimators=100, random_state=C.RANDOM_STATE, n_jobs=-1),
        "lgb": LGBMClassifier(n_estimators=1000, random_state=C.RANDOM_STATE, verbose=-1),
        "xgb": XGBClassifier(n_estimators=1000, random_state=C.RANDOM_STATE, n_jobs=-1),
    }


class CreditScoringEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models=None, n_splits=3, random_state=C.RANDOM_STATE):
        self.models = models
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _fit_one(model, X, y):
        # Баланс класів: однакова вага помилок для Poor / Standard / Good
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        return model

    def fit(self, X, y, groups=None):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        models = self.models if self.models is not None else default_models()

        self.calculate_weights(models, X, y, groups)
        self.models_ = {n: self._fit_one(clone(m), X, y) for n, m in models.items()}
        return self

    def calculate_weights(self, models, X, y, groups=None):
        if groups is not None:
            folds = list(GroupKFold(self.n_splits).split(X, y, np.asarray(groups)))
        else:
            folds = list(StratifiedKFold(self.n_splits, shuffle=True,
                                         random_state=self.random_state).split(X, y))
        self.cv_auc_ = {}
        for name, model in models.items():
            oof = np.zeros((len(y), len(self.classes_)))
            for tr, va in folds:
                est = self._fit_one(clone(model), X[tr], y[tr])
                oof[va] = est.predict_proba(X[va])
            self.cv_auc_[name] = float(roc_auc_score(y, oof, multi_class="ovr", average="macro"))
        raw = {n: max(a - 0.5, 1e-6) for n, a in self.cv_auc_.items()}
        total = sum(raw.values())
        self.weights_ = {n: v / total for n, v in raw.items()}
        return self.weights_

    def predict_proba(self, X):
        X = np.asarray(X)
        return sum(self.weights_[n] * m.predict_proba(X) for n, m in self.models_.items())

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
