import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config as C

_PLACEHOLDERS = {"_", "__", "_______", "!@9#%8", "NM_", "nan", "", "none", "null"}


def _to_number(s: pd.Series) -> pd.Series:
    # '40026.12_' -> 40026.12; '__10000__' -> 10000; сміття -> NaN.
    s = s.astype("string").str.strip().str.replace("_", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _history_months(s: pd.Series) -> pd.Series:
    # '22 Years and 1 Months' -> 265. Числа проходять без змін.
    s = s.astype("string")
    ym = s.str.extract(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?")
    months = pd.to_numeric(ym[0]) * 12 + pd.to_numeric(ym[1])
    return months.fillna(pd.to_numeric(s, errors="coerce"))


def _count_loan_types(s: pd.Series) -> pd.Series:
    def count(v):
        if pd.isna(v) or str(v).strip().lower() in {"not specified", ""}:
            return 0
        return len([p for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", str(v)) if p.strip()])
    return s.map(count)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Повертає таблицю з рівно тими ознаками, які очікує модель.
    df = df.copy()
    if "Type_of_Loan" in df:
        df["Num_Loan_Types"] = _count_loan_types(df["Type_of_Loan"])
    else:
        df["Num_Loan_Types"] = np.nan
    if "Credit_History_Age" in df:
        df["Credit_History_Age"] = _history_months(df["Credit_History_Age"])

    for col, (lo, hi) in C.VALID_RANGES.items():
        if col not in df:
            df[col] = np.nan
            continue
        values = _to_number(df[col]).astype(float)
        df[col] = values.where(values.between(lo, hi))

    for col in C.CATEGORICAL:
        if col not in df:
            df[col] = np.nan
            continue
        s = df[col].astype("string").str.strip()
        s = s.mask(s.str.lower().isin(_PLACEHOLDERS))
        df[col] = s.astype(object).where(s.notna(), np.nan)

    return add_derived(df)[C.NUMERIC + C.CATEGORICAL]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    # Похідні ознаки, що залежать від інших полів
    df = df.copy()
    income = df["Annual_Income"].replace(0, np.nan)
    df["Debt_to_Income"] = df["Outstanding_Debt"] / income
    df["EMI_to_Salary"] = df["Total_EMI_per_month"] / (income / 12)
    return df


def load_train():
    # X, y (0/1/2) та групи (Customer_ID) для розбиття без витоку по клієнтах.
    raw = pd.read_csv(C.TRAIN_CSV, dtype=str)
    y = raw[C.TARGET].map({c: i for i, c in enumerate(C.CLASSES)})
    return clean(raw), y.astype(int), raw[C.GROUP]


def split(X, y, groups):
    gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups))
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], groups.iloc[tr]
