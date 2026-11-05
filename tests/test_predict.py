import pytest

from app.main import app
from src import config as C
from src import data_processing as dp

pytestmark = pytest.mark.skipif(not C.MODEL_PATH.exists(), reason="потрібна навчена модель")

SAMPLE = {"Age": 35, "Annual_Income": 40000, "Outstanding_Debt": 1200, "Delay_from_due_date": 10}


def test_clean_handles_garbage():
    import pandas as pd
    df = dp.clean(pd.DataFrame([{"Age": "692", "Annual_Income": "40026.12_", "Credit_Mix": "_",
                                 "Credit_History_Age": "22 Years and 1 Months"}]))
    assert pd.isna(df.loc[0, "Age"])
    assert df.loc[0, "Annual_Income"] == 40026.12
    assert pd.isna(df.loc[0, "Credit_Mix"])
    assert df.loc[0, "Credit_History_Age"] == 265


def test_score_ok():
    r = app.test_client().post("/api/score", json=SAMPLE)
    assert r.status_code == 200
    d = r.get_json()
    assert 300 <= d["score"] <= 850
    assert d["class"] in C.CLASSES
    assert abs(sum(d["probabilities"].values()) - 1) < 1e-3
    assert "raise_score" in d["explanation"]
    for item in d["explanation"]["raise_score"] + d["explanation"]["lower_score"]:
        assert item["label"] and item["display"]


def test_score_missing_fields():
    r = app.test_client().post("/api/score", json={"Age": 30})
    assert r.status_code == 400


def test_pages_and_report():
    client = app.test_client()
    for url in ("/", "/models", "/dataset", "/health"):
        assert client.get(url).status_code == 200, url
    report = client.get("/api/report").get_json()
    assert {"metrics", "dataset", "best", "best_model"} <= report.keys()


def test_models_page_shows_cv_when_available():
    html = app.test_client().get("/models").get_data(as_text=True)
    assert ("Крос-валідація" in html) == (C.MODELS_DIR / "cv_report.json").exists()