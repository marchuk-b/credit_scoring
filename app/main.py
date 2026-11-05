import json

from flask import Flask, jsonify, render_template, request

from src import config as C
from src.labels import label
from src.predict import load, score_client

from .schemas import validate

app = Flask(__name__)
app.jinja_env.filters["feature_label"] = label

MODEL_NAMES = {"logreg": "Логістична регресія", "gboost": "Gradient Boosting", "rf": "Random Forest",
               "lgb": "LightGBM", "xgb": "XGBoost", "ensemble": "Ансамбль RF + LGBM + XGB"}
METRIC_LABELS = [("accuracy", "Accuracy"), ("f1_macro", "F1 macro"), ("roc_auc_ovr", "ROC-AUC"),
                 ("gini_poor_vs_rest", "Gini (Poor)"), ("ks_poor_vs_rest", "KS (Poor)")]


def load_report():
    if not C.REPORT_PATH.exists():
        return None
    return json.loads(C.REPORT_PATH.read_text(encoding="utf-8"))


def load_cv():
    path = C.MODELS_DIR / "cv_report.json"  # створюється командою python -m src.cv
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/models")
def models_page():
    return render_template("models.html", report=load_report(), cv=load_cv(), model_names=MODEL_NAMES,
                           metric_labels=METRIC_LABELS, classes=C.CLASSES)


@app.get("/dataset")
def dataset_page():
    return render_template("dataset.html", report=load_report())


@app.get("/health")
def health():
    try:
        load()
        return jsonify(status="ok")
    except FileNotFoundError as e:
        return jsonify(status="model_missing", detail=str(e)), 503


@app.get("/api/report")
def report():
    report = load_report()
    if report is None:
        return jsonify(error="Звіт не знайдено"), 404
    return jsonify(report)


@app.post("/api/score")
def score():
    data, error = validate(request.get_json(silent=True))
    if error:
        return jsonify(error=error), 400
    try:
        return jsonify(score_client(data))
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503


if __name__ == "__main__":
    app.run(debug=True)