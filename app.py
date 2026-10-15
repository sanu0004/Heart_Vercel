"""
Heart Disease Prediction - Flask app (Vercel-ready, no heavy ML libraries).

The trained Logistic Regression model is stored as plain weights in
model_weights.json, so no scikit-learn / pandas / numpy is needed at runtime.
Everything is loaded lazily and errors are reported in readable form.
"""

import json
import math
import os
import sys

from flask import Flask, Response, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_INDEX_CANDIDATES = [
    os.path.join(BASE_DIR, "index.html"),
    os.path.join(BASE_DIR, "templates", "index.html"),
    os.path.join(BASE_DIR, "public", "index.html"),
]
INDEX_PATH = next((p for p in _INDEX_CANDIDATES if os.path.exists(p)), _INDEX_CANDIDATES[0])
WEIGHTS_PATH = os.path.join(BASE_DIR, "model_weights.json")

app = Flask(__name__)

_cache = {}


def get_weights():
    if "w" not in _cache:
        with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
            _cache["w"] = json.load(f)
    return _cache["w"]


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def make_prediction(features):
    w = get_weights()
    values = []
    for col in w["features"]:
        if col not in features or str(features[col]).strip() == "":
            raise ValueError(f"Missing required input: '{col}'")
        try:
            values.append(float(features[col]))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid number for '{col}': {features[col]!r}")

    z = w["intercept"] + sum(c * v for c, v in zip(w["coef"], values))
    prob = _sigmoid(z)
    prediction = 1 if prob >= 0.5 else 0

    if prob >= 0.70:
        risk_level = "High Risk"
    elif prob >= 0.40:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    if prediction == 1:
        diagnosis = "Heart Disease Indicated"
        advice = ("Clinical indicators suggest a notable likelihood of heart disease. "
                  "A comprehensive cardiovascular evaluation is recommended.")
    else:
        diagnosis = "Healthy / Low Likelihood"
        advice = ("Clinical indicators are within typical healthy ranges. "
                  "Maintain a healthy lifestyle and regular check-ups.")

    return {
        "prediction": prediction,
        "diagnosis": diagnosis,
        "risk_level": risk_level,
        "risk_percentage": round(prob * 100, 2),
        "advice": advice,
    }


def _listing():
    try:
        return sorted(os.listdir(BASE_DIR))
    except Exception as e:  # noqa: BLE001
        return [f"<cannot list: {e}>"]


@app.route("/", methods=["GET"])
def index():
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return Response(f.read(), mimetype="text/html")
    except Exception as e:  # noqa: BLE001
        msg = (f"Could not load the page.\n\nError: {e}\n"
               f"Looked for: {INDEX_PATH}\n"
               f"Files next to app.py: {_listing()}\n\n"
               "Fix: make sure index.html is uploaded next to app.py.")
        return Response(msg, status=500, mimetype="text/plain")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if data is None:
            data = request.form.to_dict()
        return jsonify({"status": "success", "data": make_prediction(data)})
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"status": "error",
                        "message": f"Server error: {type(e).__name__}: {e}. "
                                   f"Files next to app.py: {_listing()}"}), 500


@app.route("/health", methods=["GET"])
def health():
    info = {
        "python": sys.version.split()[0],
        "index_html_found": os.path.exists(INDEX_PATH),
        "model_weights_found": os.path.exists(WEIGHTS_PATH),
        "files": _listing(),
    }
    try:
        info["features"] = get_weights()["features"]
        info["status"] = "healthy"
        info["model_loaded"] = True
    except Exception as e:  # noqa: BLE001
        info["status"] = "unhealthy"
        info["model_loaded"] = False
        info["error"] = f"{type(e).__name__}: {e}"
    return jsonify(info), (200 if info["status"] == "healthy" else 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
