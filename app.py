"""
Flask Web Application for Credit Card Approval Prediction.
"""

import os

from flask import Flask, jsonify, render_template, request

from utils.helper import DATASET_PATH, load_metrics_json, load_model_artifact
from utils.preprocess import DataPreprocessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "credit-card-approval-secret-key")

# Load model artifact at startup
artifact = load_model_artifact()
model = artifact["model"] if artifact else None
preprocessor = artifact["preprocessor"] if artifact else None
best_model_name = artifact.get("best_model_name", "N/A") if artifact else "N/A"


@app.route("/")
def index():
    """Home page with project overview."""
    return render_template("index.html", best_model=best_model_name)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Prediction page - form input and ML inference."""
    result = None
    probability = None
    error = None

    if request.method == "POST":
        if model is None or preprocessor is None:
            error = "Model not loaded. Please run models/train_model.py first."
        else:
            try:
                form_data = {
                    "gender": request.form.get("gender"),
                    "age": request.form.get("age"),
                    "annual_income": request.form.get("annual_income"),
                    "employment_status": request.form.get("employment_status"),
                    "years_of_employment": request.form.get("years_of_employment"),
                    "education_level": request.form.get("education_level"),
                    "marital_status": request.form.get("marital_status"),
                    "number_of_dependents": request.form.get("number_of_dependents"),
                    "existing_loan_amount": request.form.get("existing_loan_amount"),
                    "monthly_income": request.form.get("monthly_income"),
                    "credit_score": request.form.get("credit_score"),
                    "loan_status": request.form.get("loan_status"),
                    "credit_history": request.form.get("credit_history"),
                    "number_of_credit_inquiries": request.form.get(
                        "number_of_credit_inquiries"
                    ),
                    "debt_to_income_ratio": request.form.get("debt_to_income_ratio"),
                }

                input_df = preprocessor.encode_input(form_data)
                X = preprocessor.transform(input_df)
                prediction = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                result = preprocessor.decode_target(prediction)
                probability = round(max(proba) * 100, 2)
            except Exception as e:
                error = f"Prediction error: {str(e)}"

    return render_template(
        "predict.html",
        result=result,
        probability=probability,
        error=error,
        best_model=best_model_name,
    )


@app.route("/dashboard")
def dashboard():
    """Dashboard page with model analytics and charts."""
    metrics_data = load_metrics_json()
    dataset_exists = os.path.exists(DATASET_PATH)
    return render_template(
        "dashboard.html",
        metrics=metrics_data,
        dataset_exists=dataset_exists,
        best_model=best_model_name,
    )


@app.route("/api/metrics")
def api_metrics():
    """API endpoint for dashboard metrics JSON."""
    data = load_metrics_json()
    if data is None:
        return jsonify({"error": "Metrics not available. Run training first."}), 404
    return jsonify(data)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for predictions."""
    if model is None or preprocessor is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        input_df = preprocessor.encode_input(data)
        X = preprocessor.transform(input_df)
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        result = preprocessor.decode_target(prediction)
        return jsonify({
            "prediction": result,
            "probability": round(max(proba) * 100, 2),
            "model": best_model_name,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
