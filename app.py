import os
import json
import pandas as pd
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    send_from_directory,
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback
from datetime import datetime, timedelta

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)


def _get_engine():
    """Lazy-load torch ML engine only when training/prediction is needed."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from ml_engine import (
        preprocess_data,
        train_and_compare_models,
        evaluate_model,
        generate_predictions,
        create_charts,
        HybridRNNLSTMGRU,
    )
    import torch
    return {
        "preprocess_data": preprocess_data,
        "train_and_compare_models": train_and_compare_models,
        "evaluate_model": evaluate_model,
        "generate_predictions": generate_predictions,
        "create_charts": create_charts,
        "HybridRNNLSTMGRU": HybridRNNLSTMGRU,
        "torch": torch,
    }


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "message": "Server is running"})


@app.route("/charts")
def charts_page():
    try:
        return send_from_directory(os.getcwd(), "charts.html")
    except:
        return send_from_directory(".", "charts.html")


@app.route("/charts.html")
def charts_page_alt():
    try:
        return send_from_directory(os.getcwd(), "charts.html")
    except:
        return send_from_directory(".", "charts.html")


@app.route("/index.html")
def index_alt():
    try:
        return send_from_directory(os.getcwd(), "index.html")
    except:
        return send_from_directory(".", "index.html")


@app.route("/api/analysis-data")
def get_analysis_data():
    try:
        with open("output/dashboard_data.json", "r") as f:
            dashboard_data = json.load(f)

        try:
            with open("output/model_comparison.json", "r") as f:
                model_comparison = json.load(f)
        except:
            model_comparison = []

        try:
            with open("output/metrics.json", "r") as f:
                metrics = json.load(f)
        except:
            metrics = {}

        try:
            predictions_df = pd.read_csv("output/predictions_monthly.csv")
            monthly_data = predictions_df.to_dict()
        except:
            monthly_data = {}

        return jsonify({
            "success": True,
            "dashboard_data": dashboard_data,
            "model_comparison": model_comparison,
            "metrics": metrics,
            "monthly_predictions": monthly_data,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/output/<filename>")
def serve_output_file(filename):
    return send_from_directory("output", filename)


@app.route("/test")
def test():
    return jsonify({"status": "ok", "message": "Server is running"})


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        print(f"Processing file: {filepath}")

        df = pd.read_csv(filepath)

        required_cols = [
            "date", "year", "month", "malaria_cases",
            "rainfall_mm", "temperature_celsius", "humidity_percent",
        ]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        df = df.ffill().bfill()

        engine = _get_engine()
        preprocess_data = engine["preprocess_data"]
        train_and_compare_models = engine["train_and_compare_models"]
        evaluate_model = engine["evaluate_model"]
        generate_predictions = engine["generate_predictions"]
        create_charts = engine["create_charts"]
        HybridRNNLSTMGRU = engine["HybridRNNLSTMGRU"]
        torch = engine["torch"]

        print("Preprocessing data...")
        X_train, X_test, y_train, y_test, processed_df, case_min, case_max, feature_cols = preprocess_data(df)

        print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

        print("Training and comparing all models...")
        model_results, hybrid_train_losses, hybrid_val_losses = train_and_compare_models(
            X_train, y_train, X_test, y_test, epochs=50
        )

        print("Loading best hybrid model for predictions...")
        model = HybridRNNLSTMGRU(input_size=X_train.shape[2]).to("cpu")
        model.load_state_dict(
            torch.load("models/hybrid_model.pt", map_location="cpu", weights_only=True)["model_state_dict"]
        )

        metrics = evaluate_model(model, X_test, y_test)
        print(f"Metrics: {metrics}")

        print("Generating predictions...")
        predictions_df = generate_predictions(model, processed_df, case_min, case_max, feature_cols)

        print("Creating charts...")
        create_charts(processed_df, predictions_df, hybrid_train_losses, hybrid_val_losses, model_results)

        print("Creating dashboard data...")
        yearly_hist = processed_df.groupby("year")["malaria_cases"].sum().reset_index()
        yearly_pred = predictions_df.groupby("year")["predicted_cases"].sum().reset_index()
        yearly_pred["risk_level"] = yearly_pred["predicted_cases"].apply(
            lambda x: "Low" if x < 24000 else "Medium" if x < 48000 else "High" if x < 72000 else "Very High"
        )
        high_risk = yearly_pred[yearly_pred["risk_level"].isin(["High", "Very High"])]["year"].tolist()

        dashboard_data = {
            "historical_years": [int(y) for y in yearly_hist["year"].values],
            "historical_cases": [int(c) for c in yearly_hist["malaria_cases"].values],
            "predicted_years": [int(y) for y in yearly_pred["year"].values],
            "predicted_cases": [int(c) for c in yearly_pred["predicted_cases"].values],
            "risk_levels": list(yearly_pred["risk_level"].values),
            "high_risk_years": [int(y) for y in high_risk],
            "metrics": metrics,
        }

        print("Saving results...")
        with open("output/dashboard_data.json", "w") as f:
            json.dump(dashboard_data, f, indent=2)
        predictions_df.to_csv("output/predictions_monthly.csv", index=False)
        with open("output/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open("output/model_comparison.json", "w") as f:
            json.dump(model_results, f, indent=2)

        return jsonify({
            "success": True,
            "message": "Processing complete!",
            "metrics": metrics,
            "data": dashboard_data,
            "model_comparison": model_results,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/get-results")
def get_results():
    try:
        with open("output/dashboard_data.json", "r") as f:
            data = json.load(f)

        with open("output/metrics.json", "r") as f:
            metrics = json.load(f)

        with open("output/model_comparison.json", "r") as f:
            model_comparison = json.load(f)

        return jsonify({"success": True, "data": data, "metrics": metrics, "model_comparison": model_comparison})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def download_file(filename):
    return send_file(f"output/{filename}", as_attachment=True)


def verify_startup():
    required = ["output/dashboard_data.json", "output/metrics.json", "models/hybrid_model.pt"]
    for f in required:
        if not os.path.exists(f):
            print(f"WARNING: Missing {f}")
    print("Startup check complete.")


verify_startup()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1"
    debug = not os.environ.get("RENDER")
    print("=" * 50)
    print("Starting Malaria Prediction Server...")
    print(f"Open http://{host}:{port} in your browser")
    print("=" * 50)
    app.run(debug=debug, port=port, host=host)
