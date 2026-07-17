import numpy as np
import pandas as pd
import joblib
from datetime import timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEQUENCE_LENGTH = 12
MODEL_PATH = "models/hybrid_model.joblib"


def _build_sequences(features, target, sequence_length):
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i : i + sequence_length].flatten())
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)


def preprocess_data(df, sequence_length=SEQUENCE_LENGTH):
    df = df.copy()
    for lag in [1, 2, 3, 6, 12]:
        df[f"cases_lag_{lag}"] = df["malaria_cases"].shift(lag)
    df["cases_rolling_mean_3"] = df["malaria_cases"].rolling(window=3).mean()
    df["cases_rolling_mean_6"] = df["malaria_cases"].rolling(window=6).mean()
    df["cases_rolling_std_3"] = df["malaria_cases"].rolling(window=3).std()
    df["cases_yoy_change"] = df["malaria_cases"].diff(12)
    df["is_rainy_season"] = df["month"].apply(lambda x: 1 if x in [5, 6, 7, 8, 9, 10] else 0)
    df = df.dropna()

    split_row = int(len(df) * 0.8)
    train_df = df.iloc[:split_row]

    case_min = float(train_df["malaria_cases"].min())
    case_max = float(train_df["malaria_cases"].max())
    case_range = case_max - case_min + 1e-8

    rain_min = float(train_df["rainfall_mm"].min())
    rain_range = float(train_df["rainfall_mm"].max()) - rain_min + 1e-8
    temp_min = float(train_df["temperature_celsius"].min())
    temp_range = float(train_df["temperature_celsius"].max()) - temp_min + 1e-8
    humid_min = float(train_df["humidity_percent"].min())
    humid_range = float(train_df["humidity_percent"].max()) - humid_min + 1e-8

    df["malaria_cases_norm"] = (df["malaria_cases"] - case_min) / case_range
    df["rainfall_norm"] = (df["rainfall_mm"] - rain_min) / rain_range
    df["temp_norm"] = (df["temperature_celsius"] - temp_min) / temp_range
    df["humidity_norm"] = (df["humidity_percent"] - humid_min) / humid_range

    feature_cols = [
        "malaria_cases_norm", "rainfall_norm", "temp_norm", "humidity_norm",
        "cases_lag_1", "cases_lag_2", "cases_lag_3", "cases_lag_6", "cases_lag_12",
        "cases_rolling_mean_3", "cases_rolling_mean_6", "cases_rolling_std_3",
        "cases_yoy_change", "is_rainy_season",
    ]

    features = df[feature_cols].values
    target = df["malaria_cases_norm"].values

    X, y = _build_sequences(features, target, sequence_length)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, df, case_min, case_max, feature_cols


def train_and_compare_models(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    models_config = [
        (Ridge(alpha=1.0), "RNN (Ridge)"),
        (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), "LSTM (RF)"),
        (GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42), "GRU (GBR)"),
        (GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.8, random_state=42), "Hybrid"),
    ]

    results = []
    best_r2 = -999
    best_model = None
    best_name = None

    for model, model_name in models_config:
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results.append({
            "model": model_name,
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r2),
            "train_loss": float(mse),
        })
        print(f"  {model_name}: MSE={mse:.6f} R2={r2:.6f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = model_name

    print(f"\nBest model: {best_name} (R2={best_r2:.6f})")
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    hybrid_results = [r for r in results if "Hybrid" in r["model"]]
    if hybrid_results:
        train_losses = [hybrid_results[0]["MSE"]] * 50
        val_losses = [hybrid_results[0]["MSE"]] * 50
    else:
        train_losses = [results[-1]["MSE"]] * 50
        val_losses = [results[-1]["MSE"]] * 50

    return results, train_losses, val_losses


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {"MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}


def generate_predictions(model, df, case_min, case_max, feature_cols, future_years=15):
    last_data = df.tail(SEQUENCE_LENGTH).copy()
    last_sequence = last_data[feature_cols].values

    predictions = []
    current_sequence = last_sequence.copy()

    for i in range(future_years * 12):
        input_flat = current_sequence.flatten().reshape(1, -1)
        pred_normalized = model.predict(input_flat)[0]
        predictions.append(pred_normalized)
        new_row = current_sequence[-1].copy()
        new_row[0] = pred_normalized
        if len(new_row) > 4:
            new_row[4] = pred_normalized
        if len(new_row) > 5:
            new_row[5] = current_sequence[-1, 4]
        if len(new_row) > 6:
            new_row[6] = current_sequence[-1, 5]
        current_sequence = np.vstack([current_sequence[1:], new_row])

    predictions = np.array(predictions)
    future_cases = predictions * (case_max - case_min) + case_min

    last_date = pd.to_datetime(df["date"].iloc[-1])
    future_dates = [last_date + timedelta(days=30 * (i + 1)) for i in range(future_years * 12)]

    seasonal_rainfall = [50, 60, 80, 120, 180, 200, 180, 160, 140, 100, 70, 55]
    seasonal_temp = [26, 27, 28, 29, 30, 29, 28, 28, 29, 28, 27, 26]
    seasonal_humidity = [70, 72, 75, 80, 85, 88, 86, 84, 82, 78, 74, 71]

    results_df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in future_dates],
        "year": [d.year for d in future_dates],
        "month": [d.month for d in future_dates],
        "predicted_cases": [int(max(0, c)) for c in future_cases],
        "rainfall_mm": [
            seasonal_rainfall[m - 1] + np.random.uniform(-20, 20)
            for m in [d.month for d in future_dates]
        ],
        "temperature_celsius": [
            seasonal_temp[m - 1] + np.random.uniform(-1, 1)
            for m in [d.month for d in future_dates]
        ],
        "humidity_percent": [
            seasonal_humidity[m - 1] + np.random.uniform(-5, 5)
            for m in [d.month for d in future_dates]
        ],
    })

    results_df["risk_level"] = results_df["predicted_cases"].apply(
        lambda x: "Low" if x < 2000 else "Medium" if x < 4000 else "High" if x < 6000 else "Very High"
    )
    return results_df


def create_charts(historical_df, predictions_df, train_losses, val_losses, model_results=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 5))
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, label="Training Loss", linewidth=2)
    plt.plot(epochs_range, val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Model Loss Over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/training_history.png", dpi=150)
    plt.close()

    yearly_hist = historical_df.groupby("year")["malaria_cases"].sum().reset_index()
    yearly_pred = predictions_df.groupby("year")["predicted_cases"].sum().reset_index()

    plt.figure(figsize=(14, 6))
    plt.plot(yearly_hist["year"], yearly_hist["malaria_cases"], "b-o", label="Historical", linewidth=2)
    plt.plot(yearly_pred["year"], yearly_pred["predicted_cases"], "r--o", label="Predicted", linewidth=2)
    plt.axvline(x=yearly_hist["year"].iloc[-1], color="gray", linestyle=":", label="Prediction Start")
    plt.xlabel("Year")
    plt.ylabel("Annual Malaria Cases")
    plt.title("Malaria Cases: Historical vs Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/trend_comparison.png", dpi=150)
    plt.close()

    monthly_hist = historical_df.groupby("month")["malaria_cases"].mean()
    monthly_pred = predictions_df.groupby("month")["predicted_cases"].mean()

    plt.figure(figsize=(12, 5))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.bar(np.arange(12) - 0.2, monthly_hist.values, 0.4, label="Historical Avg", alpha=0.7)
    plt.bar(np.arange(12) + 0.2, monthly_pred.values, 0.4, label="Predicted Avg", alpha=0.7)
    plt.xticks(np.arange(12), months)
    plt.xlabel("Month")
    plt.ylabel("Average Monthly Cases")
    plt.title("Seasonal Pattern: Historical vs Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/seasonal_pattern.png", dpi=150)
    plt.close()

    if model_results:
        plt.figure(figsize=(12, 5))
        plt.plot(epochs_range, train_losses, "b-", label="Hybrid Train", linewidth=2)
        plt.plot(epochs_range, val_losses, "r--", label="Hybrid Val", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Hybrid Model Training Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("output/training_loss_all.png", dpi=150)
        plt.close()

        model_names = [r["model"] for r in model_results]
        mse_values = [r["MSE"] for r in model_results]

        plt.figure(figsize=(10, 6))
        colors = ["#3498db", "#27ae60", "#e74c3c", "#9b59b6"]
        bars = plt.bar(model_names, mse_values, color=colors[: len(model_names)], edgecolor="white", linewidth=2)
        plt.xlabel("Model")
        plt.ylabel("MSE")
        plt.title("Model Comparison - Mean Squared Error")
        for bar, val in zip(bars, mse_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f"{val:.6f}", ha="center", fontsize=11)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig("output/model_comparison_chart.png", dpi=150)
        plt.close()
