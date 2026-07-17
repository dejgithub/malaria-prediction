"""
Model Training Script for Malaria Prediction (PyTorch)
Trains the hybrid RNN+LSTM+GRU model on historical data
"""

import numpy as np
import pandas as pd
import os
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import custom modules
from data_preprocessing import MalariaDataPreprocessor
from model_architecture import HybridRNNLSTMGRU, train_model, evaluate_model, load_model


def load_processed_data():
    """Load preprocessed data"""
    print("Loading preprocessed data...")
    data = np.load("data/processed_data.npz")

    return {
        "X_train": data["X_train"],
        "X_test": data["X_test"],
        "y_train": data["y_train"],
        "y_test": data["y_test"],
    }


def plot_training_history(history, save_path="output/training_history.png"):
    """Plot training and validation loss curves"""
    print("\nPlotting training history...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history["train_loss"], label="Training Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Model Loss Over Epochs")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE plot (derived from MSE)
    train_mae = [np.sqrt(x) for x in history["train_loss"]]
    val_mae = [np.sqrt(x) for x in history["val_loss"]]
    axes[1].plot(train_mae, label="Training RMSE", linewidth=2)
    axes[1].plot(val_mae, label="Validation RMSE", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Root Mean Squared Error")
    axes[1].set_title("RMSE Over Epochs")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Training history plot saved to {save_path}")


def plot_predictions(y_true, y_pred, save_path="output/predictions_plot.png"):
    """Plot actual vs predicted values"""
    print("\nPlotting predictions...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Time series plot
    axes[0].plot(y_true, label="Actual", linewidth=2, color="blue")
    axes[0].plot(y_pred, label="Predicted", linewidth=2, color="red", linestyle="--")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Malaria Cases (Normalized)")
    axes[0].set_title("Actual vs Predicted Malaria Cases")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )
    axes[1].set_xlabel("Actual Values")
    axes[1].set_ylabel("Predicted Values")
    axes[1].set_title("Prediction Scatter Plot")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Predictions plot saved to {save_path}")


def train_model_main(epochs=150, batch_size=32):
    """Main training function"""
    print("=" * 60)
    print("MALARIA PREDICTION MODEL TRAINING (PyTorch)")
    print("=" * 60)

    # Load preprocessed data
    data = load_processed_data()

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_size = X_train.shape[2]  # number of features
    model = HybridRNNLSTMGRU(input_size=input_size, hidden_size=64, num_layers=1)

    print("\nModel Architecture:")
    print(model)

    # Train model
    print("\n" + "=" * 40)
    print("TRAINING MODEL")
    print("=" * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=0.001,
        device=device,
        model_path="models/hybrid_model.pt",
    )

    # Load best model for evaluation
    print("\n" + "=" * 40)
    print("MODEL EVALUATION")
    print("=" * 40)

    model = load_model("models/hybrid_model.pt", input_size=input_size, hidden_size=64, num_layers=1)
    model.to(device)

    metrics, y_pred = evaluate_model(model, val_loader, device=device)

    # Save metrics
    os.makedirs("output", exist_ok=True)
    with open("output/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to output/metrics.json")

    # Plot training history
    plot_training_history(history)

    # Plot predictions
    plot_predictions(y_test, y_pred)

    print("\n" + "=" * 40)
    print("TRAINING COMPLETE")
    print("=" * 40)
    print(f"\nFinal Metrics:")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  MSE:  {metrics['MSE']:.6f}")
    print(f"  R2:   {metrics['R2']:.6f}")

    return model, metrics


if __name__ == "__main__":
    model, metrics = train_model_main(epochs=150, batch_size=32)
