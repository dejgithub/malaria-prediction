"""
Hybrid Deep Learning Model Architecture (PyTorch)
Combines RNN, LSTM, and GRU layers for malaria time-series forecasting
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from datetime import datetime


class HybridRNNLSTMGRU(nn.Module):
    def __init__(
        self, input_size, hidden_size=64, num_layers=1, output_size=1, dropout=0.3
    ):
        """
        Initialize hybrid model
        Args:
            input_size: Number of input features
            hidden_size: Hidden state size
            num_layers: Number of recurrent layers
            output_size: Number of output units
            dropout: Dropout rate
        """
        super(HybridRNNLSTMGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Conv1D for feature extraction
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # RNN layer - Basic recurrent for initial sequence processing
        self.rnn = nn.RNN(
            32, hidden_size, num_layers=1, batch_first=True
        )

        # LSTM layer - Long-term dependency capture
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=1, batch_first=True
        )

        # GRU layer - Efficient temporal pattern learning
        self.gru = nn.GRU(
            hidden_size, hidden_size // 2, num_layers=1, batch_first=True
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)

        # Conv1D layer
        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        # Transpose back: (batch, seq_len, features)
        x = x.transpose(1, 2)

        # RNN layer
        rnn_out, _ = self.rnn(x)

        # LSTM layer
        lstm_out, _ = self.lstm(rnn_out)

        # GRU layer
        gru_out, _ = self.gru(lstm_out)

        # Take the last time step
        out = gru_out[:, -1, :]

        # Fully connected layers
        out = self.fc(out)

        return out


def create_model(input_size, hidden_size=64, num_layers=1):
    """Factory function to create the hybrid model"""
    model = HybridRNNLSTMGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
    )
    return model


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=0.001,
    device="cpu",
    model_path="models/hybrid_model.pt",
):
    """Train the model"""
    print("\nStarting model training...")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                model_path,
            )

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

    print("\nTraining complete!")

    # Save training history
    history = {"train_loss": train_losses, "val_loss": val_losses}
    os.makedirs("models", exist_ok=True)
    with open("models/model_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


def evaluate_model(model, test_loader, device="cpu"):
    """Evaluate model performance"""
    print("\nEvaluating model...")

    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(y_batch.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    # R2 score
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    metrics = {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
    }

    print("\nEvaluation Metrics:")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R2:   {r2:.6f}")

    return metrics, predictions


def predict(model, X, device="cpu"):
    """Make predictions on new data"""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        predictions = model(X_tensor).squeeze().cpu().numpy()
    return predictions


def load_model(filepath, input_size, hidden_size=64, num_layers=1):
    """Load a trained model"""
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)
    model = HybridRNNLSTMGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    input_size = 14  # number of features
    model = create_model(input_size)

    # Test forward pass
    batch_size = 4
    seq_length = 12
    X = torch.randn(batch_size, seq_length, input_size)
    output = model(X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print("Model created successfully!")
