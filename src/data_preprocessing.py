"""
Data Preprocessing Module for Malaria Time-Series Data
Handles missing values, normalization, and sequence creation for RNN models
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import json


class MalariaDataPreprocessor:
    def __init__(self, sequence_length=12):
        """
        Initialize preprocessor
        Args:
            sequence_length: Number of time steps to use for prediction
        """
        self.sequence_length = sequence_length
        self.case_scaler = MinMaxScaler()
        self.env_scaler = StandardScaler()
        self.feature_indices = None

    def load_data(self, filepath):
        """Load data from CSV file"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records")
        return df

    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("\nHandling missing values...")

        # Check for missing values
        missing = df.isnull().sum()
        print(f"Missing values before: {missing.sum()}")

        # Fill missing values using interpolation for time series
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear")

        # If any remaining NaN, forward/backward fill
        df = df.ffill().bfill()

        missing_after = df.isnull().sum().sum()
        print(f"Missing values after: {missing_after}")

        return df

    def add_features(self, df):
        """Add engineered features for better prediction"""
        print("\nAdding engineered features...")

        # Lag features (previous months' cases)
        for lag in [1, 2, 3, 6, 12]:
            df[f"cases_lag_{lag}"] = df["malaria_cases"].shift(lag)

        # Rolling statistics
        df["cases_rolling_mean_3"] = df["malaria_cases"].rolling(window=3).mean()
        df["cases_rolling_mean_6"] = df["malaria_cases"].rolling(window=6).mean()
        df["cases_rolling_std_3"] = df["malaria_cases"].rolling(window=3).std()

        # Year-over-year change
        df["cases_yoy_change"] = df["malaria_cases"].diff(12)

        # Seasonal indicator
        df["is_rainy_season"] = df["month"].apply(
            lambda x: 1 if x in [5, 6, 7, 8, 9, 10] else 0
        )

        # Drop rows with NaN from lag features
        df = df.dropna()

        print(f"Records after feature engineering: {len(df)}")

        return df

    def normalize_features(self, df, fit_from=None):
        """Normalize features for neural network training.

        Args:
            df: DataFrame to normalize
            fit_from: Optional DataFrame to fit scalers from (to prevent data leakage).
                      If None, fits from df itself.
        """
        print("\nNormalizing features...")

        fit_df = fit_from if fit_from is not None else df

        # Select features to use
        feature_columns = [
            "malaria_cases",
            "rainfall_mm",
            "temperature_celsius",
            "humidity_percent",
            "cases_lag_1",
            "cases_lag_2",
            "cases_lag_3",
            "cases_lag_6",
            "cases_lag_12",
            "cases_rolling_mean_3",
            "cases_rolling_mean_6",
            "cases_rolling_std_3",
            "cases_yoy_change",
            "is_rainy_season",
        ]

        # Fit scalers on training data only
        cases_col = fit_df["malaria_cases"].values.reshape(-1, 1)
        self.case_scaler.fit(cases_col)

        env_cols = fit_df[["rainfall_mm", "temperature_celsius", "humidity_percent"]].values
        self.env_scaler.fit(env_cols)

        # Transform the target df
        cases_col_all = df["malaria_cases"].values.reshape(-1, 1)
        df["malaria_cases_norm"] = self.case_scaler.transform(cases_col_all).flatten()
        env_cols_all = df[["rainfall_mm", "temperature_celsius", "humidity_percent"]].values
        env_normalized = self.env_scaler.transform(env_cols_all)
        df[["rainfall_norm", "temp_norm", "humidity_norm"]] = env_normalized

        self.feature_indices = {
            "malaria_cases": feature_columns.index("malaria_cases"),
            "rainfall": feature_columns.index("rainfall_mm"),
            "temperature": feature_columns.index("temperature_celsius"),
            "humidity": feature_columns.index("humidity_percent"),
        }

        print("Features normalized successfully")

        return df

    def create_sequences(self, df):
        """Create sequences for RNN/LSTM/GRU models"""
        print("\nCreating time-series sequences...")

        # Feature columns for the model
        feature_cols = [
            "malaria_cases_norm",
            "rainfall_norm",
            "temp_norm",
            "humidity_norm",
            "cases_lag_1",
            "cases_lag_2",
            "cases_lag_3",
            "cases_lag_6",
            "cases_lag_12",
            "cases_rolling_mean_3",
            "cases_rolling_mean_6",
            "cases_rolling_std_3",
            "cases_yoy_change",
            "is_rainy_season",
        ]

        # Extract features and target
        features = df[feature_cols].values
        target = df["malaria_cases_norm"].values

        # Create sequences
        X, y = [], []
        dates = df["date"].values
        years = df["year"].values

        for i in range(len(features) - self.sequence_length):
            X.append(features[i : i + self.sequence_length])
            y.append(target[i + self.sequence_length])

        X = np.array(X)
        y = np.array(y)

        print(f"Sequence shape: X={X.shape}, y={y.shape}")

        return X, y, dates, years

    def split_data(self, X, y, train_ratio=0.8):
        """Split data into training and testing sets"""
        print(
            f"\nSplitting data: {train_ratio * 100}% train, {(1 - train_ratio) * 100}% test"
        )

        split_idx = int(len(X) * train_ratio)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def inverse_transform_cases(self, values):
        """Convert normalized values back to original scale"""
        return self.case_scaler.inverse_transform(values.reshape(-1, 1)).flatten()

    def preprocess(self, filepath):
        """Complete preprocessing pipeline"""
        # Load data
        df = self.load_data(filepath)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Add features
        df = self.add_features(df)

        # Split into train/test at the row level BEFORE normalization
        split_row = int(len(df) * 0.8)
        train_df = df.iloc[:split_row].copy()

        # Normalize using training data stats only
        df = self.normalize_features(df, fit_from=train_df)

        # Create sequences
        X, y, dates, years = self.create_sequences(df)

        # Split sequences
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Save preprocessed data
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)

        np.savez(
            os.path.join(output_dir, "processed_data.npz"),
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            dates=dates,
            years=years,
        )

        # Save scaler parameters from training data only
        scaler_params = {
            "case_scaler_min": float(self.case_scaler.data_min_[0]),
            "case_scaler_max": float(self.case_scaler.data_max_[0]),
            "sequence_length": self.sequence_length,
        }

        with open(os.path.join(output_dir, "scaler_params.json"), "w") as f:
            json.dump(scaler_params, f, indent=2)

        print("\nPreprocessing complete!")
        print(f"Data saved to {output_dir}/processed_data.npz")

        return X_train, X_test, y_train, y_test, df


if __name__ == "__main__":
    preprocessor = MalariaDataPreprocessor(sequence_length=12)
    X_train, X_test, y_train, y_test, df = preprocessor.preprocess(
        "data/malaria_data.csv"
    )
    print(f"\nPreprocessing complete!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
