# AI-Based Time-Series Forecasting System for Malaria Disease Prediction

## Overview
This project implements a hybrid deep learning model combining RNN, LSTM, and GRU architectures for predicting malaria trends. The system uses 15 years of historical data to predict the next 15 years of malaria risk.

## Project Structure
```
Malaria/
├── data/
│   ├── malaria_data.csv          # Historical malaria data
│   └── processed_data.npz        # Preprocessed sequences
├── models/
│   ├── hybrid_model.keras        # Trained model
│   └── model_history.json        # Training history
├── src/
│   ├── data_generator.py         # Generate sample data
│   ├── data_preprocessing.py     # Data cleaning & preprocessing
│   ├── model_architecture.py     # Hybrid RNN+LSTM+GRU model
│   ├── train_model.py            # Model training
│   └── make_predictions.py        # Generate predictions
├── dashboard/
│   └── index.html                # Interactive dashboard
├── output/
│   ├── predictions.csv           # Forecast results
│   └── metrics.json             # Evaluation metrics
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Generate historical data: `python src/data_generator.py`
2. Preprocess data: `python src/data_preprocessing.py`
3. Train model: `python src/train_model.py`
4. Generate predictions: `python src/make_predictions.py`
5. View dashboard: Open `dashboard/index.html` in a browser

## Model Architecture
The hybrid model combines:
- **RNN Layer**: Basic sequence learning
- **LSTM Layer**: Long-term dependency capture
- **GRU Layer**: Efficient temporal pattern learning

## Performance Metrics
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MSE (Mean Square Error)

## Dashboard Features
- Historical trend visualization (15 years)
- Predicted trend visualization (15 years)
- Risk level analysis
- High-risk year identification
- Prevention recommendations

## Requirements
- Python 3.8+
- TensorFlow/Keras
- NumPy, Pandas
- Chart.js (for dashboard)
- Matplotlib, Seaborn (for analysis)