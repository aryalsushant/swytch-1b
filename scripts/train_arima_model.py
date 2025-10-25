"""
ARIMA Employment Forecasting Model
time series forecasting for total employment
"""
#  note: run "pip install statsmodels" if you have not

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# file paths
DATA_DIR = Path("data/processed")
TRAIN_FILE = DATA_DIR / "train_data.csv"
TEST_FILE = DATA_DIR / "test_data.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("ARIMA EMPLOYMENT FORECASTING MODEL")


# load data
print("\nLoading data...")
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Training data: {train.shape}")
print(f"Test data: {test.shape}")

# prepare time series
print("\nPreparing time series...")
train['observation_date'] = pd.to_datetime(train['observation_date'])
test['observation_date'] = pd.to_datetime(test['observation_date'])

# use total_employment as our time series
y_train = train['total_employment']
y_test = test['total_employment']

print(f"Training samples: {len(y_train)}")
print(f"Test samples: {len(y_test)}")

# train ARIMA model
print("\nTraining ARIMA model...")
print("Finding best parameters (this may take a moment)...")

# try ARIMA(1,1,1) as starting point
# p=1 (autoregressive), d=1 (differencing), q=1 (moving average)
model = ARIMA(y_train, order=(1, 1, 1))
fitted_model = model.fit()

print("Model trained successfully!")
print(f"\nARIMA order: (1, 1, 1)")

# make predictions
print("\nMaking predictions...")

# in-sample predictions (training data)
train_predictions = fitted_model.fittedvalues

# out-of-sample predictions (test data)
forecast = fitted_model.forecast(steps=len(y_test))
test_predictions = forecast.values

# save model
model_file = MODEL_DIR / "arima_model.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(fitted_model, f)
print(f"\nModel saved: {model_file}")

# save predictions
predictions_df = pd.DataFrame({
    'observation_date': test['observation_date'],
    'actual': y_test,
    'predicted': test_predictions
})

predictions_file = MODEL_DIR / "arima_predictions.csv"
predictions_df.to_csv(predictions_file, index=False)
print(f"Predictions saved: {predictions_file}")

# model summary

print("ARIMA MODEL COMPLETE!")


print(f"\nPredictions for last 3 months:")
print(predictions_df.to_string(index=False))

print("\nModel Summary:")
print(f"AIC: {fitted_model.aic:.2f}")
print(f"BIC: {fitted_model.bic:.2f}")
