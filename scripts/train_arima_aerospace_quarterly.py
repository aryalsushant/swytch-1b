"""
ARIMA Aerospace Employment Forecasting - Quarterly Data
Time series forecasting for aerospace industry
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# file paths
DATA_DIR = Path("data/processed")
AEROSPACE_FILE = DATA_DIR / "aerospace_employment_quarterly.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("ARIMA AEROSPACE EMPLOYMENT FORECASTING (QUARTERLY)")

# load data
print("\nLoading aerospace employment data...")
df = pd.read_csv(AEROSPACE_FILE)

# filter to main employment numbers only
df = df[df['employment'] > 10000].copy()
df = df.sort_values(['year', 'quarter']).reset_index(drop=True)

print(f"Total quarters: {len(df)}")
print(f"Employment range: {df['employment'].min():,.0f} to {df['employment'].max():,.0f}")

# train/test split (last 3 for test)
train = df.iloc[:-3].copy()
test = df.iloc[-3:].copy()

y_train = train['employment']
y_test = test['employment']

print(f"\nTrain/Test Split:")
print(f"  Training: {len(train)} quarters")
print(f"  Test: {len(test)} quarters")

# train ARIMA
print("\nTraining ARIMA model...")
print("Using ARIMA(1,1,1)...")

model = ARIMA(y_train, order=(1, 1, 1))
fitted_model = model.fit()

print("Model trained successfully")

# predictions
print("\nMaking predictions...")
forecast = fitted_model.forecast(steps=len(test))
test_pred = forecast.values

# save model
model_file = MODEL_DIR / "arima_aerospace_quarterly.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(fitted_model, f)

# save predictions
predictions_df = pd.DataFrame({
    'year': test['year'],
    'quarter': test['quarter'],
    'actual': y_test.values,
    'predicted': test_pred
})

predictions_file = MODEL_DIR / "arima_aerospace_quarterly_predictions.csv"
predictions_df.to_csv(predictions_file, index=False)

print("PREDICTIONS")
print(predictions_df.to_string(index=False))

print(f"\nModel saved: {model_file}")
print(f"Predictions saved: {predictions_file}")

print("\nModel Summary:")
print(f"  AIC: {fitted_model.aic:.2f}")
print(f"  BIC: {fitted_model.bic:.2f}")
