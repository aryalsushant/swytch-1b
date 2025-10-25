"""
XGBoost Employment Forecasting Model
predicts total employment using economic indicators
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import pickle

# file paths
DATA_DIR = Path("data/processed")
TRAIN_FILE = DATA_DIR / "train_data.csv"
TEST_FILE = DATA_DIR / "test_data.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("XGBOOST EMPLOYMENT FORECASTING MODEL")

# load data
print("\nLoading data...")
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Training data: {train.shape}")
print(f"Test data: {test.shape}")

# prepare features and target
print("\nPreparing features...")

# features: unemployment_rate, job_postings_index, real_gdp
# target: total_employment
feature_cols = ['unemployment_rate', 'job_postings_index', 'real_gdp']
target_col = 'total_employment'

X_train = train[feature_cols]
y_train = train[target_col]

X_test = test[feature_cols]
y_test = test[target_col]

print(f"Features: {feature_cols}")
print(f"Target: {target_col}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# train XGBoost model
print("\nTraining XGBoost model...")

model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    objective='reg:squarederror'
)

model.fit(X_train, y_train)
print("Model trained successfully!")

# make predictions
print("\nMaking predictions...")
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# feature importance
print("\nFeature Importance:")
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f} ({row['Importance']*100:.1f}%)")

# save model
model_file = MODEL_DIR / "xgboost_model.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"\nModel saved: {model_file}")

# save predictions
predictions_df = pd.DataFrame({
    'observation_date': test['observation_date'],
    'actual': y_test,
    'predicted': test_predictions
})

predictions_file = MODEL_DIR / "xgboost_predictions.csv"
predictions_df.to_csv(predictions_file, index=False)
print(f"Predictions saved: {predictions_file}")

print("XGBOOST MODEL COMPLETE!")

print(f"\nPredictions for last 3 months:")
print(predictions_df.to_string(index=False))
