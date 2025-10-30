"""
XGBoost Aerospace Employment Forecasting
Predicts aerospace industry employment for next 3 months
Uses 60-month training, 3-month test validation
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import pickle

# file paths
DATA_DIR = Path("data/processed")
AEROSPACE_FILE = DATA_DIR / "aerospace_employment_monthly.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("XGBOOST AEROSPACE EMPLOYMENT FORECASTING")

# load aerospace data
print("\nLoading aerospace employment data...")
df = pd.read_csv(AEROSPACE_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Total months: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Employment range: {df['employment'].min():,.0f} to {df['employment'].max():,.0f}")

# use last 63 months (60 train, 3 test as recommended)
df_recent = df.tail(63).reset_index(drop=True)

print(f"\nUsing most recent 63 months:")
print(f"  Date range: {df_recent['date'].min()} to {df_recent['date'].max()}")

# create features
print("\nCreating features...")
df_recent['month'] = df_recent['date'].dt.month
df_recent['month_sin'] = np.sin(2 * np.pi * df_recent['month'] / 12)
df_recent['month_cos'] = np.cos(2 * np.pi * df_recent['month'] / 12)
df_recent['time_index'] = range(len(df_recent))

# lagged features (previous months employment)
df_recent['employment_lag1'] = df_recent['employment'].shift(1)
df_recent['employment_lag2'] = df_recent['employment'].shift(2)
df_recent['employment_lag3'] = df_recent['employment'].shift(3)

# remove rows with NaN from lagging
df_recent = df_recent.dropna().reset_index(drop=True)

print(f"After creating lag features: {len(df_recent)} months")

# 60/3 split
train = df_recent.iloc[:-3].copy()
test = df_recent.iloc[-3:].copy()

print(f"\nTrain/Test Split (60/3):")
print(f"  Training: {len(train)} months ({train['date'].min()} to {train['date'].max()})")
print(f"  Test: {len(test)} months ({test['date'].min()} to {test['date'].max()})")

# features and target
feature_cols = ['month_sin', 'month_cos', 'time_index', 
                'employment_lag1', 'employment_lag2', 'employment_lag3']
target_col = 'employment'

X_train = train[feature_cols]
y_train = train[target_col]
X_test = test[feature_cols]
y_test = test[target_col]

print(f"\nFeatures: {feature_cols}")
print(f"Target: {target_col}")

# train XGBoost
print("\nTraining XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    objective='reg:squarederror'
)

model.fit(X_train, y_train)
print("Model trained successfully")

# predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# feature importance
print("\nFeature Importance:")
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# save model
model_file = MODEL_DIR / "xgboost_aerospace_model.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"\nModel saved: {model_file}")

# save predictions
predictions_df = pd.DataFrame({
    'date': test['date'],
    'actual': y_test.values,
    'predicted': test_pred
})

predictions_file = MODEL_DIR / "xgboost_aerospace_predictions.csv"
predictions_df.to_csv(predictions_file, index=False)
print(f"Predictions saved: {predictions_file}")

print("XGBOOST PREDICTIONS")
print(predictions_df.to_string(index=False))
