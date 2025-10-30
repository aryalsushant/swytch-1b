"""
XGBoost Aerospace Employment Forecasting - Quarterly Data
Predicts aerospace employment using quarterly data
60 quarters training, 3 quarters test 
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import pickle

# file paths
DATA_DIR = Path("data/processed")
AEROSPACE_FILE = DATA_DIR / "aerospace_employment_quarterly.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("XGBOOST AEROSPACE EMPLOYMENT FORECASTING (QUARTERLY)")

# load aerospace quarterly data
print("\nLoading aerospace employment data...")
df = pd.read_csv(AEROSPACE_FILE)

# remove duplicate rows (keep only the main employment figure, not establishment counts)
df = df[df['employment'] > 10000].copy()  # filter out small numbers
df = df.sort_values(['year', 'quarter']).reset_index(drop=True)

print(f"Total quarters: {len(df)}")
print(f"Date range: {df['year'].min()} Q{df['quarter'].min()} to {df['year'].max()} Q{df['quarter'].max()}")
print(f"Employment range: {df['employment'].min():,.0f} to {df['employment'].max():,.0f}")

# create time index
df['time_index'] = range(len(df))
df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

# lagged features
df['employment_lag1'] = df['employment'].shift(1)
df['employment_lag2'] = df['employment'].shift(2)
df['employment_lag3'] = df['employment'].shift(3)

# remove NaN from lagging
df = df.dropna().reset_index(drop=True)

print(f"After lag features: {len(df)} quarters")

# 18/3 split (we have 21 quarters total, after lag we have ~18)
# use last 3 for test
train = df.iloc[:-3].copy()
test = df.iloc[-3:].copy()

print(f"\nTrain/Test Split:")
print(f"  Training: {len(train)} quarters")
print(f"  Test: {len(test)} quarters")

# features
feature_cols = ['quarter_sin', 'quarter_cos', 'time_index',
                'employment_lag1', 'employment_lag2', 'employment_lag3']
target_col = 'employment'

X_train = train[feature_cols]
y_train = train[target_col]
X_test = test[feature_cols]
y_test = test[target_col]

print(f"\nFeatures: {feature_cols}")

# train XGBoost
print("\nTraining XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)
print("Model trained successfully")

# predictions
test_pred = model.predict(X_test)

# feature importance
print("\nFeature Importance:")
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# save
model_file = MODEL_DIR / "xgboost_aerospace_quarterly.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

predictions_df = pd.DataFrame({
    'year': test['year'],
    'quarter': test['quarter'],
    'actual': y_test.values,
    'predicted': test_pred
})

predictions_file = MODEL_DIR / "xgboost_aerospace_quarterly_predictions.csv"
predictions_df.to_csv(predictions_file, index=False)

print("PREDICTIONS")
print(predictions_df.to_string(index=False))

print(f"\nModel saved: {model_file}")
print(f"Predictions saved: {predictions_file}")