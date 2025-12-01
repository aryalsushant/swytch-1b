# state level aerospace employment forecasting
# predicts employment for top aerospace states

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

plt.style.use('seaborn-v0_8-darkgrid')

# setup
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
PLOTS_DIR = Path("plots")
MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

print("STATE-LEVEL AEROSPACE FORECASTING")

# load data
df = pd.read_csv(DATA_DIR / "state_aerospace_complete.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['state', 'date']).reset_index(drop=True)

print(f"\nData loaded: {df.shape}")
print(f"States: {df['state'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# focus on top 5 aerospace states (excluding US TOTAL)
top_states = [
    'Washington',
    'California', 
    'Texas',
    'Florida',
    'Arizona'
]

# filter to top states
df_states = df[df['state'].isin(top_states)].copy()

print(f"\nFocusing on top {len(top_states)} states")

# create time features
df_states['month'] = df_states['date'].dt.month
df_states['year'] = df_states['date'].dt.year
df_states['month_sin'] = np.sin(2 * np.pi * df_states['month'] / 12)
df_states['month_cos'] = np.cos(2 * np.pi * df_states['month'] / 12)
df_states['time_index'] = (df_states['date'] - df_states['date'].min()).dt.days / 30

# create lag features by state
df_states = df_states.sort_values(['state', 'date'])
df_states['employment_lag1'] = df_states.groupby('state')['employment'].shift(1)
df_states['employment_lag2'] = df_states.groupby('state')['employment'].shift(2)
df_states['employment_lag3'] = df_states.groupby('state')['employment'].shift(3)

# one-hot encode states
state_dummies = pd.get_dummies(df_states['state'], prefix='state')
df_states = pd.concat([df_states, state_dummies], axis=1)

# remove NaN from lags
df_states = df_states.dropna(subset=['employment_lag1', 'employment_lag2', 'employment_lag3'])

print(f"After feature engineering: {df_states.shape}")

# train/test split: last 3 months per state for test
train_data = []
test_data = []

for state in top_states:
    state_df = df_states[df_states['state'] == state].sort_values('date')
    train_data.append(state_df.iloc[:-3])
    test_data.append(state_df.iloc[-3:])

train = pd.concat(train_data).reset_index(drop=True)
test = pd.concat(test_data).reset_index(drop=True)

print(f"\nTrain/Test split:")
print(f"  Training: {len(train)} records ({len(train)//len(top_states)} months per state)")
print(f"  Test: {len(test)} records ({len(test)//len(top_states)} months per state)")

# features
feature_cols = ['month_sin', 'month_cos', 'time_index',
                'employment_lag1', 'employment_lag2', 'employment_lag3',
                'unemployment_rate', 'real_gdp'] + [col for col in df_states.columns if col.startswith('state_')]

# remove features not in data
feature_cols = [col for col in feature_cols if col in train.columns]

X_train = train[feature_cols]
y_train = train['employment']
X_test = test[feature_cols]
y_test = test['employment']

print(f"\nFeatures: {len(feature_cols)}")

# train XGBoost
print("\nTraining XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)
print("Model trained successfully")

# predictions
y_pred = model.predict(X_test)

# calculate metrics
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MODEL PERFORMANCE")
print(f"  MAPE: {mape:.2f}%")
print(f"  RMSE: {rmse:,.2f}")
print(f"  R2: {r2:.4f}")

# save model
model_file = MODEL_DIR / "xgboost_state_level.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"\nModel saved: {model_file}")

# predictions by state
test_results = test[['date', 'state', 'employment']].copy()
test_results['predicted'] = y_pred

predictions_file = MODEL_DIR / "state_level_predictions.csv"
test_results.to_csv(predictions_file, index=False)
print(f"Predictions saved: {predictions_file}")

# performance by state
print("PERFORMANCE BY STATE")

for state in top_states:
    state_test = test_results[test_results['state'] == state]
    
    if len(state_test) > 0:
        state_mape = mean_absolute_percentage_error(
            state_test['employment'], 
            state_test['predicted']
        ) * 100
        
        print(f"\n{state}:")
        print(f"  MAPE: {state_mape:.2f}%")
        print(f"  Actual: {state_test['employment'].values}")
        print(f"  Predicted: {state_test['predicted'].values.astype(int)}")

# visualization: Predictions by state
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, state in enumerate(top_states):
    ax = axes[idx]
    
    state_test = test_results[test_results['state'] == state]
    
    dates = [d.strftime('%Y-%m') for d in state_test['date']]
    
    ax.plot(dates, state_test['employment'], 'o-', 
            label='Actual', linewidth=2, markersize=8, color='black')
    ax.plot(dates, state_test['predicted'], 's--', 
            label='Predicted', linewidth=2, markersize=8, color='#2E86AB')
    
    ax.set_title(state.replace(' employment', ''), fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Employment', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

# hide unused subplot
axes[-1].axis('off')

plt.tight_layout()
plot_file = PLOTS_DIR / "state_level_forecasts.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"\nState forecasts plot saved: {plot_file}")

#scaled version 
y_min = min(test_results['employment'].min(), test_results['predicted'].min())
y_max = max(test_results['employment'].max(), test_results['predicted'].max())
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, state in enumerate(top_states):
    ax = axes[idx]
    
    state_test = test_results[test_results['state'] == state]
    dates = [d.strftime('%Y-%m') for d in state_test['date']]
    
    ax.plot(dates, state_test['employment'], 'o-', 
            label='Actual', linewidth=2, markersize=8, color='black')
    ax.plot(dates, state_test['predicted'], 's--', 
            label='Predicted', linewidth=2, markersize=8, color='#2E86AB')
    
    ax.set_title(state.replace(' employment', ''), fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Employment', fontsize=10)

    # *** Shared scale ***
    ax.set_ylim(y_min * 0.95, y_max * 1.05)

    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

# hide unused subplot if 5 states only
axes[-1].axis('off')

plt.tight_layout()
plot_file = PLOTS_DIR / "state_level_forecasts_scaled.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"\nState forecasts plot saved with scaled y-axis: {plot_file}")


print("STATE-LEVEL FORECASTING COMPLETE")