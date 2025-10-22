# note this is just for learning and not the actual model yet


# pip install xgboost scikit-learn
# so it works
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# simple example to understand xgboost structure
# will adapt once we have actual employment data

# Load economic data
data = pd.read_csv('data/processed/economic_indicators_ready.csv')

# for learning purposes, let's predict total_employment from other features

# drop rows with NaN
data_clean = data[['unemployment_rate', 'real_gdp', 'job_postings_index', 'total_employment']].dropna()

# features (X) and target (y)
X = data_clean[['unemployment_rate', 'real_gdp', 'job_postings_index']]
y = data_clean['total_employment']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create and train model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# make predictions
predictions = model.predict(X_test)

# evaluate
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse:.2f}")
print(f"This means predictions are off by about {rmse:,.0f} thousand jobs on average")

# feature importance
print("\nFeature Importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"  {feature}: {importance:.3f}")