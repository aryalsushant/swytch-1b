"""
Model evaluation and comparison
calculates MAPE, R2, RMSE for XGBoost and ARIMA models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from pathlib import Path

# set up plotting
plt.style.use('seaborn-v0_8-darkgrid')

# file paths
MODEL_DIR = Path("models")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# load predictions
xgb_pred = pd.read_csv(MODEL_DIR / "xgboost_predictions.csv")
arima_pred = pd.read_csv(MODEL_DIR / "arima_predictions.csv")

print("MODEL EVALUATION AND COMPARISON")

# remove rows with NaN actual values (Sept 2025)
xgb_pred_clean = xgb_pred.dropna(subset=['actual'])
arima_pred_clean = arima_pred.dropna(subset=['actual'])

print(f"\nEvaluating on {len(xgb_pred_clean)} months with actual data")

# calculate metrics for XGBoost
def calculate_metrics(actual, predicted, model_name):
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    
    return {
        'Model': model_name,
        'MAPE (%)': round(mape, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4)
    }

# evaluate both models
xgb_results = calculate_metrics(
    xgb_pred_clean['actual'], 
    xgb_pred_clean['predicted'], 
    "XGBoost"
)

arima_results = calculate_metrics(
    arima_pred_clean['actual'], 
    arima_pred_clean['predicted'], 
    "ARIMA"
)

# print results
print("XGBOOST PERFORMANCE")
print(f"  MAPE (Mean Absolute Percentage Error): {xgb_results['MAPE (%)']}%")
print(f"  RMSE (Root Mean Squared Error): {xgb_results['RMSE']:,.2f}")
print(f"  R2 (R-squared): {xgb_results['R2']}")

print("ARIMA PERFORMANCE")
print(f"  MAPE (Mean Absolute Percentage Error): {arima_results['MAPE (%)']}%")
print(f"  RMSE (Root Mean Squared Error): {arima_results['RMSE']:,.2f}")
print(f"  R2 (R-squared): {arima_results['R2']}")

# model comparison table
comparison_df = pd.DataFrame([xgb_results, arima_results])

print("MODEL COMPARISON")
print(comparison_df.to_string(index=False))

# determine best model
best_mape = comparison_df.loc[comparison_df['MAPE (%)'].idxmin(), 'Model']
best_r2 = comparison_df.loc[comparison_df['R2'].idxmax(), 'Model']
best_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']

print("\nBest Performers:")
print(f"  Lowest MAPE: {best_mape}")
print(f"  Highest R2: {best_r2}")
print(f"  Lowest RMSE: {best_rmse}")

# save comparison table
comparison_file = PLOTS_DIR / "model_comparison_table.csv"
comparison_df.to_csv(comparison_file, index=False)
print(f"\nComparison table saved: {comparison_file}")

# create time series comparison plot
fig, ax = plt.subplots(figsize=(12, 6))

dates = pd.to_datetime(xgb_pred_clean['observation_date'])

ax.plot(dates, xgb_pred_clean['actual'], 'o-', 
        label='Actual', linewidth=2, markersize=8, color='black')
ax.plot(dates, xgb_pred_clean['predicted'], 's--', 
        label='XGBoost', linewidth=2, markersize=8, color='#2E86AB')
ax.plot(dates, arima_pred_clean['predicted'], '^--', 
        label='ARIMA', linewidth=2, markersize=8, color='#A23B72')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Total Employment (thousands)', fontsize=12)
ax.set_title('Employment Forecasting: Actual vs Predicted', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plot_file = PLOTS_DIR / "time_series_comparison.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Time series plot saved: {plot_file}")

# create bar graph for each metric
metrics = ['MAPE (%)', 'RMSE', 'R2']
colors = ['#2E86AB', '#A23B72']

for metric in metrics:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = comparison_df['Model']
    values = comparison_df[metric]
    
    bars = ax.bar(models, values, color=colors)
    
    # add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    metric_clean = metric.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    plot_file = PLOTS_DIR / f"comparison_{metric_clean}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Bar graph saved: {plot_file}")

print("EVALUATION COMPLETE")

print(f"\nAll files saved to: {PLOTS_DIR}/")
print("\nGenerated files:")
print("  - model_comparison_table.csv")
print("  - time_series_comparison.png")
print("  - comparison_MAPE_pct.png")
print("  - comparison_RMSE.png")
print("  - comparison_R2.png")