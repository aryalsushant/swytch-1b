"""
Aerospace Model Evaluation
Compares XGBoost vs ARIMA with MAPE, R2, RMSE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')

# file paths
MODEL_DIR = Path("models")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

print("AEROSPACE MODEL EVALUATION")

# load predictions
xgb_pred = pd.read_csv(MODEL_DIR / "xgboost_aerospace_quarterly_predictions.csv")
arima_pred = pd.read_csv(MODEL_DIR / "arima_aerospace_quarterly_predictions.csv")

print(f"\nEvaluating on {len(xgb_pred)} quarters")

# calculate metrics
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

xgb_results = calculate_metrics(xgb_pred['actual'], xgb_pred['predicted'], "XGBoost")
arima_results = calculate_metrics(arima_pred['actual'], arima_pred['predicted'], "ARIMA")

# print results
print("XGBOOST PERFORMANCE")
print(f"  MAPE: {xgb_results['MAPE (%)']}%")
print(f"  RMSE: {xgb_results['RMSE']:,.2f}")
print(f"  R2: {xgb_results['R2']}")

print("ARIMA PERFORMANCE")
print(f"  MAPE: {arima_results['MAPE (%)']}%")
print(f"  RMSE: {arima_results['RMSE']:,.2f}")
print(f"  R2: {arima_results['R2']}")

# Comparison
comparison_df = pd.DataFrame([xgb_results, arima_results])

print("MODEL COMPARISON")
print(comparison_df.to_string(index=False))

best_mape = comparison_df.loc[comparison_df['MAPE (%)'].idxmin(), 'Model']
best_r2 = comparison_df.loc[comparison_df['R2'].idxmax(), 'Model']
best_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']

print("\nBest Performers:")
print(f"  Lowest MAPE: {best_mape}")
print(f"  Highest R2: {best_r2}")
print(f"  Lowest RMSE: {best_rmse}")

# save comparison
comparison_file = PLOTS_DIR / "aerospace_model_comparison.csv"
comparison_df.to_csv(comparison_file, index=False)
print(f"\nComparison saved: {comparison_file}")

# create time series plot
fig, ax = plt.subplots(figsize=(12, 6))

quarters = [f"{row['year']} Q{row['quarter']}" for _, row in xgb_pred.iterrows()]

ax.plot(quarters, xgb_pred['actual'], 'o-', 
        label='Actual', linewidth=2, markersize=10, color='black')
ax.plot(quarters, xgb_pred['predicted'], 's--', 
        label='XGBoost', linewidth=2, markersize=10, color='#2E86AB')
ax.plot(quarters, arima_pred['predicted'], '^--', 
        label='ARIMA', linewidth=2, markersize=10, color='#A23B72')

ax.set_xlabel('Quarter', fontsize=12)
ax.set_ylabel('Aerospace Employment', fontsize=12)
ax.set_title('Aerospace Employment Forecasting: Actual vs Predicted', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# format y-axis
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

plt.tight_layout()

plot_file = PLOTS_DIR / "aerospace_forecast_comparison.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Time series plot saved: {plot_file}")

# bar graphs for each metric
metrics = ['MAPE (%)', 'RMSE', 'R2']
colors = ['#2E86AB', '#A23B72']

for metric in metrics:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = comparison_df['Model']
    values = comparison_df[metric]
    
    bars = ax.bar(models, values, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Aerospace Model Comparison - {metric}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    metric_clean = metric.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    plot_file = PLOTS_DIR / f"aerospace_comparison_{metric_clean}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Bar graph saved: {plot_file}")

print("EVALUATION COMPLETE")
print(f"\nAll files saved to: {PLOTS_DIR}/")