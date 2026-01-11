import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from pathlib import Path
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')

# file paths
MODEL_DIR = Path("models")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

print("AEROSPACE MODEL EVALUATION")

# load predictions
pred_file = MODEL_DIR / "state_level_predictions_xgb_arima.csv"
df = pd.read_csv(pred_file, parse_dates=['date'])

top_states = df['state'].unique()
print(f"Evaluating {len(df)} records across {len(top_states)} states")

# calculate metrics
def calculate_metrics(actual, predicted, model_name):
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return {'Model': model_name, 'MAPE (%)': round(mape, 2), 'RMSE': round(rmse, 2), 'R2': round(r2, 4)}

# overall metrics
xgb_results = calculate_metrics(df['employment'], df['xgb_predicted'], "XGBoost")
arima_results = calculate_metrics(df['employment'], df['arima_predicted'], "ARIMA")

print("\nOVERALL MODEL PERFORMANCE")
print("XGBoost:", xgb_results)
print("ARIMA:", arima_results)

# per-state metrics
print("\nPERFORMANCE BY STATE")
state_metrics = []
for state in top_states:
    state_df = df[df['state'] == state]
    xgb_m = calculate_metrics(state_df['employment'], state_df['xgb_predicted'], "XGBoost")
    arima_m = calculate_metrics(state_df['employment'], state_df['arima_predicted'], "ARIMA")
    state_metrics.append({'state': state, **xgb_m, 'Model': 'XGBoost'})
    state_metrics.append({'state': state, **arima_m, 'Model': 'ARIMA'})
    
state_metrics_df = pd.DataFrame(state_metrics)
print(state_metrics_df)

# save comparison
comparison_file = PLOTS_DIR / "state_level_model_comparison.csv"
state_metrics_df.to_csv(comparison_file, index=False)
print(f"\nComparison saved: {comparison_file}")

# plot time series for each state
sns.set_style("whitegrid")

metrics = ['RMSE', 'MAPE (%)']
colors = {'XGBoost': '#A23B72', 'ARIMA':'#2E86AB'}

for metric in metrics:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # pivot data for bar plot
    plot_df = state_metrics_df.pivot(index='state', columns='Model', values=metric)
    
    plot_df.plot(kind='bar', ax=ax, color=[colors['XGBoost'], colors['ARIMA']])
    
    # add labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xlabel('State', fontsize=12)
    ax.set_title(f'State-Level Model Comparison - {metric}', fontsize=14, fontweight='bold')
    ax.legend(title='Model')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_file = PLOTS_DIR / f"state_level_{metric.replace(' ', '_').replace('(', '').replace(')', '').replace('%','pct')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"{metric} bar chart saved: {plot_file}")

print("\nEVALUATION COMPLETE")