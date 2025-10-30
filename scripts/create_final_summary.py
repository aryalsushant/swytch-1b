# comprehensive aerospace employment analysis summary
# combines national forecasting, state-level analysis, and key insights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')

# setup
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
PLOTS_DIR = Path("plots")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

print( "AEROSPACE EMPLOYMENT ANALYSIS")
print( "FINAL SUMMARY REPORT")

# PART 1: DATA OVERVIEW

print("DATA OVERVIEW")

# load state data
state_df = pd.read_csv(DATA_DIR / "state_aerospace_complete.csv")
state_df['date'] = pd.to_datetime(state_df['date'])

print(f"\nDataset Coverage:")
print(f"  Time Period: {state_df['date'].min().strftime('%B %Y')} to {state_df['date'].max().strftime('%B %Y')}")
print(f"  Total Months: {state_df['date'].nunique()}")
print(f"  States Analyzed: {state_df['state'].nunique()}")

# US Total employment trend
us_total = state_df[state_df['state'] == 'U.S. TOTAL employment'].sort_values('date')

print(f"\nU.S. Aerospace Employment:")
print(f"  Starting (Jan 2020): {us_total.iloc[0]['employment']:,.0f}")
print(f"  COVID Low (Dec 2020): {us_total[us_total['date'] == '2020-12-01']['employment'].values[0]:,.0f}")
print(f"  Latest (Mar 2025): {us_total.iloc[-1]['employment']:,.0f}")
print(f"  5-Year Change: {((us_total.iloc[-1]['employment'] - us_total.iloc[0]['employment']) / us_total.iloc[0]['employment'] * 100):.1f}%")

# PART 2: NATIONAL FORECASTING RESULTS
print("NATIONAL FORECASTING MODELS")

# load quarterly predictions
xgb_nat = pd.read_csv(MODEL_DIR / "xgboost_aerospace_quarterly_predictions.csv")
arima_nat = pd.read_csv(MODEL_DIR / "arima_aerospace_quarterly_predictions.csv")

print("\nModel Comparison (National Level - Quarterly):")
print(f"\n  XGBoost:")
print(f"    MAPE: 0.43%")
print(f"    RMSE: 2,956 jobs")
print(f"    R²: -2.05 (limited test data)")

print(f"\n  ARIMA (1,1,1):")
print(f"    MAPE: 0.25%  (BEST)")
print(f"    RMSE: 2,125 jobs (BEST)")
print(f"    R²: -0.57")

print(f"\n  Winner: ARIMA")
print(f"  Conclusion: ARIMA provides more accurate national-level forecasts")

# PART 3: STATE-LEVEL FORECASTING RESULTS
print("STATE-LEVEL FORECASTING (WITH ECONOMIC INDICATORS)")

state_pred = pd.read_csv(MODEL_DIR / "state_level_predictions.csv")

print("\nXGBoost State-Level Model:")
print(f"  Overall MAPE: 3.56%")
print(f"  Overall R²: 0.936 (excellent)")
print(f"  Features Used: Unemployment rate, GDP, job postings, seasonality, lag values")

print("\n  Performance by State:")

states_performance = [
    ('California employment', 0.47),
    ('Florida employment', 0.60),
    ('Texas employment', 1.11),
    ('Arizona employment', 1.26),
    ('Washington employment', 14.39)
]

for state, mape in states_performance:
    state_name = state.replace(' employment', '')
    marker = " ⭐" if mape < 1.5 else ""
    print(f"    {state_name}: {mape:.2f}% MAPE{marker}")

# PART 4: STATE GROWTH ANALYSIS
print("STATE GROWTH ANALYSIS (2020-2025)")

# calculate growth for each state
growth_analysis = []

for state in state_df['state'].unique():
    if state != 'U.S. TOTAL employment':
        state_data = state_df[state_df['state'] == state].sort_values('date')
        
        start_emp = state_data.iloc[0]['employment']
        end_emp = state_data.iloc[-1]['employment']
        growth_pct = ((end_emp - start_emp) / start_emp) * 100
        
        growth_analysis.append({
            'State': state.replace(' employment', ''),
            'Start (Jan 2020)': f"{start_emp:,.0f}",
            'End (Mar 2025)': f"{end_emp:,.0f}",
            'Growth %': f"{growth_pct:+.1f}%"
        })

growth_df = pd.DataFrame(growth_analysis)
growth_df['Growth_Numeric'] = growth_df['Growth %'].str.replace('%', '').astype(float)
growth_df = growth_df.sort_values('Growth_Numeric', ascending=False)

print("\nTop Growing States:")
for idx, row in growth_df.head(3).iterrows():
    print(f"  {row['State']}: {row['Growth %']}")

print("\nDeclining States:")
for idx, row in growth_df.tail(3).iterrows():
    print(f"  {row['State']}: {row['Growth %']}")

# save growth table
growth_table = growth_df.drop('Growth_Numeric', axis=1)
growth_file = OUTPUT_DIR / "state_growth_analysis.csv"
growth_table.to_csv(growth_file, index=False)

# PART 5: SEASONALITY ANALYSIS
print("SEASONALITY ANALYSIS")

# analyze US Total by month
us_total['month'] = us_total['date'].dt.month
us_total['month_name'] = us_total['date'].dt.strftime('%B')

# calculate average change by month (year-over-year)
monthly_patterns = us_total.groupby('month')['employment'].mean().sort_values(ascending=False)

print("\nAverage Employment by Month (US Total):")
month_names = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

for month_num in monthly_patterns.head(3).index:
    month_name = month_names[month_num - 1]
    avg_emp = monthly_patterns[month_num]
    print(f"  {month_name}: {avg_emp:,.0f} (High)")

print("\nLowest Employment Months:")
for month_num in monthly_patterns.tail(3).index:
    month_name = month_names[month_num - 1]
    avg_emp = monthly_patterns[month_num]
    print(f"  {month_name}: {avg_emp:,.0f}")

print("\nKey Finding: Aerospace shows moderate seasonality with peaks in summer months")

# PART 6: CREATE COMPREHENSIVE VISUALIZATION
print("GENERATING VISUALIZATIONS")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# plot 1: US Total Employment Trend
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(us_total['date'], us_total['employment'], linewidth=2, color='#2E86AB')
ax1.axvline(pd.to_datetime('2020-12-01'), color='red', linestyle='--', alpha=0.5, label='COVID Low')
ax1.set_title('U.S. Aerospace Employment Trend (2020-2025)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Employment', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

# plot 2: State Rankings (2024-2025 Average)
ax2 = fig.add_subplot(gs[1, 0])
recent = state_df[state_df['date'] >= '2024-01-01']
state_avg = recent[recent['state'] != 'U.S. TOTAL employment'].groupby('state')['employment'].mean().sort_values(ascending=True)
state_names_clean = [s.replace(' employment', '') for s in state_avg.index]

colors_grad = plt.cm.Blues(np.linspace(0.4, 0.8, len(state_avg)))
ax2.barh(state_names_clean, state_avg.values, color=colors_grad)
ax2.set_title('Aerospace Employment by State (2024-2025 Avg)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Employment', fontsize=10)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
ax2.grid(True, axis='x', alpha=0.3)

# plot 3: Growth Rates
ax3 = fig.add_subplot(gs[1, 1])
growth_sorted = growth_df.sort_values('Growth_Numeric')
colors = ['#A23B72' if x < 0 else '#2E86AB' for x in growth_sorted['Growth_Numeric']]
ax3.barh(growth_sorted['State'], growth_sorted['Growth_Numeric'], color=colors)
ax3.set_title('Employment Growth by State (2020-2025)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Growth %', fontsize=10)
ax3.axvline(0, color='black', linewidth=0.8)
ax3.grid(True, axis='x', alpha=0.3)

# plot 4: Model Comparison
ax4 = fig.add_subplot(gs[2, 0])
models = ['ARIMA\n(National)', 'XGBoost\n(National)', 'XGBoost\n(State-Level)']
mapes = [0.25, 0.43, 3.56]
colors_model = ['#2E86AB', '#F18F01', '#C73E1D']
bars = ax4.bar(models, mapes, color=colors_model)
ax4.set_title('Model Performance Comparison (MAPE)', fontsize=12, fontweight='bold')
ax4.set_ylabel('MAPE (%)', fontsize=10)
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax4.grid(True, axis='y', alpha=0.3)

# plot 5: Seasonality
ax5 = fig.add_subplot(gs[2, 1])
month_order = list(range(1, 13))
monthly_avg = [monthly_patterns.get(m, 0) for m in month_order]
ax5.plot(month_order, monthly_avg, marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax5.set_title('Seasonal Employment Pattern (Monthly Average)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Month', fontsize=10)
ax5.set_ylabel('Average Employment', fontsize=10)
ax5.set_xticks(month_order)
ax5.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
ax5.grid(True, alpha=0.3)

plt.suptitle('Aerospace Employment Analysis: Comprehensive Summary', 
             fontsize=16, fontweight='bold', y=0.995)

summary_plot = PLOTS_DIR / "comprehensive_summary.png"
plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
print(f"  Comprehensive visualization saved: {summary_plot}")

# PART 7: KEY INSIGHTS
print("KEY INSIGHTS")

insights = [
    "1. National Recovery: Aerospace employment recovered from COVID lows, growing 4.3% overall",
    "2. State Leaders: California (94K), Washington (81K), and Texas (50K) dominate aerospace employment",
    "3. Best Growth: Florida (+26.3%) and Georgia (+18.9%) show strongest growth 2020-2025",
    "4. Model Performance: ARIMA best for national forecasts (0.25% MAPE), XGBoost excellent for state-level (3.56% MAPE)",
    "5. Economic Factors: Unemployment rate, GDP, and job postings are strong predictors",
    "6. Seasonality: Moderate seasonal patterns with higher employment in summer months",
    "7. State Diversity: Growth varies significantly by state - some growing 26%, others declining 9%"
]

for insight in insights:
    print(f"\n  {insight}")


# SAVE SUMMARY REPORT
print("SAVING SUMMARY REPORT")

# create markdown report
report_lines = [
    "# Aerospace Employment Analysis - Final Report\n",
    "## Project Overview\n",
    f"- **Time Period:** {state_df['date'].min().strftime('%B %Y')} to {state_df['date'].max().strftime('%B %Y')}",
    f"- **Data Sources:** BLS QCEW (aerospace industry NAICS 3364), FRED economic indicators",
    f"- **States Analyzed:** {state_df['state'].nunique()}",
    f"- **Total Records:** {len(state_df)}\n",
    
    "## Model Results\n",
    "### National-Level Forecasting",
    "- **ARIMA (1,1,1):** 0.25% MAPE  (Best for national forecasts)",
    "- **XGBoost:** 0.43% MAPE\n",
    
    "### State-Level Forecasting",
    "- **XGBoost Multi-State Model:** 3.56% MAPE, R² = 0.936",
    "- **Features:** Unemployment, GDP, job postings, seasonality, employment lags",
    "- **Top Performer:** California (0.47% MAPE)\n",
    
    "## Key Findings\n",
]

for insight in insights:
    report_lines.append(f"{insight}\n")



report_file = OUTPUT_DIR / "FINAL_REPORT.md"
with open(report_file, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"  Final report saved: {report_file}")

# save growth analysis
print(f"  Growth analysis saved: {growth_file}")

print("ANALYSIS COMPLETE :) READY FOR PRESENTATION!")

print(f"\nGenerated Files:")
print(f"  1. {report_file}")
print(f"  2. {summary_plot}")
print(f"  3. {growth_file}")
print(f"\nAll visualizations available in: {PLOTS_DIR}/")
print(f"All models saved in: {MODEL_DIR}/")