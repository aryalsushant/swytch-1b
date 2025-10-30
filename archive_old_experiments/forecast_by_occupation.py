"""
Occupation - level Employment Forecasting
predicts employment trends for major occupation categories
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# file paths
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("models")
PLOTS_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

print("OCCUPATION-LEVEL FORECASTING")

# load occupation time-series data
print("\nLoading occupation data...")
occ_data = pd.read_csv(DATA_DIR / "occupation_timeseries_2019_2023.csv")
print(f"Loaded: {len(occ_data)} records")

# focus on major occupation categories (broader categories for stability)
# these are the main BLS occupation groups
major_occupations = [
    'Total employed',
    'Management, professional, and related occupations',
    'Service occupations',
    'Sales and office occupations',
    'Natural resources, construction, and maintenance occupations',
    'Production, transportation, and material moving occupations'
]

# filter to major occupations
major_occ_data = occ_data[occ_data['occupation'].isin(major_occupations)].copy()

print(f"\nFocusing on {len(major_occupations)} major occupation categories")
print(f"Filtered data: {len(major_occ_data)} records")

# load economic indicators
print("\nLoading economic indicators...")
econ_data = pd.read_csv(DATA_DIR / "economic_indicators_ready.csv")
econ_data['observation_date'] = pd.to_datetime(econ_data['observation_date'])
econ_data['year'] = econ_data['observation_date'].dt.year

# calculate annual averages of economic indicators
econ_annual = econ_data.groupby('year').agg({
    'unemployment_rate': 'mean',
    'total_employment': 'mean',
    'job_postings_index': 'mean',
    'real_gdp': 'mean'
}).reset_index()

print(f"Economic indicators by year:")
print(econ_annual)

# merge occupation data with economic indicators
print("\nMerging occupation data with economic indicators...")
merged = major_occ_data.merge(econ_annual, on='year', how='left')

print(f"Merged dataset: {merged.shape}")
print("\nSample merged data:")
print(merged.head(10))

# save merged dataset
merged_file = DATA_DIR / "occupation_with_economic_data.csv"
merged.to_csv(merged_file, index=False)
print(f"\nSaved: {merged_file}")

# calculate growth rates for each occupation
print("\nCalculating year-over-year growth rates...")

growth_data = []

for occupation in major_occupations:
    occ_subset = merged[merged['occupation'] == occupation].sort_values('year')
    
    if len(occ_subset) >= 2:
        for i in range(1, len(occ_subset)):
            prev_year = occ_subset.iloc[i-1]
            curr_year = occ_subset.iloc[i]
            
            growth_rate = ((curr_year['employment'] - prev_year['employment']) / 
                          prev_year['employment'] * 100)
            
            growth_data.append({
                'occupation': occupation,
                'year': curr_year['year'],
                'employment': curr_year['employment'],
                'prev_employment': prev_year['employment'],
                'growth_rate': growth_rate,
                'unemployment_rate': curr_year['unemployment_rate'],
                'job_postings_index': curr_year['job_postings_index'],
                'real_gdp': curr_year['real_gdp']
            })

growth_df = pd.DataFrame(growth_data)

print(f"\nGrowth analysis:")
print(growth_df.head(15))

# summary by occupation
print("\nAverage annual growth by occupation (2019-2023):")
summary = growth_df.groupby('occupation')['growth_rate'].mean().sort_values(ascending=False)
print(summary)

# save growth analysis
growth_file = DATA_DIR / "occupation_growth_with_economics.csv"
growth_df.to_csv(growth_file, index=False)
print(f"\nSaved: {growth_file}")


# simple forecast: Project 2024 based on average growth
print("2024 EMPLOYMENT FORECASTS")
forecasts = []

for occupation in major_occupations:
    occ_data = merged[merged['occupation'] == occupation].sort_values('year')
    
    if len(occ_data) > 0:
        # get 2023 employment
        employment_2023 = occ_data[occ_data['year'] == 2023]['employment'].values[0]
        
        # calculate average annual growth rate
        occ_growth = growth_df[growth_df['occupation'] == occupation]['growth_rate'].mean()
        
        # forecast 2024
        forecast_2024 = employment_2023 * (1 + occ_growth / 100)
        
        forecasts.append({
            'occupation': occupation,
            'employment_2023': employment_2023,
            'avg_growth_rate': occ_growth,
            'forecast_2024': forecast_2024,
            'forecast_change': forecast_2024 - employment_2023
        })

forecast_df = pd.DataFrame(forecasts)
forecast_df = forecast_df.sort_values('avg_growth_rate', ascending=False)

print("\nOccupation Employment Forecasts for 2024:")
print(forecast_df.to_string(index=False))

# save forecasts
forecast_file = OUTPUT_DIR / "occupation_forecasts_2024.csv"
forecast_df.to_csv(forecast_file, index=False)
print(f"\nSaved forecasts: {forecast_file}")

# identify emerging and declining careers
print("CAREER TREND ANALYSIS")

growing = forecast_df[forecast_df['avg_growth_rate'] > 0].copy()
declining = forecast_df[forecast_df['avg_growth_rate'] < 0].copy()

print(f"\nEmerging Career Paths ({len(growing)} categories):")
if len(growing) > 0:
    for _, row in growing.iterrows():
        print(f"  {row['occupation']}: +{row['avg_growth_rate']:.2f}% avg annual growth")

print(f"\nDeclining Career Paths ({len(declining)} categories):")
if len(declining) > 0:
    for _, row in declining.iterrows():
        print(f"  {row['occupation']}: {row['avg_growth_rate']:.2f}% avg annual growth")

print("ANALYSIS COMPLETE")


print("\nKey Deliverables Created:")
print(f"  1. {merged_file.name} - Occupation data with economic indicators")
print(f"  2. {growth_file.name} - Growth rates with economic context")
print(f"  3. {forecast_file.name} - 2024 employment forecasts by occupation")
