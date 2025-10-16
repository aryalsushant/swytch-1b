import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
# import seaborn as sns

import os

os.makedirs('plots', exist_ok=True)

print("economic indicators eda")

# load FRED datasets
try:
    unrate = pd.read_csv('data/raw/economic-indicators/UNRATE.csv')
    payems = pd.read_csv('data/raw/economic-indicators/PAYEMS.csv')
    gdpc1 = pd.read_csv('data/raw/economic-indicators/GDPC1.csv')
    indeed = pd.read_csv('data/raw/economic-indicators/IHLIDXUS.csv')
    print("Datasets loaded successfully.")


except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the dataset is available in the specified path.")
    

datasets = {
    'Unemployment Rate': unrate,
    'Total Employment': payems,
    'Real GDP': gdpc1,
    'Indeed Job Postings': indeed
}

for name, df in datasets.items():
    print(f"\n{name}:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:\n{df.head(3)}")


    # Convert observation_date to datetime
    for name, df in datasets.items():
        if 'observation_date' in df.columns:
            df['observation_date'] = pd.to_datetime(df['observation_date'])
            print(f"{name} date range: {df['observation_date'].min()} to {df['observation_date'].max()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"{name} missing values:\n{missing}")


# time series visualizations

# can only view one at a time for now so comment out the other figures 

# Unemployment Rate
# UNCOMMENT TO BELOW VIEW FIGURE

plt.figure(figsize=(10, 6))
plt.plot(unrate['observation_date'], unrate['UNRATE'], color='red', linewidth=2)
plt.title('Unemployment Rate (2019-2025)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Percent')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/unemployment_rate.png', dpi=150, bbox_inches='tight')
plt.show()

# Total Employment
# UNCOMMENT TO BELOW VIEW FIGURE

# plt.figure(figsize=(10, 6))
# plt.plot(payems['observation_date'], payems['PAYEMS'], color='blue', linewidth=2)
# plt.title('Total Nonfarm Employment (2019-2025)', fontsize=14)
# plt.xlabel('Date')
# plt.ylabel('Thousands of Jobs')
# plt.grid(True, alpha=0.3)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('plots/total_employment.png', dpi=150, bbox_inches='tight')
# plt.show()

# Real GDP
# UNCOMMENT TO BELOW VIEW FIGURE

# plt.figure(figsize=(10, 6))
# plt.plot(gdpc1['observation_date'], gdpc1['GDPC1'], color='green', linewidth=2, marker='o')
# plt.title('Real GDP (Quarterly, 2019-2025)', fontsize=14)
# plt.xlabel('Date')
# plt.ylabel('Billions of Dollars')
# plt.grid(True, alpha=0.3)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('plots/real_gdp.png', dpi=150, bbox_inches='tight')
# plt.show()

# Indeed Job Postings
# UNCOMMENT TO BELOW VIEW FIGURE

# plt.figure(figsize=(10, 6))
# plt.plot(indeed['observation_date'], indeed['IHLIDXUS'], color='orange', linewidth=1)
# plt.title('Indeed Job Postings Index (Daily, 2020-2025)', fontsize=14)
# plt.xlabel('Date')
# plt.ylabel('Index (Feb 2020 = 100)')
# plt.axhline(y=100, color='black', linestyle='--', alpha=0.7, label='Pre-pandemic baseline')
# plt.grid(True, alpha=0.3)
# plt.xticks(rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.savefig('plots/indeed_job_postings.png', dpi=150, bbox_inches='tight')
# plt.show()

# basic statistics and observations/patterns
print("\n" + "="*50)
print("KEY FINDINGS:")
print("="*50)

for name, df in datasets.items():
    if len(df) > 0:
        value_col = df.columns[1]  # second col is the data
        values = df[value_col].dropna()
        
        print(f"\n{name}:")
        print(f"  Mean: {values.mean():.2f}")
        print(f"  Min: {values.min():.2f} | Max: {values.max():.2f}")
        print(f"  Standard deviation: {values.std():.2f}")
        
        # to look for COVID impact (around 2020)
        covid_period = df[df['observation_date'].dt.year == 2020] if 'observation_date' in df.columns else None
        if covid_period is not None and len(covid_period) > 0:
            covid_values = covid_period[value_col].dropna()
            if len(covid_values) > 0:
                print(f"  2020 average: {covid_values.mean():.2f}")


# Convert dates for analysis
unrate['observation_date'] = pd.to_datetime(unrate['observation_date'])
payems['observation_date'] = pd.to_datetime(payems['observation_date'])
gdpc1['observation_date'] = pd.to_datetime(gdpc1['observation_date'])
indeed['observation_date'] = pd.to_datetime(indeed['observation_date'])

print("\n" + "="*50)
print("KEY FINDINGS:")
print("="*50)

# Unemployment Rate Analysis
print(f"\nUnemployment Rate:")
print(f"  Range: {unrate['UNRATE'].min():.1f}% to {unrate['UNRATE'].max():.1f}%")
print(f"  Current (latest): {unrate['UNRATE'].iloc[-1]:.1f}%")

# Find COVID peak (2020)
covid_2020 = unrate[unrate['observation_date'].dt.year == 2020]
if not covid_2020.empty:
    print(f"  COVID peak (2020): {covid_2020['UNRATE'].max():.1f}%")

# Total Employment Analysis
print(f"\nTotal Employment:")
employment_start = payems['PAYEMS'].iloc[0]
employment_end = payems['PAYEMS'].iloc[-1]
print(f"  Start (2019): {employment_start:,.0f} thousand")
print(f"  Current: {employment_end:,.0f} thousand")
print(f"  Net change: {employment_end - employment_start:,.0f} thousand jobs")

# Indeed Job Postings Analysis
print(f"\nIndeed Job Postings (Feb 2020 = 100):")
print(f"  Current level: {indeed['IHLIDXUS'].iloc[-1]:.1f}")
print(f"  Peak: {indeed['IHLIDXUS'].max():.1f}")
print(f"  Low: {indeed['IHLIDXUS'].min():.1f}")

print(f"\nData Summary:")
print(f"  Unemployment: {len(unrate)} monthly observations")
print(f"  Employment: {len(payems)} monthly observations")
print(f"  GDP: {len(gdpc1)} quarterly observations")
print(f"  Job Postings: {len(indeed)} daily observations")