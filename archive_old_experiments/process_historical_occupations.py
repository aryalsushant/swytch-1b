"""
Historical Occupation Data Processing
Load 2019-2023 occupation data and create time-series dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

# file paths
DATA_DIR = Path("data/raw/bls_employment")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

print("HISTORICAL OCCUPATION DATA PROCESSING")

# load all years
years = [2019, 2020, 2021, 2022, 2023]
occupation_dfs = []

print("\nLoading historical data...")

for year in years:
    filename = DATA_DIR / f"cpsaat11b_{year}.xlsx"
    
    try:
        # load file
        df = pd.read_excel(filename)
        
        # skip header rows (first 4 rows are headers)
        df = df.iloc[4:].reset_index(drop=True)
        
        # extract occupation name and total employment
        # column 0: occupation name
        # column 1: total employment (16 years and over)
        df_clean = pd.DataFrame({
            'occupation': df.iloc[:, 0],
            'employment': df.iloc[:, 1],
            'year': year
        })
        
        # remove rows with NaN occupation names or "NOTE:" text
        df_clean = df_clean[df_clean['occupation'].notna()]
        df_clean = df_clean[~df_clean['occupation'].str.contains('NOTE:', na=False)]
        df_clean = df_clean[~df_clean['occupation'].str.contains('Numbers in thousands', na=False)]
        df_clean = df_clean[~df_clean['occupation'].str.contains('Occupation', na=False)]
        
        # convert employment to numeric
        df_clean['employment'] = pd.to_numeric(df_clean['employment'], errors='coerce')
        
        # remove any remaining NaN employment values
        df_clean = df_clean.dropna(subset=['employment'])
        
        occupation_dfs.append(df_clean)
        
        print(f"  Loaded {year}: {len(df_clean)} occupations")
        
    except Exception as e:
        print(f"  Error loading {year}: {e}")

# combine all years
print("\nCombining all years...")
occupation_timeseries = pd.concat(occupation_dfs, ignore_index=True)

print(f"Total records: {len(occupation_timeseries)}")
print(f"Years covered: {occupation_timeseries['year'].unique()}")
print(f"Unique occupations: {occupation_timeseries['occupation'].nunique()}")

# check data structure
print("\nSample data:")
print(occupation_timeseries.head(20))

# save combined dataset
output_file = OUTPUT_DIR / "occupation_timeseries_2019_2023.csv"
occupation_timeseries.to_csv(output_file, index=False)
print(f"\nSaved: {output_file}")

# create pivot table for easier analysis (occupations as rows, years as columns)
print("\nCreating pivot table...")
pivot = occupation_timeseries.pivot(index='occupation', columns='year', values='employment')

print(f"\nPivot shape: {pivot.shape}")
print("\nSample pivot (first 10 occupations):")
print(pivot.head(10))

# save pivot table
pivot_file = OUTPUT_DIR / "occupation_pivot_2019_2023.csv"
pivot.to_csv(pivot_file)
print(f"\nSaved pivot: {pivot_file}")

# calculate year-over-year growth
print("\nCalculating growth trends...")
pivot['growth_2019_2023'] = ((pivot[2023] - pivot[2019]) / pivot[2019] * 100)
pivot['avg_annual_growth'] = pivot['growth_2019_2023'] / 4

# sort by growth
top_growing = pivot.nlargest(10, 'avg_annual_growth')[['avg_annual_growth', 2019, 2023]]
top_declining = pivot.nsmallest(10, 'avg_annual_growth')[['avg_annual_growth', 2019, 2023]]

print("\nTop 10 Growing Occupations (2019-2023):")
print(top_growing)

print("\nTop 10 Declining Occupations (2019-2023):")
print(top_declining)

# save growth analysis
growth_file = OUTPUT_DIR / "occupation_growth_analysis.csv"
growth_df = pivot[['avg_annual_growth', 2019, 2023, 'growth_2019_2023']].copy()
growth_df = growth_df.sort_values('avg_annual_growth', ascending=False)
growth_df.to_csv(growth_file)
print(f"\nSaved growth analysis: {growth_file}")

print("PROCESSING COMPLETE")
