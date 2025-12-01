# process state level zerospace cata
# uses exact row/column positions

import pandas as pd
import numpy as np
from pathlib import Path

# setup
#path to economic indicator files and output directory
ECONOMIC_FILE = Path('data/state_economic')
EMPLOYMENT_FILE = Path('data/raw/swytch_data.csv')
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("PROCESSING STATE-LEVEL AEROSPACE DATA (CLEAN)")

df = pd.read_csv(EMPLOYMENT_FILE, header=None)

print(f"\nRaw data shape: {df.shape}")

# PART 1: STATE EMPLOYMENT DATA (Rows 0-10, Columns B-BL)
print("EXTRACTING STATE EMPLOYMENT DATA")

# row 0: column headers (dates)
date_headers = df.iloc[0, 1:64].values  # B1:BL1

# rows 1-10: state data
state_names = [s.replace(' employment','').strip() for s in df.iloc[1:11, 0].values]  # A2:A11
state_data_raw = df.iloc[1:11, 1:64].values  # B2:BL11

# parse dates from headers (format: "Jan-20", "Feb-20", etc.)
dates = []
for date_str in date_headers:
    if pd.notna(date_str):
        try:
            # parse "Jan-20" format
            month_abbr, year_abbr = str(date_str).split('-')
            month_map = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            month = month_map.get(month_abbr, 1)
            year = 2000 + int(year_abbr)
            dates.append(f"{year}-{month:02d}-01")
        except:
            dates.append(None)

# create state employment dataframe
state_records = []
for i, state in enumerate(state_names):
    for j, date in enumerate(dates):
        if date and j < len(state_data_raw[i]):
            emp_value = state_data_raw[i][j]
            
            # clean employment value (remove commas)
            if pd.notna(emp_value):
                try:
                    emp = float(str(emp_value).replace(',', ''))
                    state_records.append({
                        'date': date,
                        'state': state,
                        'employment': emp
                    })
                except:
                    pass

state_df = pd.DataFrame(state_records)
state_df['date'] = pd.to_datetime(state_df['date'])
state_df = state_df.sort_values(['state', 'date']).reset_index(drop=True)

print(f"States extracted: {state_df['state'].nunique()}")
print(f"Date range: {state_df['date'].min()} to {state_df['date'].max()}")
print(f"Total records: {len(state_df)}")


# mapping for month abbreviations
month_map = {
    'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
    'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12
}

# PART 2: Process State Economic Indicator Data (Unemployment Rate, GDP)
all_files = sorted(ECONOMIC_FILE.glob("*_unemployment.xlsx"))
all_data = []

for file in all_files:
    print(f"Processing {file.name}...")
    df = pd.read_excel(file, skiprows=10)  # skip header rows
    
    df = df.rename(columns={
        'labor force participation rate': 'labor_force_participation',
        'employment-population ratio': 'employment_population_ratio',
        'labor force': 'labor_force',
        'employment': 'employment',
        'unemployment': 'unemployment',
        'unemployment rate': 'unemployment_rate',
        'GDP': 'real_gdp'
    })
    
    # extract state name
    if 'Area' in df.columns:
        state_name = df['Area'].iloc[0].strip()  # use the Area column if available
    else:
    # clean the filename
        state_name = file.stem.replace('_unemployment','').replace('_employment','').title()

    df['state'] = state_name
    
    # map period/month to numeric month
    df['month'] = df['Period'].map(month_map)
    
    # build date column
    df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['month'].astype(str) + '-01')
    
    # select relevant columns
    df_subset = df[['date','state','unemployment_rate','real_gdp']].copy()
    
    all_data.append(df_subset)

# combine all states
econ_df = pd.concat(all_data, ignore_index=True)
econ_df = econ_df.sort_values(['state','date']).reset_index(drop=True)


#PART 3: Merge Employment with Economic Indicators
combined_df = pd.merge(state_df, econ_df, on=['state','date'], how='left')

#remove the data from 4/2025-8/2025 since we are testing our model prediction 1/2025-3/2025
start_remove = pd.to_datetime("2025-04-01")
end_remove = pd.to_datetime("2025-08-31")

combined_df = combined_df[~((combined_df['date'] >= start_remove) & (combined_df['date'] <= end_remove))].reset_index(drop=True)

# save final file
OUTPUT_FILE = OUTPUT_DIR / "state_aerospace_complete.csv"
combined_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved cleaned data: {OUTPUT_FILE}")
print(combined_df.head())
