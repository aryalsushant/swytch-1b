# process state level zerospace cata
# uses exact row/column positions

import pandas as pd
import numpy as np
from pathlib import Path

# setup
INPUT_FILE = Path("data/raw/swytch_data.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

print("PROCESSING STATE-LEVEL AEROSPACE DATA (CLEAN)")

# read the entire CSV
df = pd.read_csv(INPUT_FILE, header=None)

print(f"\nRaw data shape: {df.shape}")

# PART 1: STATE EMPLOYMENT DATA (Rows 0-10, Columns B-BL)
print("EXTRACTING STATE EMPLOYMENT DATA")

# row 0: column headers (dates)
date_headers = df.iloc[0, 1:64].values  # B1:BL1

# rows 1-10: state data
state_names = df.iloc[1:11, 0].values  # A2:A11
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

# PART 2: UNEMPLOYMENT DATA (Rows 15-77, Columns A-D)
print("EXTRACTING UNEMPLOYMENT DATA")

unemployment_records = []
for row_idx in range(15, 78):  # rows 16-78 in excel = 15-77 in python (0-indexed)
    date = df.iloc[row_idx, 0]  # col A
    unemp_rate = df.iloc[row_idx, 1]  # col B
    
    if pd.notna(date):
        try:
            unemployment_records.append({
                'date': pd.to_datetime(date),
                'unemployment_rate': float(unemp_rate) if pd.notna(unemp_rate) else np.nan
            })
        except:
            pass

unemp_df = pd.DataFrame(unemployment_records)
print(f"Unemployment data: {len(unemp_df)} months")

# PART 3: GDP DATA (Rows 80-141, Columns C-D)
print("EXTRACTING GDP DATA")

gdp_records = []
for row_idx in range(80, 142):  # rows 81-142 in excel = 80-141 in python
    date = df.iloc[row_idx, 2]  # col C (Dates, Monthly)
    gdp_value = df.iloc[row_idx, 3]  # col D (GDP Linear Extrapolation)
    
    if pd.notna(date):
        try:
            # handle both date formats
            parsed_date = pd.to_datetime(date)
            gdp_records.append({
                'date': parsed_date,
                'gdp': float(gdp_value) if pd.notna(gdp_value) else np.nan
            })
        except:
            pass

gdp_df = pd.DataFrame(gdp_records)
print(f"GDP data: {len(gdp_df)} months")

# PART 4: JOB POSTINGS DATA (Rows 82-143, Columns F-G)
print("EXTRACTING JOB POSTINGS DATA")

job_postings_records = []
for row_idx in range(82, 144):  # rows 83-144 in excel = 82-143 in python
    date = df.iloc[row_idx, 5]  # col F (Month)
    index_value = df.iloc[row_idx, 6]  # col G (Index Monthly Avg)
    
    if pd.notna(date):
        try:
            parsed_date = pd.to_datetime(date)
            job_postings_records.append({
                'date': parsed_date,
                'job_postings_index': float(index_value) if pd.notna(index_value) else np.nan
            })
        except:
            pass

job_postings_df = pd.DataFrame(job_postings_records)
print(f"Job postings data: {len(job_postings_df)} months")

# MERGE ALL DATA
print("MERGING ALL DATA")

# merge economic indicators
econ_df = unemp_df.merge(gdp_df, on='date', how='outer')
econ_df = econ_df.merge(job_postings_df, on='date', how='outer')
econ_df = econ_df.sort_values('date').reset_index(drop=True)

print(f"Economic indicators combined: {len(econ_df)} months")

# save economic indicators
econ_file = OUTPUT_DIR / "economic_indicators_complete.csv"
econ_df.to_csv(econ_file, index=False)
print(f"Saved: {econ_file}")

# merge state data with economic indicators
state_with_econ = state_df.merge(econ_df, on='date', how='left')

print(f"\nMerged state + economic data: {state_with_econ.shape}")

# save merged data
merged_file = OUTPUT_DIR / "state_aerospace_complete.csv"
state_with_econ.to_csv(merged_file, index=False)
print(f"Saved: {merged_file}")

# SUMMARY STATISTICS
print("DATA SUMMARY")

# recent state averages (2024-2025)
recent = state_with_econ[state_with_econ['date'] >= '2024-01-01']
state_avg = recent.groupby('state')['employment'].mean().sort_values(ascending=False)

print("\nAverage Aerospace Employment by State (2024-2025):")
for state, avg in state_avg.items():
    print(f"  {state}: {avg:,.0f}")

# data completeness
print(f"\nData Completeness:")
print(f"  Unemployment: {unemp_df['unemployment_rate'].notna().sum()}/{len(unemp_df)}")
print(f"  GDP: {gdp_df['gdp'].notna().sum()}/{len(gdp_df)}")
print(f"  Job Postings: {job_postings_df['job_postings_index'].notna().sum()}/{len(job_postings_df)}")

print("PROCESSING COMPLETE")