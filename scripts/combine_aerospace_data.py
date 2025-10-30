# combine downloaded QCEW files
# filters to aerospace industry (NAICS 3364) and creates clean dataset


import pandas as pd
from pathlib import Path

# setup
DATA_DIR = Path("data/raw/aerospace_qcew")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

print("PROCESSING AEROSPACE EMPLOYMENT DATA")

# find all downloaded files
csv_files = sorted(DATA_DIR.glob("qcew_*.csv"))

print(f"\nFound {len(csv_files)} files:")
for f in csv_files:
    print(f"  {f.name}")

all_data = []

print("\nProcessing files...")

for file in csv_files:
    # extract year and quarter from filename
    # format: qcew_2020_q1.csv
    parts = file.stem.split('_')
    year = int(parts[1])
    quarter = int(parts[2].replace('q', ''))
    
    try:
        # read CSV
        df = pd.read_csv(file)
        
        # filter to aerospace industry (NAICS 3364)
        aerospace = df[df['industry_code'] == '3364'].copy()
        
        if len(aerospace) > 0:
            aerospace['year'] = year
            aerospace['quarter'] = quarter
            all_data.append(aerospace)
            print(f"  {year} Q{quarter}: {len(aerospace)} aerospace records")
        else:
            print(f"  {year} Q{quarter}: No aerospace data found")
            
    except Exception as e:
        print(f"  {year} Q{quarter}: ERROR - {e}")

if len(all_data) == 0:
    print("\nNo aerospace data found in any files!")
    print("\nTroubleshooting:")
    print("  1. Check that files are in data/raw/aerospace_qcew/")
    print("  2. Open one file and check column names")
    print("  3. Look for industry_code column")
else:
    # combine all quarters
    combined = pd.concat(all_data, ignore_index=True)
    
    print("COMBINED DATASET")
    print(f"Shape: {combined.shape}")
    print(f"\nColumns: {combined.columns.tolist()}")
    
    # save full dataset
    full_output = OUTPUT_DIR / "aerospace_qcew_full.csv"
    combined.to_csv(full_output, index=False)
    print(f"\nSaved full dataset: {full_output}")
    
    # create simplified dataset with key columns
    if 'month3_emplvl' in combined.columns:
        simple = combined[['year', 'quarter', 'month3_emplvl', 'area_fips']].copy()
        simple = simple.rename(columns={'month3_emplvl': 'employment'})
        
        # filter to US total (area_fips US000 or similar)
        us_total = simple[simple['area_fips'] == 'US000'].copy()
        
        if len(us_total) > 0:
            us_total = us_total.sort_values(['year', 'quarter'])
            
            print(f"\nUS Total Aerospace Employment by Quarter:")
            print(us_total[['year', 'quarter', 'employment']].to_string(index=False))
            
            # save simplified dataset
            simple_output = OUTPUT_DIR / "aerospace_employment_quarterly.csv"
            us_total.to_csv(simple_output, index=False)
            print(f"\nSaved quarterly employment: {simple_output}")
            
            # create monthly approximation (repeat each quarter 3 times)
            monthly_data = []
            for _, row in us_total.iterrows():
                year = row['year']
                quarter = row['quarter']
                employment = row['employment']
                
                # map quarter to months
                month_start = (quarter - 1) * 3 + 1
                for month_offset in range(3):
                    month = month_start + month_offset
                    monthly_data.append({
                        'year': year,
                        'month': month,
                        'quarter': quarter,
                        'employment': employment
                    })
            
            monthly_df = pd.DataFrame(monthly_data)
            monthly_df['date'] = pd.to_datetime(
                monthly_df['year'].astype(str) + '-' + 
                monthly_df['month'].astype(str) + '-01'
            )
            
            monthly_output = OUTPUT_DIR / "aerospace_employment_monthly.csv"
            monthly_df.to_csv(monthly_output, index=False)
            print(f"Saved monthly employment: {monthly_output}")
            
            print(f"\nMonthly dataset: {len(monthly_df)} months")
            print(f"Date range: {monthly_df['date'].min()} to {monthly_df['date'].max()}")
            
        else:
            print("\nWarning: No US total data found (area_fips != US000)")
    else:
        print("\nWarning: month3_emplvl column not found")
        print(f"Available columns: {combined.columns.tolist()}")

print("PROCESSING COMPLETE")