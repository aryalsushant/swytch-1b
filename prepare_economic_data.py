import pandas as pd
import os

print("Preparing economic indicators for modeling...")
os.makedirs('data/processed', exist_ok=True)

# load all FRED data
unrate = pd.read_csv('data/raw/economic-indicators/UNRATE.csv')
payems = pd.read_csv('data/raw/economic-indicators/PAYEMS.csv')
gdpc1 = pd.read_csv('data/raw/economic-indicators/GDPC1.csv')
indeed = pd.read_csv('data/raw/economic-indicators/IHLIDXUS.csv')

# convert dates to datetime
unrate['observation_date'] = pd.to_datetime(unrate['observation_date'])
payems['observation_date'] = pd.to_datetime(payems['observation_date'])
gdpc1['observation_date'] = pd.to_datetime(gdpc1['observation_date'])
indeed['observation_date'] = pd.to_datetime(indeed['observation_date'])

# renamed columns 
unrate = unrate.rename(columns={'UNRATE': 'unemployment_rate'})
payems = payems.rename(columns={'PAYEMS': 'total_employment'})
gdpc1 = gdpc1.rename(columns={'GDPC1': 'real_gdp'})
indeed = indeed.rename(columns={'IHLIDXUS': 'job_postings_index'})

# merge unemployment and employment (both monthly)
economic = unrate.merge(payems, on='observation_date', how='outer')

# since indeed data is daily data, resample to monthly averages first
indeed['year_month'] = indeed['observation_date'].dt.to_period('M')
indeed_monthly = indeed.groupby('year_month')['job_postings_index'].mean().reset_index()
indeed_monthly['observation_date'] = indeed_monthly['year_month'].dt.to_timestamp()
indeed_monthly = indeed_monthly[['observation_date', 'job_postings_index']]

# then merge indeed data
economic = economic.merge(indeed_monthly, on='observation_date', how='outer')

# since GDP is quarterly, merge but it will have NaN for non-quarter months
economic = economic.merge(gdpc1, on='observation_date', how='outer')

# sort by date
economic = economic.sort_values('observation_date').reset_index(drop=True)

# save cleaned economic indicators
economic.to_csv('data/processed/economic_indicators_ready.csv', index=False)
print(f"Saved {len(economic)} rows of economic data") # to check
print(f"Date range: {economic['observation_date'].min()} to {economic['observation_date'].max()}")
print(f"\nColumns: {list(economic.columns)}")