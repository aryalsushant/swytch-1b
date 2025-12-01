# merge Employment Data with economic indicators
# purpose: Combine cleaned BLS employment data with economic indicators.
# creates final model-ready dataset with a 60/3 month train/test split.


import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# detect whether script is being run from /scripts or project root
if os.path.basename(os.getcwd()) == "scripts":
    PROCESSED_DIR = Path("../data/processed")
else:
    PROCESSED_DIR = Path("data/processed")

ECONOMIC_INDICATORS = PROCESSED_DIR / "economic_indicators_ready.csv"

def load_economic_indicators():
    # load the cleaned economic indicator data
    df = pd.read_csv(ECONOMIC_INDICATORS)
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    return df

def load_employment_data():
    # load cleaned BLS employment data from the processed folder
    cleaned_files = list(PROCESSED_DIR.glob("cpsaat*_cleaned.csv"))
    if not cleaned_files:
        raise FileNotFoundError("No cleaned employment files found. Run clean_bls_employment_data.py first.")
    
    employment_dfs = {}
    for file in cleaned_files:
        df = pd.read_csv(file)
        name = file.stem.replace('_cleaned', '')
        employment_dfs[name] = df
    return employment_dfs

def prepare_employment_for_merge(employment_dfs):
    # quick inspection of employment data to understand structure
    for name, df in employment_dfs.items():
        print(f"{name}: {df.shape}, columns: {df.columns.tolist()[:5]}...")
    return employment_dfs

def merge_datasets(economic_df, employment_dfs):
    # merge employment data with economic indicators based on date
    merged = economic_df.copy()
    # later: integrate employment_dfs once structure is finalized
    return merged

def create_train_test_split(df, test_months=3):
    # split dataset into training (60 months) and testing (3 months)
    df = df.sort_values('observation_date').reset_index(drop=True)
    train = df.iloc[:-test_months].copy()
    test = df.iloc[-test_months:].copy()
    return train, test

def save_model_ready_data(train, test, merged):
    # save merged data and train/test splits to the processed folder
    OUTPUT_DIR = Path("data/processed")
    merged.to_csv(OUTPUT_DIR / "employment_with_economic_indicators.csv", index=False)
    train.to_csv(OUTPUT_DIR / "train_data.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test_data.csv", index=False)

def main():
    # main function for the merge pipeline
    economic_df = load_economic_indicators()
    employment_dfs = load_employment_data()
    employment_prepared = prepare_employment_for_merge(employment_dfs)
    merged = merge_datasets(economic_df, employment_prepared)
    train, test = create_train_test_split(merged, test_months=3)
    save_model_ready_data(train, test, merged)

if __name__ == "__main__":
    main()
