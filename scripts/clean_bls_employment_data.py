# BLS Employment Data Cleaning Script
# Purpose: Clean and prepare BLS employment data for modeling.
# - filter to private sector only
# - standardize occupation categories
# - merge datasets
# - prep for integration with economic indicators


import pandas as pd
import numpy as np
from pathlib import Path

# file paths
DATA_DIR = Path("data/raw/bls_employment")
OUTPUT_DIR = Path("data/processed")

# input files
CPSAAT09 = DATA_DIR / "cpsaat09.xlsx"   # employment by occupation
CPSAAT11B = DATA_DIR / "cpsaat11b.xlsx" # detailed occupation breakdowns
CPSAAT14 = DATA_DIR / "cpsaat14.xlsx"   # industry employment

def load_bls_file(filepath, sheet_name=0, skiprows=None):
    # load a BLS Excel file with basic error handling.
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name, skiprows=skiprows)
        return df
    except Exception as e:
        print(f"Error loading {filepath.name}: {e}")
        return None

def clean_cpsaat09(df):
    # clean cpsaat09 (employment by Occupation)
    df = df.iloc[4:].reset_index(drop=True)
    df = df[df.iloc[:, 0].notna()]
    df = df[~df.iloc[:, 0].str.contains('NOTE:', na=False)]

    df_clean = pd.DataFrame({
        'occupation': df.iloc[:, 0],
        'employment_2024': df.iloc[:, 2]
    }).dropna()

    return df_clean

def clean_cpsaat11b(df):
    # clean cpsaat11b (Detailed Occupations)
    df = df.dropna(how='all').reset_index(drop=True)
    return df

def clean_cpsaat14(df):
    # clean cpsaat14 (Industry Employment) and filter to private sector
    df = df.dropna(how='all').reset_index(drop=True)
    # actual private-sector filtering will depend on structure
    return df

def filter_private_sector(df, sector_column=None):
    # filter dataset to private sector only if column is available
    if sector_column and sector_column in df.columns:
        private_df = df[df[sector_column].str.contains('private', case=False, na=False)]
        return private_df
    else:
        return df

def standardize_employment_data(dfs_dict):
    # standardize datasets to have consistent column names
    standardized = {}
    for name, df in dfs_dict.items():
        df_std = df.copy()
        df_std.columns = (
            df_std.columns.str.lower()
            .str.replace(' ', '_')
            .str.replace('.', '')
        )
        standardized[name] = df_std
    return standardized

def create_master_employment_dataset(dfs_dict):
    # save each cleaned dataset to the processed folder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in dfs_dict.items():
        output_file = OUTPUT_DIR / f"{name}_cleaned.csv"
        df.to_csv(output_file, index=False)
    return dfs_dict

def main():
    # main function for cleaning all BLS employment datasets.
    print("Starting BLS data cleaning...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cpsaat09 = load_bls_file(CPSAAT09, skiprows=0)
    cpsaat11b = load_bls_file(CPSAAT11B, skiprows=0)
    cpsaat14 = load_bls_file(CPSAAT14, skiprows=0)

    if cpsaat09 is None or cpsaat11b is None or cpsaat14 is None:
        raise FileNotFoundError("One or more input files could not be loaded.")

    print("Cleaning individual files...")
    cpsaat09_clean = clean_cpsaat09(cpsaat09)
    cpsaat11b_clean = clean_cpsaat11b(cpsaat11b)
    cpsaat14_clean = clean_cpsaat14(cpsaat14)

    datasets = {
        'cpsaat09': cpsaat09_clean,
        'cpsaat11b': cpsaat11b_clean,
        'cpsaat14': cpsaat14_clean
    }

    standardized_datasets = standardize_employment_data(datasets)
    create_master_employment_dataset(standardized_datasets)
    print("Cleaning complete. Files saved in data/processed.")

if __name__ == "__main__":
    main()
