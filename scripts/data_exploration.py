# for learning purposes

import pandas as pd
import matplotlib.pyplot as plt

file1 = pd.read_excel('data/raw/bls_employment/cpsaat09.xlsx')
file2 = pd.read_excel('data/raw/bls_employment/cpsaat11b.xlsx')
file3 = pd.read_excel('data/raw/bls_employment/cpsaat14.xlsx')

print(file1.shape)
print(file2.shape)
print(file3.shape)

print("First few rows of file1:")
print(file1.head(10))

# find rows that look like they contain employment numbers
employment_data = []
for i in range(len(file1)):
    row = file1.iloc[i]
    first_col = str(row.iloc[0])

    if pd.isna(row.iloc[0]) or len(first_col) < 5:
        continue

    for j in range(1, len(row)):
        val = row.iloc[j]
        if pd.notna(val):
            try:
                num = float(str(val).replace(',', ''))
                if num > 100:
                    employment_data.append((first_col, num))
                    break
            except:
                continue

print(f"found {len(employment_data)} job categories:")
for job, count in employment_data[:15]:
    print(f"{job}: {count:,.0f}k")