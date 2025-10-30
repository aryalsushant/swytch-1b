# Economic Indicators Documentation

## Overview
This dataset contains economic indicators (2019-2025) for forecasting employment trends.

## Variables

### unemployment_rate
- **Source:** FRED (Federal Reserve Economic Data)
- **Frequency:** Monthly
- **Range:** 3.4% to 14.8%
- **Description:** Percentage of labor force that is unemployed
- **Key Pattern:** Spiked to 14.8% in April 2020 (COVID), recovered to 4.3%

### total_employment
- **Source:** FRED - BLS Total Nonfarm Payroll
- **Frequency:** Monthly
- **Range:** 130,424 to 159,540 thousand jobs
- **Description:** Total number of employed people (in thousands)
- **Key Pattern:** Lost 20M jobs in 2020, recovered to record 159.5M

### real_gdp
- **Source:** FRED - Bureau of Economic Analysis
- **Frequency:** Quarterly
- **Range:** $19,056B to $23,703B (2017 dollars)
- **Description:** Inflation-adjusted economic output
- **Key Pattern:** V-shaped recovery from 2020 crash

### job_postings_index
- **Source:** Indeed via FRED
- **Frequency:** Monthly average (originally daily)
- **Baseline:** February 2020 = 100
- **Range:** 61.8 to 161.5
- **Description:** Volume of job postings relative to pre-pandemic
- **Key Pattern:** Crashed to 61.8, surged to 161.5, now at 104.5

## Data Quality
- **Missing Values:** None in unemployment, employment, job postings
- **GDP:** Has NaN for non-quarter months (expected)
- **Time Period:** 2019-01-01 to 2025-08-01
- **Ready for:** XGBoost modeling as features

## Usage Notes
- GDP should be forward-filled for monthly modeling or kept as quarterly
- Indeed index baseline is Feb 2020 (pre-pandemic)
- All data is seasonally adjusted except where noted