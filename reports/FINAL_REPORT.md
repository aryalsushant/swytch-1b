# Aerospace Employment Analysis - Final Report

## Project Overview

- **Time Period:** January 2020 to March 2025
- **Data Sources:** BLS QCEW (aerospace industry NAICS 3364), FRED economic indicators
- **States Analyzed:** 10
- **Total Records:** 630

## Model Results

### National-Level Forecasting
- **ARIMA (1,1,1):** 0.25% MAPE  (Best for national forecasts)
- **XGBoost:** 0.43% MAPE

### State-Level Forecasting
- **XGBoost Multi-State Model:** 3.56% MAPE, R² = 0.936
- **Features:** Unemployment, GDP, job postings, seasonality, employment lags
- **Top Performer:** California (0.47% MAPE)

## Key Findings

1. National Recovery: Aerospace employment recovered from COVID lows, growing 4.3% overall

2. State Leaders: California (94K), Washington (81K), and Texas (50K) dominate aerospace employment

3. Best Growth: Florida (+26.3%) and Georgia (+18.9%) show strongest growth 2020-2025

4. Model Performance: ARIMA best for national forecasts (0.25% MAPE), XGBoost excellent for state-level (3.56% MAPE)

5. Economic Factors: Unemployment rate, GDP, and job postings are strong predictors

6. Seasonality: Moderate seasonal patterns with higher employment in summer months

7. State Diversity: Growth varies significantly by state - some growing 26%, others declining 9%
