# Data Cleaning Decisions


### `cpsaat11b_cleaned.csv` - Not Using
**What it contains:** 599 detailed occupation categories with age breakdowns

**Why we're not using it:**
- **Too granular for initial modeling** - 599 occupations vs 30 in cpsaat09 would make the model unnecessarily complex
- **Single time point limitation** - Only contains 2024 data, not time series data needed for forecasting
- **Project scope** - Our goal is to identify broad career trends, not predict every specific job title
- **Advisor guidance** - recommended starting with manageable dataset and 60/3 month train/test split

**Decision:** Use cpsaat09 (30 broad occupation categories) for cleaner, more interpretable models. Can revisit detailed occupations if time permits after completing initial XGBoost and ARIMA models.

### `cpsaat14_cleaned.csv` - Not Using  
**What it contains:** 88 rows of employment data broken down by industry, age, sex, race, and ethnicity

**Why we're not using it:**
- **Different analysis dimension** - This file analyzes by *industry* (manufacturing, retail, etc.), but our project focuses on *occupations* (managers, engineers, etc.)
- **Demographic breakdowns not needed** - Age/sex/race breakdowns don't align with our forecasting goal of predicting career trends using economic indicators
- **Private sector filter already applied** - Initially planned to use this to filter private vs government sector, but cpsaat09 occupation data doesn't require this distinction
- **Redundant with economic indicators** - Total employment trends already captured in our economic_indicators_ready.csv

**Decision:** Focus on occupation-based forecasting (cpsaat09) rather than industry-based analysis. This aligns better with our goal of helping job seekers understand career paths.

## What We ARE Using

### `cpsaat09_cleaned.csv` 
- **30 occupation categories** (Management, Healthcare, Technology, etc.)
- **2024 employment numbers** 
- Broad enough for trend identification, specific enough to be actionable

### `train_data.csv` & `test_data.csv` 
- **81 months of economic indicators** (2019-2025)
  - Unemployment rate
  - Total employment  
  - Real GDP
  - Job postings index
- **78/3 month train/test split** (per recommendation for model validation)
- **Time-series format** ready for XGBoost and ARIMA modeling

## Modeling Approach
1. Predict total employment using economic indicators
2. Compare XGBoost vs ARIMA (or Random Forest)
3. Evaluate with MAPE, R², RMSE
4. Use 60-month training, 3-month test for out-of-sample validation
5. Create time series and bar graph visualizations

## Future Considerations

If time permits after completing initial models:
- Could incorporate cpsaat11b for occupation-specific predictions
- Could add cpsaat14 for industry-level analysis
- Could expand to predict employment by both occupation AND industry



