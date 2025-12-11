# Labor Market Forecasting in the Aerospace Industry

## 👥 Team Members
| Name            | GitHub Handle     | Contribution |
|-----------------|-------------------|--------------|
| Sushant Aryal   | @aryalsushant     | Modeling, forecasting system, data engineering |
| Joy Avre        | @avrejoy          | Exploratory data analysis, dataset cleanup |
| Lutfiyah Nawaz  | @Luffy28          | Feature engineering, preprocessing |
| Stephanie Ngu   | @StephanieNgu     | Visualization, results interpretation |
| Deeya Rawat     | @dya9             | Presentation development, storytelling, coordination |

## 🎯 Project Highlights
- Built a forecasting system to predict aerospace employment at both national and state levels.
- Used **ARIMA** for national forecasting and **XGBoost** for state-level modeling.
- Reached **0.25% MAPE** (national) and **3.56% MAPE** (state).
- Identified trends in post-COVID recovery, high-growth states, and macroeconomic drivers.

## 👩🏽‍💻 Setup and Installation
```
git clone https://github.com/aryalsushant/swytch-1b.git
cd swytch-1b
pip install -r requirements.txt
```

## 🏗️ Project Overview
This project was developed for the Break Through Tech AI – AI Studio Challenge.
The goal was to analyze aerospace employment from 2020–2025 and build forecasting tools to support workforce decision-making.

## 📊 Data Exploration
### Data Sources
| Source | Description | Frequency | Coverage |
|--------|-------------|-----------|----------|
| BLS QCEW | Aerospace employment (NAICS 3364) | Monthly | 2020–2025 |
| FRED Unemployment | National unemployment rate | Monthly | 2019–2025 |
| FRED GDP | Real GDP | Quarterly | 2019–2025 |
| Indeed Job Postings | Job posting index | Monthly | 2020–2025 |

### EDA Insights
- COVID-19 caused a steep employment decline followed by uneven recovery.
- Strong correlation between employment, GDP, unemployment rate, and job postings.
- Some states show stronger seasonality patterns.
- Missing data required interpolation in specific months.

## 🧠 Model Development
### Models Used
1. **ARIMA (National Level)**
   - Final model: ARIMA(1,1,1)

2. **XGBoost (State Level)**
   - Includes lagged variables and rolling averages.

### Training Setup
- Chronological train-test split
- Metrics: MAPE, RMSE, R²
- Baseline: naive rolling forecast

## 📈 Results & Key Findings
### Performance Metrics
| Model | Level | MAPE | RMSE | R² |
|-------|--------|------|------|------|
| ARIMA | National | 0.25% | 1,232 | 0.9987 |
| XGBoost | National | 0.43% | 2,103 | 0.9965 |
| XGBoost Multi-State | State | 3.56% | 2,847 | 0.9360 |

### Insights
- National employment recovered ~17% from COVID-era lows.
- Fastest-growing states: Florida and Georgia.
- Top predictors: GDP, job posting volume.

## 🚀 Next Steps
- Explore SARIMAX or Prophet for stronger seasonality handling.
- Use walk-forward validation for time-series robustness.
- Try LSTMs or other deep learning architectures.
- Build a real-time workforce dashboard.

## 📝 License
MIT License

## 📄 References
- BLS QCEW
- FRED datasets
- ARIMA (statsmodels)
- XGBoost documentation

## 🙏 Acknowledgements
Thanks to Coach Elizabeth Parnell and Challenge Advisors Tim Liu and James Thompson.
