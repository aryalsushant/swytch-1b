Labor Market Forecasting in the Aerospace Industry
--------------------------------------------------

TEAM MEMBERS
Name | GitHub Handle | Contribution
Sushant Aryal | @aryalsushant | Modeling, forecasting framework, data engineering
Joy Avre | @avrejoy | Exploratory data analysis, dataset cleanup
Lutfiyah Nawaz | @Luffy28 | Feature engineering, preprocessing
Stephanie Ngu | @StephanieNgu | Visualization, results interpretation
Deeya Rawat | @dya9 | Presentation prep, storytelling, coordination

PROJECT HIGHLIGHTS
- Built a time-series forecasting system to predict aerospace employment trends at national and state levels.
- Used ARIMA for national forecasting and XGBoost for multi-state modeling.
- Achieved 0.25% MAPE nationally and 3.56% MAPE at the state level.
- Delivered insights on employment recovery, high-growth regions, and economic drivers.

SETUP AND INSTALLATION
git clone https://github.com/aryalsushant/swytch-1b.git
cd swytch-1b
pip install -r requirements.txt

PROJECT OVERVIEW
This project was developed as part of the Break Through Tech AI – AI Studio Challenge. 
The goal was to understand aerospace employment patterns, identify recovery trends, and build predictive tools that support workforce planning.

DATA EXPLORATION
Datasets used include BLS QCEW, FRED unemployment, FRED GDP, and Indeed job posting trends.
Insights included strong post-COVID recovery, regional variation, and correlations with macroeconomic indicators.

MODEL DEVELOPMENT
Models:
- ARIMA(1,1,1) for national-level forecasting
- XGBoost for state-level modeling with lagged features

RESULTS & KEY FINDINGS
Performance:
ARIMA National — 0.25% MAPE  
XGBoost National — 0.43% MAPE  
XGBoost State-Level — 3.56% MAPE  

Insights:
- Strong national recovery trend  
- Florida and Georgia were the fastest-growing states  
- GDP and job postings were top predictors  

NEXT STEPS
- Use SARIMAX or Prophet for improved seasonality modeling  
- Try walk-forward validation  
- Explore deep learning architectures  
- Build a real-time dashboard  

LICENSE
MIT License

REFERENCES
- BLS QCEW  
- FRED database  
- ARIMA statsmodels documentation  
- XGBoost documentation  

ACKNOWLEDGEMENTS
Thanks to Coach Elizabeth Parnell and Challenge Advisors Tim Liu and James Thompson.
