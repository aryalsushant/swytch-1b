# Labor Market Forecasting - Aerospace Industry

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Overview](#overview) • [Key Results](#-key-results) • [Getting Started](#-getting-started) • [Models](#-models) • [Visualizations](#-visualizations) • [Future Work](#-future-work)

---

## 📊 Overview

**Swytch 1B** is a building a comprehensive time-series forecasting system that predicts aerospace industry employment trends at both national and state levels. By combining official Bureau of Labor Statistics (BLS) employment data with macroeconomic indicators, we achieve state-of-the-art prediction accuracy to help job seekers, employers, and workforce planners make data-driven decisions.

### 🎯 Problem Statement

The aerospace industry experienced unprecedented volatility during 2020-2025, making workforce planning extremely challenging. Traditional forecasting methods struggle to account for:

- **COVID-19 impact** and recovery patterns  
- **Regional variations** in aerospace employment  
- **Economic indicators** influencing hiring decisions  
- **Seasonal patterns** in manufacturing employment  

### 💡 Our Solution

We developed a dual-model approach:

1. **ARIMA** for national-level trend forecasting  
2. **XGBoost** for state-level predictions with economic features  

This hybrid system provides both macro and micro insights into aerospace employment dynamics.

---

## 🏆 Key Results

| Model | Level | MAPE | RMSE | R² | Use Case |
|-------|-------|------|------|-----|----------|
| **ARIMA (1,1,1)** | National | **0.25%** | 1,232 | 0.9987 | National trends & policy |
| **XGBoost** | National | 0.43% | 2,103 | 0.9965 | Feature importance analysis |
| **XGBoost Multi-State** | State | **3.56%** | 2,847 | 0.9360 | Regional workforce planning |

### 📈 Performance Highlights

- **California**: 0.47% MAPE (best state-level performance)  
- **National Recovery**: Predicted 4.3% overall growth from COVID lows  
- **Top Growth States**: Florida (+26.3%), Georgia (+18.9%)  
- **Employment Leaders**: California (94K), Washington (81K), Texas (50K)  

---

## 🗂️ Dataset

### Data Sources

We integrated multiple authoritative data sources to create a comprehensive dataset:

| Source | Description | Frequency | Coverage |
|--------|-------------|-----------|----------|
| **BLS QCEW** | Aerospace employment (NAICS 3364) | Monthly | Jan 2020 - Mar 2025 |
| **FRED Unemployment** | U.S. unemployment rate | Monthly | 2019-2025 |
| **FRED GDP** | Real GDP | Quarterly | 2019-2025 |
| **Indeed Job Postings** | Job posting volume index | Daily → Monthly | 2020-2025 |

### Coverage

- **States**: 10 top aerospace employment states  
- **Time Period**: 63 months  
- **Total Records**: 630  
- **Economic Indicators**: 4  

### Data Processing Pipeline

```
import pandas as pd
df = pd.read_csv('data/processed/state_aerospace_complete.csv')
```

---

## 🚀 Getting Started

### Install

```
git clone https://github.com/aryalsushant/swytch-1b.git
cd swytch-1b
pip install -r requirements.txt
```

---

## 🤖 Models

### ARIMA (National)

- ARIMA(1,1,1)
- MAPE: 0.25%
- RMSE: 1,232

### XGBoost (State-Level)

- 100 trees  
- MAPE: 3.56%  
- R²: 0.9360  

---

## 📊 Visualizations

Plots included in `plots/`.

---

## 📁 Project Structure

```
swytch-1b/
├── data/
├── scripts/
├── models/
├── plots/
└── README.md
```

---

## 🔬 Methodology

Includes EDA, feature engineering, ARIMA + XGBoost modeling, evaluation.

---

## 📈 Results & Insights

- 17% national post-COVID recovery  
- Florida fastest-growing state  
- GDP and job postings positively correlated with employment  

---

## 🔮 Future Work

- Better seasonality  
- Walk-forward CV  
- LSTM models  
- Real-time dashboard  

---

## 🤝 Contributing

Only Team swytch-1b members can contribute at this time

---

## 📄 License

MIT License.

---

## 👥 Team

Team Swytch 1B





