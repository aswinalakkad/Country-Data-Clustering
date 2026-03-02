
# 🌍 Country Development Status Predictor

A machine learning web application that classifies countries into development categories based on socio-economic indicators.

## Overview

This app was built for **Help International Foundation** to assist in identifying underdeveloped and developing countries that may need aid and resources. It uses unsupervised clustering (trained offline) to predict a country's development status from key economic and health metrics.

## Live Demo

🔗 [Streamlit App](https://streamlit.io) 

## Features

- Predicts whether a country is **Developed**, **Developing**, or **Under Developed**
- Simple, user-friendly web interface built with Streamlit
- Powered by a PCA-reduced, scaled ML pipeline

## Input Features

| Feature | Description |
|---|---|
| Child Mortality Rate | Deaths of children under 5 per 1,000 live births |
| Exports | Exports as % of GDP |
| Imports | Imports as % of GDP |
| Health Expenditure | Health spending as % of GDP |
| Income | Average income per person |
| Inflation | Inflation rate (%) |
| Life Expectancy | Average life expectancy (years) |
| Total Fertility | Avg. children born per woman |
| GDP per Capita | GDP per capita (USD) |

## Tech Stack

- **Frontend:** Streamlit
- **ML Pipeline:** Scikit-learn (Scaler → PCA → Clustering Model)
- **Language:** Python

## Setup & Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Install dependencies
pip install streamlit numpy pandas scikit-learn

# Run the app
streamlit run app.py
```

> **Note:** Ensure `final_model.pkl`, `pca.pkl`, and `scaler.pkl` are present in the project root before running.

## Project Structure

```
├── app.py               # Main Streamlit application
├── final_model.pkl      # Trained clustering model
├── pca.pkl              # Fitted PCA transformer
├── scaler.pkl           # Fitted data scaler
└── README.md
```

## Model Pipeline

1. Raw inputs are **standardized** using a pre-fitted `StandardScaler`
2. Dimensionality is reduced using **PCA**
3. The clustering model predicts the **development class**

---
