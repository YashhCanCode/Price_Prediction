# House Price Predictor

## Overview
A machine learning project predicting house sale prices using the Ames Housing 
dataset, comparing Linear Regression, Random Forest, and Gradient Boosting.

## Dataset
- Ames Housing Dataset (Kaggle)
- 1,460 houses, 79 features
- Target variable: SalePrice ($34,900 — $755,000)

## Key Findings from EDA
- 19 columns had missing values — handled based on domain understanding
- Most NaN values meant "feature doesn't exist" (no pool, no garage etc.)
- Price distribution is right-skewed — few very expensive houses pull the average up

## Feature Engineering
Created 4 new features that outperformed raw columns:
- TotalSF — combined all floor areas (0.78 correlation with price)
- HouseAge — years between built and sold
- TotalBath — weighted combination of all bathroom types
- Remodelled — binary flag for renovated houses

TotalSF, HouseAge and TotalBath ranked in top 5 most important features.

## Results

| Model | R² | RMSE |
|-------|----|------|
| Linear Regression | 0.46 | $64,566 |
| Random Forest | 0.88 | $29,767 |
| Gradient Boost | 0.91 | $25,991 |

## Key Insight
Linear Regression failed (R² 0.46) because house prices have complex 
non-linear relationships. Gradient Boost succeeded by sequentially correcting 
errors — matching the complexity of real estate pricing.

## Tech Stack
Python, pandas, scikit-learn, matplotlib, seaborn, Google Colab

## Files
- `price_predictor.ipynb` — full notebook with code, outputs, and charts
- `train.csv` — dataset
