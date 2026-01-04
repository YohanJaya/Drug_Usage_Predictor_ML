# Drug Demand Prediction with XGBoost Quantile Regression

A machine learning pipeline for predicting hospital drug demand using XGBoost Quantile Regression. This project focuses on conservative demand forecasting for inventory optimization using time-series feature engineering.

## 📊 Project Overview

This project implements a quantile regression model to predict drug demand at hospitals, specifically designed for inventory management where conservative predictions (avoiding stockouts) are crucial.

**Key Features:**
- XGBoost Quantile Regression for conservative predictions
- Temporal train-test split for time-series integrity
- Comprehensive feature engineering (lag features, rolling statistics, EWM)
- Automated visualization generation
- Coverage optimization for inventory management

## 🎯 Problem Statement

**Objective:** Predict hospital drug demand using quantile regression to optimize inventory management and prevent stockouts.

**Approach:** 
- Train XGBoost models with quantile regression (90th percentile predictions)
- Engineer time-based features from historical demand data
- Evaluate using Quantile Loss and Coverage metrics

## 📁 Project Structure

```
drugUsagePredictorML/
├── BasicImplementation/
│   ├── data/
│   │   └── hospital_drug_demand.csv       # Raw data
│   ├── src/
│   │   ├── data/
│   │   │   ├── readData.py                # Data loading & feature engineering
│   │   │   └── preprocessing.py           # Train-test split
│   │   ├── models/
│   │   │   ├── train.py                   # XGBoost training
│   │   │   ├── predict.py                 # Prediction with non-negative constraint
│   │   │   └── evaluate.py                # Quantile loss & coverage metrics
│   │   └── utils/
│   │       └── visualization.py           # Plotting utilities
│   ├── reports/
│   │   └── figures/                       # Generated plots
│   ├── models/                            # Saved model files
│   ├── main.py                            # Main pipeline
│   └── venv/                              # Virtual environment
└── notebooks/
    └── Drug_Demand_Prediction_Competition.ipynb  # Jupyter notebook
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YohanJaya/Drug_Usage_Predictor_ML.git
cd Drug_Usage_Predictor_ML/BasicImplementation
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages:**
```bash
pip install pandas numpy xgboost scikit-learn matplotlib
```

### Running the Pipeline

```bash
cd BasicImplementation
python main.py
```

This will:
- Load and process Drug_1 data
- Engineer 19 time-series features
- Train XGBoost Quantile Regression model (q=0.9)
- Generate and save 4 visualization plots to `reports/figures/`
- Print performance metrics (MAE, Quantile Loss, Coverage)

## 📊 Output Visualizations

The pipeline generates 4 plots in `reports/figures/`:

1. **Drug_1_predictions.png** - Scatter plot of Actual vs Predicted demand
2. **Drug_1_residuals.png** - Residual plot showing prediction errors
3. **Drug_1_feature_importance.png** - Top 15 most important features
4. **Drug_1_timeseries.png** - Time series comparison of actual vs predicted

## 🔧 Feature Engineering

The model uses 19 engineered features:

### Date Features (4)
- `day_of_week` - Day of week (0=Monday, 6=Sunday)
- `week_of_year` - Week number (1-52)
- `month` - Month (1-12)
- `is_weekend` - Weekend flag (1=Sat/Sun, 0=weekday)

### Lag Features (6)
- `lag_1`, `lag_2`, `lag_3` - Recent demand (1-3 days ago)
- `lag_7`, `lag_14`, `lag_28` - Weekly patterns (7, 14, 28 days ago)

### Rolling Statistics (5)
- `rolling_mean_7`, `rolling_std_7` - 7-day window
- `rolling_mean_14`, `rolling_std_14` - 14-day window
- `rolling_mean_28` - 28-day window

### Exponentially Weighted Moving Average (3)
- `ewm_7`, `ewm_14`, `ewm_28` - EWM with 7, 14, 28-day spans

### Target
- `Demand` - Daily drug demand (target variable)

## 📈 Model Configuration

```python
XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=0.9,          # 90th percentile prediction
    n_estimators=3000,
    learning_rate=0.005,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bylevel=0.9,
    gamma=0.05,
    reg_alpha=0.05,
    reg_lambda=1.0,
    random_state=42
)
```

## 📊 Evaluation Metrics

### Quantile Loss (Pinball Loss)
Primary metric for quantile regression:
```
QuantileLoss = mean(max(q*(y_true - y_pred), (q-1)*(y_true - y_pred)))
```

### Coverage
Percentage of actual values ≤ predicted values (target: 90%)

### MAE (Mean Absolute Error)
Secondary metric for sanity checking

## 🎓 Competition Notebook

A comprehensive Jupyter notebook is available at `notebooks/Drug_Demand_Prediction_Competition.ipynb` with:
- 12 sections covering the full ML pipeline
- Data exploration visualizations
- Feature engineering explanations
- Model training and evaluation
- Results analysis and conclusions

## 🔬 Key Findings

- **Best Quantile:** 0.9 (90th percentile) balances accuracy and coverage
- **Coverage:** ~73% with quantile=0.9 (conservative predictions)
- **Top Features:** Lag features (lag_1, lag_7, lag_14) are most important
- **MAE:** ~55 units average prediction error

## 📝 Results

Example output for Drug_1 (quantile=0.9):
```
Train samples: 447
Test samples: 112
MAE: 54.96
Quantile Loss: 14.92
Coverage: 73.21%
```

## 🛠️ Customization

### Change Quantile Level
Edit `main.py`:
```python
QUANTILE = 0.95  # Change from 0.9 to 0.95 for higher coverage
```

### Different Drug
Edit `main.py`:
```python
DRUG_NAME = 'Drug_2'  # Change from 'Drug_1'
```

### Adjust Features
Edit `src/data/readData.py` to modify:
- Lag periods in `LAGS` list
- Rolling window sizes
- EWM spans

## 📚 Dependencies

- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **xgboost** - Gradient boosting framework
- **scikit-learn** - Metrics and preprocessing
- **matplotlib** - Visualization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Yohan Jayasinghe**
- GitHub: [@YohanJaya](https://github.com/YohanJaya)

## 🙏 Acknowledgments

- XGBoost team for the excellent gradient boosting library
- Hospital data providers for the dataset
- Competition organizers for the problem statement

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note:** This project was developed for a hackathon competition focusing on drug demand forecasting for inventory optimization

