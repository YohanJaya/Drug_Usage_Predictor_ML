import xgboost as xgb

import numpy as np
from preprocessing import split_data


def train_model(xTrain, yTrain):
    """
    Train an XGBoost regression model.

    Parameters:
    xTrain (pd.DataFrame): Training features.
    yTrain (pd.Series): Training target variable.

    Returns:
    model: Trained XGBoost model.
    """
    # Create XGBoost regressor
    model = xgb.XGBRegressor(
        objective='reg:squarederror',  # regression
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Fit the model
    model.fit(xTrain, yTrain)

    return model