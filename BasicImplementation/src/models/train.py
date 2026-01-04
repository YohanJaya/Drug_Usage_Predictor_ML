import xgboost as xgb
import numpy as np


def train_model(xTrain, yTrain, quantile=0.9):
    """
    Train an XGBoost Quantile Regression model
    WITHOUT using a validation set.
    """

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,

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

        random_state=42,
        n_jobs=-1
    )

    model.fit(xTrain, yTrain, verbose=False)

    return model
