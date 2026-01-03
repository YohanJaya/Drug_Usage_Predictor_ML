import numpy as np
import xgboost as xgb


def train_model(xTrain, yTrain):
    """
    Train an XGBoost regression model with log-transformed target.
    
    The target variable is log-transformed to handle skewed demand distribution,
    reduce spike dominance, and improve learning smoothness.

    Parameters:
    xTrain (pd.DataFrame): Training features.
    yTrain (pd.Series): Training target variable.

    Returns:
    model: Trained XGBoost model (trained on log-transformed target).
    """
    # Log-transform the target to handle skewed distribution
    # Using log1p (log(1 + x)) to handle zero values safely
    yTrain_log = np.log1p(yTrain)
    
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

    # Fit the model on log-transformed target
    model.fit(xTrain, yTrain_log)

    return model