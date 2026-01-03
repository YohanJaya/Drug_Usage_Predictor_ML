"""
Prediction module for trained models with inverse log transformation
"""

import numpy as np


def predict(model, xTest):
    """
    Make predictions using a trained model and inverse log-transform.
    
    Since the model was trained on log-transformed targets, we need to
    apply the inverse transformation (expm1) to get predictions in the
    original scale.
    
    Args:
        model: Trained XGBoost model (trained on log-transformed target)
        xTest: Test features
    
    Returns:
        numpy array of predictions in original scale
    """
    # Model predicts in log space
    predictions_log = model.predict(xTest)
    
    # Inverse transform: expm1 is inverse of log1p
    predictions = np.expm1(predictions_log)
    
    return predictions
