import numpy as np
from .predict import predict

def evaluate_model(X_test, y_test, w, b):
    """
    Evaluate model performance using various metrics
    
    Args:
        X_test (ndarray (m,n)): Test feature matrix
        y_test (ndarray (m,)): Test target values
        w (ndarray (n,)): Model weights
        b (scalar): Model bias
        
    Returns:
        metrics (dict): Dictionary containing evaluation metrics
    """
    y_pred = predict(X_test, w, b)

    mae = np.mean(np.abs(y_pred - y_test))
    mse = np.mean((y_pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
    
    print("Mean Absolute Error:", round(mae, 2))
    print("Mean Squared Error:", round(mse, 2))
    print("Root Mean Squared Error:", round(rmse, 2))
    print("R-squared:", round(r2, 4))
    
    return metrics
