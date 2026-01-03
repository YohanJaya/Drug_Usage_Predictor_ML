import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(yTest, predictions):
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        yTest: Actual values
        predictions: Predicted values
    
    Returns:
        dict with mse, mae, rmse, mape
    """
    mse = mean_squared_error(yTest, predictions)
    mae = mean_absolute_error(yTest, predictions)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (Mean Absolute Percentage Error) - relative error
    # Only include samples where actual value > threshold to avoid division issues
    threshold = 10  # Ignore very small actual values
    mask = np.abs(yTest) > threshold
    
    if mask.sum() > 0:
        mape = np.mean(np.abs((yTest[mask] - predictions[mask]) / yTest[mask])) * 100
    else:
        mape = np.nan
    
    print(f"Model Evaluation Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    if not np.isnan(mape):
        print(f"MAPE: {mape:.2f}%")
    else:
        print(f"MAPE: N/A (insufficient non-zero samples)")
   
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }


if __name__ == "__main__":
    print("Evaluate module loaded. Use evaluate_model(yTest, predictions) to evaluate.")

