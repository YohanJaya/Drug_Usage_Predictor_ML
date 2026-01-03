import numpy as np
from .predict import predict
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(yTest, predictions):

    
    mse = mean_squared_error(yTest, predictions)
    mae = mean_absolute_error(yTest, predictions)
    rmse = np.sqrt(mse)
    
    print(f"Model Evaluation Metrics:")

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
   
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }
 