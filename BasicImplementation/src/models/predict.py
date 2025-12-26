import numpy as np

def predict(X, w, b):
    
    """
    Make predictions using trained model
    
    Args:
        X (ndarray (m,n)): Feature matrix
        w (ndarray (n,)): Model weights
        b (scalar): Model bias
        
    Returns:
        predictions (ndarray (m,)): Predicted values
    """
    return np.dot(X, w) + b
