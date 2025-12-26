import numpy as np
from .linear_regression import gradientDescent

def train_model(X_train, y_train, alpha=0.01, num_iters=10000):
    """
    Train linear regression model using gradient descent
    
    Args:
        X_train (ndarray (m,n)): Training feature matrix
        y_train (ndarray (m,)): Training target values
        alpha (float): Learning rate
        num_iters (int): Number of iterations
        
    Returns:
        w (ndarray (n,)): Trained weights
        b (scalar): Trained bias
        J_history (list): Cost history
    """
    w_init = np.zeros(X_train.shape[1])
    b_init = 0

    w, b, J_history = gradientDescent(
        X_train,
        y_train,
        w_init,
        b_init,
        alpha=alpha,
        num_iters=num_iters
    )

    print("Trained weights:", w)
    print("Trained bias:", b)
    
    return w, b, J_history
