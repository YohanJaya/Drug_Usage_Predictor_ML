import numpy as np


def calCost(xTrain, yTrain, w, b):

        """
        Docstring for calCost
        
        Args:
            xTrain (ndarray (m,n)): Data, m examples with n features
            yTrain (ndarray (m,)): target values
            w (ndarray (n,)): model parameters
            b (scalar)    : model parameter

        Returns:
            totalCost(int)
        """

        m = xTrain.shape[0]
        n = xTrain.shape[1]
        totalCost = 0

        for i in range(m):
            f_wb = np.dot(w, xTrain[i]) + b
            totalCost += (f_wb - yTrain[i]) ** 2

        totalCost = totalCost / (2 * m)
        return totalCost

def calGradient(xTrain, yTrain, w, b, lambda_= 1):

    """
    Docstring for calGradient
    
    Args:
        xTrain (ndarray (m,n)): Data, m examples with n features
        yTrain (ndarray (m,)): target values
        w (ndarray (n,)): model parameters
        b (scalar)    : model parameter
        lambda_ (float): regularization parameter

    Returns:
        dj_dw (ndarray (n,)): gradient w.r.t. w
        dj_db (scalar)      : gradient w.r.t. b
    """

    m = xTrain.shape[0]
    n = xTrain.shape[1]

    dw = np.zeros(m)
    db = 0

    for i in range(m):
        f_wb = np.dot(w, xTrain[i]) + b

        for j in range(n):
            dw[j] += (f_wb - yTrain[i]) * xTrain[i][j] 

        db += (f_wb - yTrain[i])

    dj_dw = dw / m
    dj_db = db / m

    for j in range(n):
        dj_dw[j] += (lambda_ / m) * w[j]
        
    return dj_dw, dj_db

def gradientDescent(xTrain, yTrain, w_in, b_in, alpha, num_iters):

    """
    Docstring for gradientDescent
    
    Args:
        xTrain (ndarray (m,n)): Data, m examples with n features
        yTrain (ndarray (m,)): target values
        w_in (ndarray (n,)): initial model parameters
        b_in (scalar)    : initial model parameter
        alpha (float)   : learning rate
        num_iters (int) : number of iterations to run gradient descent

    Returns:
        w (ndarray (n,)): updated model parameters
        b (scalar)      : updated model parameter
        J_history (list): history of cost function values
    """

    J_history = []

    w = w_in.copy()
    b = b_in

    for i in range(num_iters):

        dj_dw, dj_db = calGradient(xTrain, yTrain, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(calCost(xTrain, yTrain, w, b))

    return w, b, J_history