import numpy as np

def normalize_features(X):
    """
    Normalize features using standardization
    
    Args:
        X (ndarray (m,n)): Feature matrix
        
    Returns:
        X_norm (ndarray (m,n)): Normalized feature matrix
        mean (ndarray (n,)): Mean of each feature
        std (ndarray (n,)): Standard deviation of each feature
    """
    mean  = np.mean(X, axis=0)
    std   = np.std(X, axis=0)

    X_norm = (X - mean) / std
    return X_norm, mean, std

def split_data(X, y, train_ratio=0.8, random_state=None):
    """
    Split data into training and testing sets
    
    Args:
        X (ndarray (m,n)): Feature matrix
        y (ndarray (m,)): Target variable
        train_ratio (float): Ratio of training data
        random_state (int): Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    m = X.shape[0]
    indices = np.random.permutation(m)
    train_size = int(m * train_ratio)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test
