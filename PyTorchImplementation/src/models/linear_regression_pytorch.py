"""
Linear Regression implemented using PyTorch
"""

import torch
import torch.nn as nn
import numpy as np


class LinearRegressionPyTorch(nn.Module):
    """
    Linear Regression model using PyTorch
    """
    
    def __init__(self, n_features):
        """
        Initialize the model
        
        Args:
            n_features (int): Number of input features
        """
        super(LinearRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Predictions
        """
        return self.linear(x)


def train_pytorch_model(X_train, y_train, learning_rate=0.01, num_epochs=10000, verbose=True):
    """
    Train linear regression model using PyTorch
    
    Args:
        X_train (ndarray): Training features
        y_train (ndarray): Training targets
        learning_rate (float): Learning rate
        num_epochs (int): Number of training epochs
        verbose (bool): Whether to print progress
        
    Returns:
        model: Trained PyTorch model
        loss_history (list): History of loss values
        w (ndarray): Trained weights
        b (float): Trained bias
    """
    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    
    # Initialize model
    n_features = X_train.shape[1]
    model = LinearRegressionPyTorch(n_features)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    loss_history = []
    
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record loss
        loss_history.append(loss.item())
        
        # Print progress
        if verbose and (epoch + 1) % 1000 == 0:
            print(f"   Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # Extract weights and bias
    w = model.linear.weight.data.numpy().flatten()
    b = model.linear.bias.data.numpy()[0]
    
    if verbose:
        print(f"\nTrained weights (PyTorch): {w}")
        print(f"Trained bias (PyTorch): {b}")
    
    return model, loss_history, w, b


def predict_pytorch(model, X):
    """
    Make predictions using trained PyTorch model
    
    Args:
        model: Trained PyTorch model
        X (ndarray): Input features
        
    Returns:
        ndarray: Predictions
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        predictions = model(X_tensor).numpy().flatten()
    return predictions
