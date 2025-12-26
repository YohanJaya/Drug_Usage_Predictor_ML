"""
Visualization functions for model analysis
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_cost_history(J_history, save_path=None):
    """
    Plot the cost function history during training
    
    Args:
        J_history (list): History of cost values
        save_path (str): Path to save the figure (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(J_history, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost J', fontsize=12)
    plt.title('Cost Function History - Gradient Descent', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Cost history plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_predictions(y_test, y_pred, save_path=None):
    """
    Plot actual vs predicted values
    
    Args:
        y_test (ndarray): Actual values
        y_pred (ndarray): Predicted values
        save_path (str): Path to save the figure (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Drug Usage', fontsize=12)
    plt.ylabel('Predicted Drug Usage', fontsize=12)
    plt.title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Predictions plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_residuals(y_test, y_pred, save_path=None):
    """
    Plot residuals (errors) distribution
    
    Args:
        y_test (ndarray): Actual values
        y_pred (ndarray): Predicted values
        save_path (str): Path to save the figure (optional)
    """
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residual Plot', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Residuals plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(w, feature_names=None, save_path=None):
    """
    Plot feature importance based on trained weights
    
    Args:
        w (ndarray): Model weights
        feature_names (list): Names of features (optional)
        save_path (str): Path to save the figure (optional)
    """
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(len(w))]
    
    # Get absolute values for importance
    importance = np.abs(w)
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(w)))
    bars = plt.bar(range(len(w)), importance[indices], color=colors, edgecolor='black', linewidth=1.2)
    
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Absolute Weight (Importance)', fontsize=12)
    plt.title('Feature Importance', fontsize=14, fontweight='bold')
    plt.xticks(range(len(w)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Feature importance plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()
