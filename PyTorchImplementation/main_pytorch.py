"""
Main script using PyTorch implementation
Linear Regression with PyTorch
"""

import numpy as np
from src.data import generate_synthetic_data, normalize_features, split_data
from src.models import train_pytorch_model, predict_pytorch
from src.utils import plot_cost_history, plot_predictions, plot_residuals, plot_feature_importance

def main():
    print("=" * 60)
    print("Drug Usage Predictor - Linear Regression with PyTorch")
    print("=" * 60)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    X, y = generate_synthetic_data(days=365, random_state=42)
    print(f"   Generated {X.shape[0]} samples with {X.shape[1]} features")
    
    # 2. Split data
    print("\n2. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8, random_state=42)
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # 3. Normalize features
    print("\n3. Normalizing features...")
    X_train_norm, mean, std = normalize_features(X_train)
    X_test_norm = (X_test - mean) / std
    print("   Features normalized")
    
    # 4. Train model with PyTorch
    print("\n4. Training model with PyTorch...")
    model, loss_history, w, b = train_pytorch_model(
        X_train_norm, 
        y_train, 
        learning_rate=0.01, 
        num_epochs=10000,
        verbose=True
    )
    print(f"   Final loss: {loss_history[-1]:.2f}")
    
    # 5. Evaluate model
    print("\n5. Evaluating PyTorch model on test set...")
    y_pred_test = predict_pytorch(model, X_test_norm)
    
    # Calculate metrics
    mae = np.mean(np.abs(y_pred_test - y_test))
    mse = np.mean((y_pred_test - y_test) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_test - y_pred_test) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print("Mean Absolute Error:", round(mae, 2))
    print("Mean Squared Error:", round(mse, 2))
    print("Root Mean Squared Error:", round(rmse, 2))
    print("R-squared:", round(r2, 4))
    
    # 6. Make sample predictions
    print("\n6. Sample predictions:")
    y_pred_sample = predict_pytorch(model, X_test_norm[:5])
    for i in range(5):
        print(f"   Actual: {y_test[i]:.0f}, Predicted: {y_pred_sample[i]:.0f}")
    
    # 7. Visualize results
    print("\n7. Generating visualizations...")
    
    # Feature names
    feature_names = ['Patient Count', 'Emergency Cases', 'Is Holiday', 'Previous Day Usage']
    
    # Plot cost history (convert to match numpy implementation scale)
    J_history_scaled = [loss * 0.5 for loss in loss_history]  # PyTorch uses MSE, numpy uses 1/2 MSE
    plot_cost_history(J_history_scaled, save_path='reports/figures/cost_history_pytorch.png')
    
    # Plot predictions vs actual
    plot_predictions(y_test, y_pred_test, save_path='reports/figures/predictions_pytorch.png')
    
    # Plot residuals
    plot_residuals(y_test, y_pred_test, save_path='reports/figures/residuals_pytorch.png')
    
    # Plot feature importance
    plot_feature_importance(w, feature_names=feature_names, save_path='reports/figures/feature_importance_pytorch.png')
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("PyTorch plots saved to reports/figures/")
    print("=" * 60)

if __name__ == "__main__":
    main()
