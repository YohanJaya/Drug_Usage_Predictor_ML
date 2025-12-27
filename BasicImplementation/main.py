"""
Main script to run the complete drug usage prediction pipeline
Linear Regression with Gradient Descent
Now using combined real (Hyper.csv) + synthetic data
"""

import numpy as np
from src.data import normalize_features, split_data
from src.data.data_combiner import combine_datasets
from src.models import train_model, predict, evaluate_model
from src.utils import plot_cost_history, plot_predictions, plot_residuals, plot_feature_importance

def main():
    print("=" * 60)
    print("Drug Usage Predictor - Linear Regression with Gradient Descent")
    print("Combined Dataset: Real Pharmacy Data + Synthetic Data")
    print("=" * 60)
    
    # 1. Load and combine datasets
    print("\n1. Loading combined dataset (Hyper.csv + synthetic data)...")
    X, y, metadata = combine_datasets(hyper_rows=50000, synthetic_days=200, random_state=42)
    print(f"   Total samples: {X.shape[0]}")
    print(f"   - Real pharmacy data: {metadata['hyper_samples']} samples")
    print(f"   - Synthetic data: {metadata['synthetic_samples']} samples")
    print(f"   Features: {X.shape[1]}")
    
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
    
    # 4. Train model
    print("\n4. Training model...")
    w, b, J_history = train_model(X_train_norm, y_train, alpha=0.01, num_iters=10000)
    print(f"   Final cost: {J_history[-1]:.2f}")
    
    # 5. Evaluate model
    print("\n5. Evaluating model on test set...")
    metrics = evaluate_model(X_test_norm, y_test, w, b)
    
    # 6. Make sample predictions
    print("\n6. Sample predictions:")
    y_pred_test = predict(X_test_norm[:5], w, b)
    for i in range(5):
        print(f"   Actual: {y_test[i]:.0f}, Predicted: {y_pred_test[i]:.0f}")
    
    # 7. Visualize results
    print("\n7. Generating visualizations...")
    
    # Feature names (unified across real + synthetic data)
    feature_names = ['Customers/Patients', 'Day of Week', 'Is Holiday', 'Promo/Emergency']
    
    # Get full predictions for plotting
    y_pred_full = predict(X_test_norm, w, b)
    
    # Plot cost history
    plot_cost_history(J_history, save_path='reports/figures/cost_history.png')
    
    # Plot predictions vs actual
    plot_predictions(y_test, y_pred_full, save_path='reports/figures/predictions.png')
    
    # Plot residuals
    plot_residuals(y_test, y_pred_full, save_path='reports/figures/residuals.png')
    
    # Plot feature importance
    plot_feature_importance(w, feature_names=feature_names, save_path='reports/figures/feature_importance.png')
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("Plots saved to reports/figures/")
    print("=" * 60)

if __name__ == "__main__":
    main()
