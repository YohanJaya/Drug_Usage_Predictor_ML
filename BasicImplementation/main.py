"""
Multi-Drug Usage Prediction Pipeline
Trains one XGBoost model per drug
"""

import os
import pandas as pd
from src.data import split_data, read_data_single_drug
from src.models import train_model, predict, evaluate_model
from src.utils import plot_predictions, plot_residuals


def train_drug_model(drug_name, file_path, quantile=0.5):
    """
    Train a quantile regression model for a single drug.
    
    Args:
        drug_name: Name of the drug
        file_path: Path to the data CSV
        quantile: Quantile to predict (default 0.5 for median)
    
    Returns:
        dict containing model, metrics, and predictions
    """
    # Load data
    df = read_data_single_drug(file_path, drug_name=drug_name)
    df = df.dropna().reset_index(drop=True)
    
    if len(df) < 50:
        return None
    
    # Split data (no validation split)
    xTrain, yTrain, xTest, yTest = split_data(df, validation_split=False)
    
    # Train model with quantile regression
    model = train_model(xTrain, yTrain, quantile=quantile)
    
    # Make predictions
    predictions = predict(model, xTest)
    
    # Evaluate
    metrics = evaluate_model(yTest, predictions, quantile=quantile)
    
    return {
        'model': model,
        'metrics': metrics,
        'predictions': predictions,
        'yTest': yTest,
        'n_train': len(xTrain),
        'n_test': len(xTest),
        'quantile': quantile
    }


def main():
    print("=" * 70)
    print("DRUG_1 USAGE PREDICTOR")
    print("XGBoost Quantile Regression for Drug_1")
    print("=" * 70)
    
    data_file = 'data/hospital_drug_demand.csv'
    models_dir = 'models'
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Focus only on Drug_1
    drug = 'Drug_1'
    quantile = 0.5  # Very high quantile for maximum coverage
    
    print(f"\nTraining quantile regression model for {drug} (quantile={quantile})...")
    result = train_drug_model(drug, data_file, quantile=quantile)
    
    if result is None:
        print(f"   Error - insufficient data")
        return
    
    print(f"   Train samples: {result['n_train']}")
    print(f"   Test samples: {result['n_test']}")
    print(f"   MAE: {result['metrics']['mae']:.2f}")
    print(f"   Quantile Loss: {result['metrics']['quantile_loss']:.4f}")
    
    print("\nSaving model...")
    import joblib
    model_path = os.path.join(models_dir, f'{drug}_model.pkl')
    joblib.dump(result['model'], model_path)
    print(f"   Model saved to: {model_path}")
    
    print("\nGenerating visualizations...")
    figures_dir = 'reports/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    plot_predictions(
        result['yTest'], 
        result['predictions'],
        save_path=os.path.join(figures_dir, f'{drug}_predictions.png')
    )
    plot_residuals(
        result['yTest'],
        result['predictions'],
        save_path=os.path.join(figures_dir, f'{drug}_residuals.png')
    )
    
    print(f"\nDone! {drug} model trained successfully.")
    print(f"   Predictions plot: {figures_dir}/{drug}_predictions.png")
    print(f"   Residuals plot: {figures_dir}/{drug}_residuals.png")


if __name__ == "__main__":
    main()
