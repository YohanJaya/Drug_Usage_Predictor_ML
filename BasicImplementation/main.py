"""
Multi-Drug Usage Prediction Pipeline
Trains one XGBoost model per drug with proper data isolation
"""

import os
import pandas as pd
from src.data import split_data, read_data_single_drug, get_available_drugs
from src.models import train_model, predict, evaluate_model
from src.utils import plot_predictions, plot_residuals


def train_drug_model(drug_name, file_path):
    """
    Train a model for a single drug.
    
    Args:
        drug_name: Name of the drug (e.g., 'Drug_1')
        file_path: Path to the data CSV
    
    Returns:
        dict containing model, metrics, and predictions
    """
    # Load data for this specific drug only
    df = read_data_single_drug(file_path, drug_name=drug_name)
    df = df.dropna().reset_index(drop=True)
    
    # Check if we have enough data
    if len(df) < 50:
        return None
    
    # Split data (no leakage - each drug's data is independent)
    xTrain, yTrain, xTest, yTest = split_data(df)
    
    # Train model
    model = train_model(xTrain, yTrain)
    
    # Make predictions
    predictions = predict(model, xTest)
    
    # Evaluate
    metrics = evaluate_model(yTest, predictions)
    
    return {
        'model': model,
        'metrics': metrics,
        'predictions': predictions,
        'yTest': yTest,
        'n_train': len(xTrain),
        'n_test': len(xTest)
    }


def main():
    print("=" * 70)
    print("MULTI-DRUG USAGE PREDICTOR")
    print("Training individual XGBoost models for each drug")
    print("=" * 70)
    
    # Configuration
    data_file = 'data/hospital_drug_demand.csv'
    models_dir = 'models'
    
    # Get all available drugs from the dataset
    print("\n1. Discovering drugs in dataset...")
    drugs = get_available_drugs(data_file)
    print(f"   Found {len(drugs)} drugs: {', '.join(drugs)}")
    
    # Train models for all drugs
    print("\n2. Training models...")
    results = {}
    
    for drug in drugs:
        print(f"\n   Training {drug}...")
        result = train_drug_model(drug, data_file)
        
        if result is None:
            print(f"      ⚠ Skipped {drug} - insufficient data")
            continue
        
        results[drug] = result
        print(f"      ✓ Train samples: {result['n_train']}, Test samples: {result['n_test']}")
        print(f"      ✓ MAE: {result['metrics']['mae']:.2f}, RMSE: {result['metrics']['rmse']:.2f}, MAPE: {result['metrics']['mape']:.2f}%")
    
    # Save models
    print("\n3. Saving models...")
    import joblib
    for drug, result in results.items():
        model_path = os.path.join(models_dir, f'{drug}_model.pkl')
        joblib.dump(result['model'], model_path)
        print(f"   ✓ Saved {drug} model to {model_path}")
    
    # Create summary table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY - DAILY MODELS")
    print("=" * 70)
    
    summary_data = []
    for drug, result in results.items():
        summary_data.append({
            'Drug': drug,
            'Train_Samples': result['n_train'],
            'Test_Samples': result['n_test'],
            'MAE': f"{result['metrics']['mae']:.2f}",
            'RMSE': f"{result['metrics']['rmse']:.2f}",
            'MAPE': f"{result['metrics']['mape']:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_path = 'reports/model_evaluation_summary.csv'
    os.makedirs('reports', exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n   Summary saved to {summary_path}")

    
    # Generate visualizations for each drug
    print("\n4. Generating visualizations...")
    figures_dir = 'reports/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    for drug, result in results.items():
        # Predictions plot
        plot_predictions(
            result['yTest'], 
            result['predictions'],
            title=f'{drug} - Predictions vs Actual',
            save_path=os.path.join(figures_dir, f'{drug}_predictions.png')
        )
        
        # Residuals plot
        plot_residuals(
            result['yTest'],
            result['predictions'],
            title=f'{drug} - Residuals',
            save_path=os.path.join(figures_dir, f'{drug}_residuals.png')
        )
    
    print(f"   ✓ Saved {len(results)} sets of visualizations to {figures_dir}/")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Trained {len(results)} models | All models saved | Summary generated")
    print("=" * 70)


if __name__ == "__main__":
    main()
