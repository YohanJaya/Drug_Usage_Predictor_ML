"""
IMPROVED Multi-Drug Usage Prediction Pipeline
Implements multiple strategies to reduce MAPE and improve accuracy
"""

import os
import pandas as pd
from src.data import split_data, read_data_single_drug, get_available_drugs
from src.data.feature_engineering import add_advanced_features, remove_outliers
from src.models.train_optimized import train_model_optimized
from src.models import predict, evaluate_model
from src.utils import plot_predictions, plot_residuals


def train_drug_model_improved(drug_name, file_path, remove_outliers_flag=True, tune_params=False):
    """
    Train an improved model for a single drug with advanced features.
    
    Args:
        drug_name: Name of the drug (e.g., 'Drug_1')
        file_path: Path to the data CSV
        remove_outliers_flag: Whether to remove outliers
        tune_params: Whether to tune hyperparameters (slower but better)
    
    Returns:
        dict containing model, metrics, and predictions
    """
    # Load data for this specific drug only
    df = read_data_single_drug(file_path, drug_name=drug_name)
    
    # Add advanced features
    df = add_advanced_features(df)
    
    # Remove outliers if requested
    if remove_outliers_flag:
        df = remove_outliers(df, column='Demand', method='iqr', threshold=2.5)
    
    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)
    
    # Check if we have enough data
    if len(df) < 50:
        return None
    
    # Split data (no leakage - each drug's data is independent)
    xTrain, yTrain, xTest, yTest = split_data(df)
    
    # Train optimized model
    model = train_model_optimized(xTrain, yTrain, tune_hyperparameters=tune_params)
    
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
        'n_test': len(xTest),
        'n_features': xTrain.shape[1]
    }


def main():
    print("=" * 70)
    print("IMPROVED MULTI-DRUG USAGE PREDICTOR")
    print("Advanced features + Optimized hyperparameters")
    print("=" * 70)
    
    # Configuration
    data_file = 'data/hospital_drug_demand.csv'
    models_dir = 'models/improved'
    remove_outliers_flag = True
    tune_hyperparameters = False  # Set to True for better results (slower)
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Get all available drugs from the dataset
    print("\n1. Discovering drugs in dataset...")
    drugs = get_available_drugs(data_file)
    print(f"   Found {len(drugs)} drugs: {', '.join(drugs)}")
    
    print(f"\n   Configuration:")
    print(f"   - Remove outliers: {remove_outliers_flag}")
    print(f"   - Hyperparameter tuning: {tune_hyperparameters}")
    print(f"   - Advanced features: ✓ Enabled")
    
    # Train models for all drugs
    print("\n2. Training improved models...")
    results = {}
    
    for drug in drugs:
        print(f"\n   Training {drug}...")
        result = train_drug_model_improved(
            drug, data_file, 
            remove_outliers_flag=remove_outliers_flag,
            tune_params=tune_hyperparameters
        )
        
        if result is None:
            print(f"      ⚠ Skipped {drug} - insufficient data")
            continue
        
        results[drug] = result
        print(f"      ✓ Train: {result['n_train']}, Test: {result['n_test']}, Features: {result['n_features']}")
        print(f"      ✓ MAE: {result['metrics']['mae']:.2f}, RMSE: {result['metrics']['rmse']:.2f}, MAPE: {result['metrics']['mape']:.2f}%")
    
    # Save models
    print("\n3. Saving improved models...")
    import joblib
    for drug, result in results.items():
        model_path = os.path.join(models_dir, f'{drug}_improved_model.pkl')
        joblib.dump(result['model'], model_path)
        print(f"   ✓ Saved {drug} model to {model_path}")
    
    # Create summary table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY - IMPROVED DAILY MODELS")
    print("=" * 70)
    
    summary_data = []
    for drug, result in results.items():
        summary_data.append({
            'Drug': drug,
            'Features': result['n_features'],
            'Train_Samples': result['n_train'],
            'Test_Samples': result['n_test'],
            'MAE': f"{result['metrics']['mae']:.2f}",
            'RMSE': f"{result['metrics']['rmse']:.2f}",
            'MAPE': f"{result['metrics']['mape']:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))
    
    # Calculate average improvement
    avg_mape = summary_df['MAPE'].str.replace('%', '').astype(float).mean()
    print(f"\n   📊 Average MAPE: {avg_mape:.2f}%")
    
    # Save summary to CSV
    summary_path = 'reports/model_evaluation_summary_improved.csv'
    os.makedirs('reports', exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n   Summary saved to {summary_path}")
    
    # Generate visualizations for each drug
    print("\n4. Generating visualizations...")
    figures_dir = 'reports/figures/improved'
    os.makedirs(figures_dir, exist_ok=True)
    
    for drug, result in results.items():
        # Predictions plot
        plot_predictions(
            result['yTest'], 
            result['predictions'],
            title=f'{drug} - Improved Predictions vs Actual',
            save_path=os.path.join(figures_dir, f'{drug}_improved_predictions.png')
        )
        
        # Residuals plot
        plot_residuals(
            result['yTest'],
            result['predictions'],
            title=f'{drug} - Improved Residuals',
            save_path=os.path.join(figures_dir, f'{drug}_improved_residuals.png')
        )
    
    print(f"   ✓ Saved {len(results)} sets of visualizations to {figures_dir}/")
    
    # Compare with baseline models
    baseline_path = 'reports/model_evaluation_summary.csv'
    if os.path.exists(baseline_path):
        print("\n" + "=" * 70)
        print("IMPROVEMENT vs BASELINE MODELS")
        print("=" * 70)
        
        baseline_df = pd.read_csv(baseline_path)
        comparison_data = []
        
        for drug in results.keys():
            improved_metrics = results[drug]['metrics']
            baseline_row = baseline_df[baseline_df['Drug'] == drug]
            
            if not baseline_row.empty:
                baseline_mape_str = str(baseline_row['MAPE'].values[0])
                baseline_mape = float(baseline_mape_str.replace('%', ''))
                
                mape_improvement = baseline_mape - improved_metrics['mape']
                mape_improvement_pct = (mape_improvement / baseline_mape) * 100
                
                comparison_data.append({
                    'Drug': drug,
                    'Baseline_MAPE': f"{baseline_mape:.2f}%",
                    'Improved_MAPE': f"{improved_metrics['mape']:.2f}%",
                    'Reduction': f"{mape_improvement:.2f}%",
                    'Improvement_%': f"{mape_improvement_pct:.1f}%"
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print("\n", comparison_df.to_string(index=False))
            
            avg_baseline = baseline_df['MAPE'].str.replace('%', '').astype(float).mean()
            print(f"\n   📈 Baseline Average MAPE: {avg_baseline:.2f}%")
            print(f"   🎯 Improved Average MAPE: {avg_mape:.2f}%")
            print(f"   ✨ Overall Improvement: {avg_baseline - avg_mape:.2f}% ({((avg_baseline - avg_mape) / avg_baseline * 100):.1f}%)")
    
    print("\n" + "=" * 70)
    print("IMPROVED PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Trained {len(results)} models with advanced features")
    print("=" * 70)


if __name__ == "__main__":
    main()
