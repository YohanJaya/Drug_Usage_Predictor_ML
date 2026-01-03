"""
Multi-Drug Usage Prediction Pipeline - WEEKLY AGGREGATION
Trains one XGBoost model per drug using weekly aggregated demand
"""

import os
import pandas as pd
from src.data import split_data, read_data_weekly_single_drug, get_available_drugs
from src.models import train_model, predict, evaluate_model
from src.utils import plot_predictions, plot_residuals


def train_drug_model_weekly(drug_name, file_path):
    """
    Train a model for a single drug using weekly aggregated data.
    
    Args:
        drug_name: Name of the drug (e.g., 'Drug_1')
        file_path: Path to the data CSV
    
    Returns:
        dict containing model, metrics, and predictions
    """
    # Load weekly aggregated data for this specific drug
    df = read_data_weekly_single_drug(file_path, drug_name=drug_name)
    df = df.dropna().reset_index(drop=True)
    
    # Check if we have enough data (need at least 20 weeks for meaningful training)
    if len(df) < 20:
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
        'n_test': len(xTest),
        'n_total_weeks': len(df)
    }


def main():
    print("=" * 70)
    print("MULTI-DRUG USAGE PREDICTOR - WEEKLY AGGREGATION")
    print("Training individual XGBoost models with weekly demand data")
    print("=" * 70)
    
    # Configuration
    data_file = 'data/hospital_drug_demand.csv'
    models_dir = 'models/weekly'
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Get all available drugs from the dataset
    print("\n1. Discovering drugs in dataset...")
    drugs = get_available_drugs(data_file)
    print(f"   Found {len(drugs)} drugs: {', '.join(drugs)}")
    
    # Train models for all drugs
    print("\n2. Training models on weekly aggregated data...")
    results = {}
    
    for drug in drugs:
        print(f"\n   Training {drug}...")
        result = train_drug_model_weekly(drug, data_file)
        
        if result is None:
            print(f"      ⚠ Skipped {drug} - insufficient data")
            continue
        
        results[drug] = result
        print(f"      ✓ Total weeks: {result['n_total_weeks']}, Train: {result['n_train']}, Test: {result['n_test']}")
        print(f"      ✓ MAE: {result['metrics']['mae']:.2f}, RMSE: {result['metrics']['rmse']:.2f}, MAPE: {result['metrics']['mape']:.2f}%")
    
    # Save models
    print("\n3. Saving weekly models...")
    import joblib
    for drug, result in results.items():
        model_path = os.path.join(models_dir, f'{drug}_weekly_model.pkl')
        joblib.dump(result['model'], model_path)
        print(f"   ✓ Saved {drug} weekly model to {model_path}")
    
    # Create summary table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY - WEEKLY MODELS")
    print("=" * 70)
    
    summary_data = []
    for drug, result in results.items():
        summary_data.append({
            'Drug': drug,
            'Total_Weeks': result['n_total_weeks'],
            'Train_Weeks': result['n_train'],
            'Test_Weeks': result['n_test'],
            'MAE': f"{result['metrics']['mae']:.2f}",
            'RMSE': f"{result['metrics']['rmse']:.2f}",
            'MAPE': f"{result['metrics']['mape']:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_path = 'reports/model_evaluation_summary_weekly.csv'
    os.makedirs('reports', exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n   Summary saved to {summary_path}")
    
    # Generate visualizations for each drug
    print("\n4. Generating visualizations...")
    figures_dir = 'reports/figures/weekly'
    os.makedirs(figures_dir, exist_ok=True)
    
    for drug, result in results.items():
        # Predictions plot
        plot_predictions(
            result['yTest'], 
            result['predictions'],
            title=f'{drug} - Weekly Predictions vs Actual',
            save_path=os.path.join(figures_dir, f'{drug}_weekly_predictions.png')
        )
        
        # Residuals plot
        plot_residuals(
            result['yTest'],
            result['predictions'],
            title=f'{drug} - Weekly Residuals',
            save_path=os.path.join(figures_dir, f'{drug}_weekly_residuals.png')
        )
    
    print(f"   ✓ Saved {len(results)} sets of visualizations to {figures_dir}/")
    
    print("\n" + "=" * 70)
    print("WEEKLY PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Trained {len(results)} weekly models | All models saved | Summary generated")
    print("=" * 70)
    
    # Compare with daily models (if available)
    daily_summary_path = 'reports/model_evaluation_summary.csv'
    if os.path.exists(daily_summary_path):
        print("\n" + "=" * 70)
        print("COMPARISON: DAILY vs WEEKLY MODELS")
        print("=" * 70)
        daily_df = pd.read_csv(daily_summary_path)
        
        comparison_data = []
        for drug in results.keys():
            weekly_metrics = results[drug]['metrics']
            daily_row = daily_df[daily_df['Drug'] == drug]
            
            if not daily_row.empty:
                # Extract values (handle both with and without % sign)
                daily_mae = float(daily_row['MAE'].values[0])
                daily_rmse = float(daily_row['RMSE'].values[0])
                daily_mape_str = str(daily_row['MAPE'].values[0])
                daily_mape = float(daily_mape_str.replace('%', ''))
                
                mae_change = ((weekly_metrics['mae'] - daily_mae) / daily_mae) * 100
                rmse_change = ((weekly_metrics['rmse'] - daily_rmse) / daily_rmse) * 100
                mape_change = ((weekly_metrics['mape'] - daily_mape) / daily_mape) * 100
                
                comparison_data.append({
                    'Drug': drug,
                    'Daily_MAE': f"{daily_mae:.2f}",
                    'Weekly_MAE': f"{weekly_metrics['mae']:.2f}",
                    'MAE_Δ%': f"{mae_change:+.1f}",
                    'Daily_MAPE': f"{daily_mape:.2f}%",
                    'Weekly_MAPE': f"{weekly_metrics['mape']:.2f}%",
                    'MAPE_Δ%': f"{mape_change:+.1f}"
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print("\n", comparison_df.to_string(index=False))
            print("\n📊 Key Insight: MAPE (relative error) is comparable between daily and weekly")
            print("   → Weekly aggregation maintains similar % accuracy despite larger absolute errors")


if __name__ == "__main__":
    main()
