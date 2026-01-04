"""
Simple Drug Demand Prediction Pipeline
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data import split_data, read_data_single_drug
from src.models import train_model, predict, evaluate_model

# Configuration
DRUG_NAME = 'Drug_1'
QUANTILE = 0.9
DATA_FILE = 'data/hospital_drug_demand.csv'
FIGURES_DIR = 'reports/figures'

os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    print("="*70)
    print(f"Training {DRUG_NAME} with Quantile={QUANTILE}")
    print("="*70)
    
    # Load and prepare data
    df = read_data_single_drug(DATA_FILE, drug_name=DRUG_NAME)
    xTrain, yTrain, xTest, yTest = split_data(df)
    
    print(f"Train samples: {len(xTrain)}, Test samples: {len(xTest)}")
    
    # Train model
    model = train_model(xTrain, yTrain, quantile=QUANTILE)
    
    # Predict
    yPred = predict(model, xTest)
    
    # Evaluate
    metrics = evaluate_model(yTest, yPred, quantile=QUANTILE)
    print(f"MAE: {metrics['mae']:.2f}, Quantile Loss: {metrics['quantile_loss']:.4f}")
    
    # Save plots
    print("\nSaving plots...")
    
    # 1. Predictions vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(yTest, yPred, alpha=0.5, color='steelblue', edgecolors='black')
    plt.plot([yTest.min(), yTest.max()], [yTest.min(), yTest.max()], 'r--', lw=2)
    plt.xlabel('Actual Demand')
    plt.ylabel('Predicted Demand')
    plt.title(f'{DRUG_NAME} - Predictions vs Actual (MAE={metrics["mae"]:.2f})')
    plt.grid(alpha=0.3)
    plt.savefig(f'{FIGURES_DIR}/{DRUG_NAME}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {DRUG_NAME}_predictions.png")
    
    # 2. Residuals
    residuals = yTest - yPred
    plt.figure(figsize=(10, 6))
    plt.scatter(yPred, residuals, alpha=0.5, color='coral', edgecolors='black')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Demand')
    plt.ylabel('Residuals')
    plt.title(f'{DRUG_NAME} - Residual Plot')
    plt.grid(alpha=0.3)
    plt.savefig(f'{FIGURES_DIR}/{DRUG_NAME}_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {DRUG_NAME}_residuals.png")
    
    # 3. Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': xTrain.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_importance)), feature_importance['Importance'], color='steelblue')
    plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
    plt.xlabel('Importance Score')
    plt.title(f'{DRUG_NAME} - Top 15 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/{DRUG_NAME}_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {DRUG_NAME}_feature_importance.png")
    
    # 4. Time Series
    plt.figure(figsize=(14, 6))
    test_indices = range(len(yTest))
    plt.plot(test_indices, yTest.values, 'o-', label='Actual', color='blue', alpha=0.6, markersize=4)
    plt.plot(test_indices, yPred, 's-', label=f'Predicted (q={QUANTILE})', color='red', alpha=0.6, markersize=3)
    plt.fill_between(test_indices, yTest.values, yPred, alpha=0.2, color='gray')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Demand')
    plt.title(f'{DRUG_NAME} - Time Series Predictions')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'{FIGURES_DIR}/{DRUG_NAME}_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {DRUG_NAME}_timeseries.png")
    
    print("\n" + "="*70)
    print(f"Done! All plots saved to {FIGURES_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()
