"""
Combine real Hyper.csv pharmacy data with synthetic data for training
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .synthetic_data_generator import generate_synthetic_data


def load_hyper_data(csv_path=None, max_rows=None):
    """
    Load and preprocess Hyper.csv pharmacy data
    
    Args:
        csv_path (str): Path to Hyper.csv file
        max_rows (int): Maximum rows to load (None = all)
        
    Returns:
        X (ndarray): Feature matrix
        y (ndarray): Target variable (Sales)
    """
    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[2] / "data" / "Hyper.csv"
    
    # Load data
    if max_rows:
        df = pd.read_csv(csv_path, nrows=max_rows)
    else:
        df = pd.read_csv(csv_path)
    
    # Filter only open stores
    df = df[df['Open'] == 1].copy()
    
    # Convert date to datetime and extract features
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    # Create feature matrix matching synthetic data structure:
    # [Customers, DayOfWeek, IsHoliday, Promo]
    X = np.c_[
        df['Customers'].values,
        df['DayOfWeek'].values,
        df['SchoolHoliday'].values,  # IsHoliday
        df['Promo'].values
    ]
    
    # Target variable
    y = df['Sales'].values
    
    return X, y


def combine_datasets(hyper_rows=10000, synthetic_days=365, random_state=42):
    """
    Combine real Hyper.csv data with synthetic data
    
    Args:
        hyper_rows (int): Number of rows to load from Hyper.csv
        synthetic_days (int): Number of days of synthetic data to generate
        random_state (int): Random seed for reproducibility
        
    Returns:
        X_combined (ndarray): Combined feature matrix
        y_combined (ndarray): Combined target values
        metadata (dict): Information about the combination
    """
    print(f"Loading {hyper_rows} rows from Hyper.csv...")
    X_hyper, y_hyper = load_hyper_data(max_rows=hyper_rows)
    print(f"   Loaded: {X_hyper.shape[0]} samples from Hyper.csv")
    
    print(f"\nGenerating {synthetic_days} days of synthetic data...")
    X_synth, y_synth = generate_synthetic_data(days=synthetic_days, random_state=random_state)
    print(f"   Generated: {X_synth.shape[0]} synthetic samples")
    
    # Combine datasets
    X_combined = np.vstack([X_hyper, X_synth])
    y_combined = np.hstack([y_hyper, y_synth])
    
    metadata = {
        'total_samples': X_combined.shape[0],
        'hyper_samples': X_hyper.shape[0],
        'synthetic_samples': X_synth.shape[0],
        'hyper_percentage': (X_hyper.shape[0] / X_combined.shape[0]) * 100,
        'features': ['Customers/PatientCount', 'DayOfWeek', 'IsHoliday', 'Promo/Emergency'],
        'target': 'Sales/DrugUsage'
    }
    
    print(f"\n{'='*60}")
    print("COMBINED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {metadata['total_samples']}")
    print(f"  - Real data: {metadata['hyper_samples']} ({metadata['hyper_percentage']:.1f}%)")
    print(f"  - Synthetic: {metadata['synthetic_samples']} ({100-metadata['hyper_percentage']:.1f}%)")
    print(f"Features: {X_combined.shape[1]}")
    print(f"Feature names: {metadata['features']}")
    print(f"{'='*60}\n")
    
    return X_combined, y_combined, metadata


def generate_combined_data(hyper_rows=10000, synthetic_days=365, random_state=42):
    """
    Wrapper function for easy import - generates combined dataset
    
    Args:
        hyper_rows (int): Number of rows from Hyper.csv
        synthetic_days (int): Days of synthetic data
        random_state (int): Random seed
        
    Returns:
        X (ndarray): Feature matrix
        y (ndarray): Target variable
    """
    X, y, _ = combine_datasets(hyper_rows, synthetic_days, random_state)
    return X, y


if __name__ == "__main__":
    # Test the combiner
    X, y, meta = combine_datasets(hyper_rows=5000, synthetic_days=200)
    print(f"\nTest successful!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Sample X[0]: {X[0]}")
    print(f"Sample y[0]: {y[0]}")
