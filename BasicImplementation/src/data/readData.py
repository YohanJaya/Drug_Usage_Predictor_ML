
import pandas as pd
import numpy as np


def read_data_single_drug(filePath, drug_name='Drug_1'):
    """
    Read and process data for a single specific drug.
    Based on the competition notebook approach.
    
    Args:
        filePath: Path to the CSV file
        drug_name: Name of the drug to filter (e.g., 'Drug_1')
    
    Returns:
        DataFrame with features for the specified drug only
    """
    # Read CSV
    df = pd.read_csv(filePath)
    
    # Filter for Dispense action and specific drug
    df = df[(df['Action_Type'] == 'Dispense') & (df['Drug'] == drug_name)]
    
    # Drop unnecessary columns
    df = df.drop(['Hospital', 'Drug', 'Action_Type', 'Stock_Level', 'Restock_Amount'], axis=1)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Drop Year_Month if it exists (from visualization)
    if 'Year_Month' in df.columns:
        df = df.drop('Year_Month', axis=1)
    
    # Date features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Lag features
    LAGS = [1, 2, 3, 7, 14, 28]
    for lag in LAGS:
        df[f"lag_{lag}"] = df["Demand"].shift(lag)
    
    # Rolling statistics (7-day window)
    df["rolling_mean_7"] = df["Demand"].shift(1).rolling(window=7).mean()
    df["rolling_std_7"] = df["Demand"].shift(1).rolling(window=7).std()
    
    # Rolling statistics (14-day window)
    df["rolling_mean_14"] = df["Demand"].shift(1).rolling(window=14).mean()
    df["rolling_std_14"] = df["Demand"].shift(1).rolling(window=14).std()
    
    # Rolling statistics (28-day window)
    df["rolling_mean_28"] = df["Demand"].shift(1).rolling(window=28).mean()
    
    # Exponentially weighted moving average
    df["ewm_7"] = df["Demand"].shift(1).ewm(span=7).mean()
    df["ewm_14"] = df["Demand"].shift(1).ewm(span=14).mean()
    df["ewm_28"] = df["Demand"].shift(1).ewm(span=28).mean()
    
    # Drop rows with NaN (from lag features)
    df = df.dropna().reset_index(drop=True)
    
    return df


def split_data(df, test_size=0.2, validation_split=False):
    """
    Temporal train-test split (no validation set for competition).
    
    Args:
        df: DataFrame with features
        test_size: Fraction of data for test set
        validation_split: Ignored for compatibility (always False)
    
    Returns:
        xTrain, yTrain, xTest, yTest
    """
    # Separate features and target
    X = df.drop(columns=['Demand', 'Date'])
    y = df['Demand']
    
    # Temporal split
    split_idx = int(len(df) * (1 - test_size))
    
    xTrain = X.iloc[:split_idx]
    yTrain = y.iloc[:split_idx]
    xTest = X.iloc[split_idx:]
    yTest = y.iloc[split_idx:]
    
    return xTrain, yTrain, xTest, yTest

