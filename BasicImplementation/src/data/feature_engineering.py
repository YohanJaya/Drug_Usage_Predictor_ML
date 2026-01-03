"""
Advanced feature engineering module to improve model accuracy
"""

import pandas as pd
import numpy as np


def add_advanced_features(df):
    """
    Add advanced features to improve MAPE and model accuracy.
    
    Strategies implemented:
    1. Interaction features (lag × calendar features)
    2. Exponential weighted moving averages (emphasize recent data)
    3. Momentum/trend indicators
    4. Cyclic encoding for time features
    5. Lag differences (rate of change)
    
    Args:
        df: DataFrame with Demand and basic features
    
    Returns:
        DataFrame with additional advanced features
    """
    
    # 1. EXPONENTIAL WEIGHTED MOVING AVERAGES (more weight to recent data)
    df['ewm_7'] = df['Demand'].shift(1).ewm(span=7, adjust=False).mean()
    df['ewm_14'] = df['Demand'].shift(1).ewm(span=14, adjust=False).mean()
    df['ewm_28'] = df['Demand'].shift(1).ewm(span=28, adjust=False).mean()
    
    # 2. LAG DIFFERENCES (rate of change / momentum)
    df['lag_diff_1'] = df['Demand'].shift(1) - df['Demand'].shift(2)
    df['lag_diff_7'] = df['Demand'].shift(7) - df['Demand'].shift(14)
    
    # 3. RATIO FEATURES (relative comparisons)
    df['lag_7_to_lag_14_ratio'] = df['lag_7'] / (df['lag_14'] + 1)  # +1 to avoid division by zero
    df['recent_to_avg_ratio'] = df['lag_1'] / (df['rolling_mean_7'] + 1)
    
    # 4. VOLATILITY FEATURES
    df['demand_volatility'] = df['Demand'].shift(1).rolling(window=7).std() / (df['Demand'].shift(1).rolling(window=7).mean() + 1)
    
    # 5. CYCLIC ENCODING for periodic features (better than raw numbers)
    # Day of week - convert to sine/cosine to capture cyclical nature
    if 'day_of_week' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Month - convert to sine/cosine
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 6. INTERACTION FEATURES (lag × calendar)
    if 'is_weekend' in df.columns:
        df['lag_7_weekend'] = df['lag_7'] * df['is_weekend']
        df['rolling_mean_7_weekend'] = df['rolling_mean_7'] * df['is_weekend']
    
    # 7. TREND FEATURES
    # Simple linear trend over last 7 days
    for i in range(1, 8):
        if f'lag_{i}' not in df.columns:
            df[f'lag_{i}'] = df['Demand'].shift(i)
    
    # Calculate trend slope
    df['trend_7'] = (df['lag_1'] - df['lag_7']) / 7
    
    # 8. PERCENTILE FEATURES (where does current demand stand?)
    df['demand_percentile_7'] = df['Demand'].shift(1).rolling(window=7).apply(
        lambda x: (x.iloc[-1] / x.quantile(0.5)) if x.quantile(0.5) > 0 else 1
    )
    
    return df


def add_advanced_weekly_features(df):
    """
    Add advanced features for weekly aggregated data.
    
    Args:
        df: DataFrame with weekly Demand data
    
    Returns:
        DataFrame with additional advanced features
    """
    
    # 1. EXPONENTIAL WEIGHTED MOVING AVERAGES
    df['ewm_4'] = df['Demand'].shift(1).ewm(span=4, adjust=False).mean()
    df['ewm_8'] = df['Demand'].shift(1).ewm(span=8, adjust=False).mean()
    df['ewm_12'] = df['Demand'].shift(1).ewm(span=12, adjust=False).mean()
    
    # 2. LAG DIFFERENCES (weekly momentum)
    df['lag_diff_1'] = df['Demand'].shift(1) - df['Demand'].shift(2)
    df['lag_diff_4'] = df['Demand'].shift(4) - df['Demand'].shift(8)
    
    # 3. RATIO FEATURES
    df['lag_1_to_lag_4_ratio'] = df['lag_1'] / (df['lag_4'] + 1)
    df['lag_2_to_lag_8_ratio'] = df['lag_2'] / (df['lag_8'] + 1)
    
    # 4. VOLATILITY
    df['weekly_volatility'] = df['Demand'].shift(1).rolling(window=4).std() / (df['Demand'].shift(1).rolling(window=4).mean() + 1)
    
    # 5. CYCLIC ENCODING for week_of_year
    if 'week_of_year' in df.columns:
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # 6. CYCLIC ENCODING for month
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 7. TREND
    df['trend_4'] = (df['lag_1'] - df['lag_4']) / 4
    df['trend_8'] = (df['lag_1'] - df['lag_8']) / 8
    
    # 8. ACCELERATION (trend of trend)
    df['acceleration'] = df['trend_4'] - df['lag_diff_1']
    
    return df


def remove_outliers(df, column='Demand', method='iqr', threshold=3):
    """
    Remove outliers from demand data which can hurt MAPE.
    
    Args:
        df: DataFrame
        column: Column to check for outliers
        method: 'iqr' or 'zscore'
        threshold: IQR multiplier or z-score threshold
    
    Returns:
        DataFrame with outliers removed
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    else:  # zscore
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        mask = z_scores < threshold
    
    outliers_removed = len(df) - mask.sum()
    if outliers_removed > 0:
        print(f"   → Removed {outliers_removed} outliers from {len(df)} samples")
    
    return df[mask].copy()
