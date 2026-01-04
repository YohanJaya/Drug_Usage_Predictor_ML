import numpy as np


def split_data(df, validation_split=True):
    """
    Split data into training, validation (optional), and testing sets.
    Uses temporal split to respect time series nature.
    
    Args:
        df: DataFrame with features and target
        validation_split: If True, creates train/val/test split (70/10/20)
                         If False, creates train/test split (80/20)
    
    Returns:
        If validation_split=True: xTrain, yTrain, xVal, yVal, xTest, yTest
        If validation_split=False: xTrain, yTrain, xTest, yTest
    """
    target = 'Demand'
    features = [c for c in df.columns if c not in ['Demand', 'Date', 'year_week']]
    
    if validation_split:
        # 70% train, 10% validation, 20% test
        train_end = int(0.7 * len(df))
        val_end = int(0.8 * len(df))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        xTrain = train_df[features]
        yTrain = train_df[target]
        
        xVal = val_df[features]
        yVal = val_df[target]
        
        xTest = test_df[features]
        yTest = test_df[target]
        
        return xTrain, yTrain, xVal, yVal, xTest, yTest
    else:
        # 80% train, 20% test (original behavior)
        splitIndex = int(0.8 * len(df))
        
        train_df = df.iloc[:splitIndex]
        test_df = df.iloc[splitIndex:]
        
        xTrain = train_df[features]
        yTrain = train_df[target]
        
        xTest = test_df[features]
        yTest = test_df[target]
        
        return xTrain, yTrain, xTest, yTest
