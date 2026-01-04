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

