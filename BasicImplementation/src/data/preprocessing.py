import numpy as np



def split_data(df):
    
    splitIndex = int(0.8 * len(df))

    train_df = df.iloc[:splitIndex]
    test_df = df.iloc[splitIndex:]

    target = 'Demand'
    # Exclude target, date columns, and other non-feature columns
    exclude_columns = {target, 'Date', 'year_week'}
    features = []

    for c in df.columns:
        if c not in exclude_columns:
            features.append(c)

    xTrain = train_df[features]
    yTrain = train_df[target]

    xTest = test_df[features]
    yTest = test_df[target]

    return xTrain, yTrain, xTest, yTest