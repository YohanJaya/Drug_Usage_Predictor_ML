import numpy as np



def split_data():
    
    splitIndex = int(0.8 * len(df))

    train_df = df.iloc[:splitIndex]
    test_df = df.iloc[splitIndex:]

    target = 'Demand'
    features = []

    for c in df.columns:

        if c != targer and c != 'Date':
            features.append(c)

    xTrain = train_df[features]
    yTrain = train_df[target]

    xTest = test_df[features]
    yTest = test_df[target]

    return xTrain, yTrain, xTest, yTest