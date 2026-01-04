import pandas as pd
import matplotlib.pyplot as plt

# filePath = 'data/hospital_drug_demand.csv'



# def read_data(filePath):

#     # Read all columns to see what features we have
#     df = pd.read_csv(filePath, usecols = [0,2,3,4])

#     #filter only 'Dispense' Action_Type
#     df = df[df['Action_Type'] == 'Dispense']

#     # Drop the 'Action_Type' column as it is no longer needed
#     df = df.drop(columns=['Action_Type'])
    

#     # Convert Date to datetime
#     df['Date'] = pd.to_datetime(df['Date'])

#     #Group by Date and Drug, summing the Demand. make sure there are no any duplicate entries
#     df = (
#     df.groupby(["Date", "Drug"], as_index=False)
#       .agg({"Demand": "sum"})
#     )
#     df = df.sort_values(["Drug", "Date"]).reset_index(drop=True)



#     # Extract date features
#     # Day of the week (0 = Monday, 6 = Sunday)
#     df["day_of_week"] = df["Date"].dt.dayofweek

#     # Week of the year (1-52)
#     df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
#     # Month (1-12)
#     df["month"] = df["Date"].dt.month

#     # Weekend flag (1 = Saturday/Sunday, 0 = weekday)
#     df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

#     # Create lag features
#     LAGS = [1, 7, 14, 28]
#     for lag in LAGS:
#         df[f"lag_{lag}"] = (
#             df
#             .groupby("Drug")["Demand"]
#             .shift(lag)
#         )

    
#     # ROLLING FEATURES 
#     df["rolling_mean_7"] = (
#         df.groupby("Drug")["Demand"]
#         .shift(1)
#         .rolling(window=7)
#         .mean()
#     )

#     df["rolling_mean_14"] = (
#         df.groupby("Drug")["Demand"]
#         .shift(1)
#         .rolling(window=14)
#         .mean()
#     )

#     df["rolling_std_7"] = (
#         df.groupby("Drug")["Demand"]
#         .shift(1)
#         .rolling(window=7)
#         .std()
#     )

#     # One-hot encode 'Drug'
#     df = pd.get_dummies(df, columns=['Drug'])


#     return df

def read_data(filePath):

    df = pd.read_csv(filePath, usecols=[0,2,3,4])

    df = df[(df['Action_Type'] == 'Dispense') & (df['Drug'] == 'Drug_1')]
    df = df.drop(['Hospital','Drug','Action_Type','Stock_Level','Restock_Amount'],axis = 1)
    df['Date'] = pd.to_datetime(df['Date'])

    # Aggregate daily demand
    LAGS = [1,7,14,28]
    for lag in LAGS:
        df[f'demand_lag_{lag}'] = df['Demand'].shift(lag)


    df = df.fillna(0)


  
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

   

    

    return df



def read_data_single_drug(filePath, drug_name='Drug_3'):
    """
    Read and process data for a single specific drug.
    
    Args:
        filePath: Path to the CSV file
        drug_name: Name of the drug to filter (e.g., 'Drug_1')
    
    Returns:
        DataFrame with features for the specified drug only
    """
    # Read all columns
    df = pd.read_csv(filePath, usecols=[0, 2, 3, 4])

    # Filter only 'Dispense' Action_Type
    df = df[df['Action_Type'] == 'Dispense']
    
    # Drop the 'Action_Type' column
    df = df.drop(columns=['Action_Type'])
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Group by Date and Drug, summing the Demand
    df = (
        df.groupby(["Date", "Drug"], as_index=False)
        .agg({"Demand": "sum"})
    )
    
    # Filter for specific drug BEFORE creating features
    df = df[df['Drug'] == drug_name].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Drop Drug column since we're only working with one drug
    df = df.drop(columns=['Drug'])
    
    # Extract date features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    #df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Create lag features (no need to group by Drug since it's single drug)
    LAGS = [1, 2, 3, 7, 14, 28]
    for lag in LAGS:
        df[f"lag_{lag}"] = df["Demand"].shift(lag)
    
    # ROLLING FEATURES
    df["rolling_mean_7"] = df["Demand"].shift(1).rolling(window=7).mean()
    df["rolling_mean_14"] = df["Demand"].shift(1).rolling(window=14).mean()
    df["rolling_mean_28"] = df["Demand"].shift(1).rolling(window=28).mean()
    df["rolling_std_7"] = df["Demand"].shift(1).rolling(window=7).std()

    # Exponential moving averages (no groupby needed - single drug)
    df["ewm_7"] = df["Demand"].shift(1).ewm(span=7, adjust=False).mean()
    df["ewm_14"] = df["Demand"].shift(1).ewm(span=14, adjust=False).mean()
    df["ewm_28"] = df["Demand"].shift(1).ewm(span=28, adjust=False).mean()

    # Create lag_2 and lag_3 if not already created
    if "lag_2" not in df.columns:
        df["lag_2"] = df["Demand"].shift(2)
    if "lag_3" not in df.columns:
        df["lag_3"] = df["Demand"].shift(3)

    # Momentum (first derivative)
    df["momentum_1d"] = df["lag_1"] - df["lag_2"]
    df["momentum_7d"] = df["lag_7"] - df["lag_14"]

    # Trend strength (slope approximation)
    df["trend_7d"] = (df["lag_1"] - df["lag_7"]) / 7

    # Short vs long trend imbalance
    df["trend_gap"] = df["rolling_mean_7"] - df["rolling_mean_28"]

    # Relative position (scale-free)
    df["rel_to_avg"] = df["lag_1"] / (df["rolling_mean_14"] + 1)

    # Volatility regime
    df["volatility_ratio"] = df["rolling_std_7"] / (df["rolling_mean_7"] + 1)

    df["demand_p90"] = df["rolling_mean_28"].rolling(60).quantile(0.9)
    df["high_demand_regime"] = (
        df["rolling_mean_7"] > df["demand_p90"]
    ).astype(int)

    df["deviation"] = df["lag_1"] - df["rolling_mean_28"]
    df["abs_deviation"] = df["deviation"].abs()

    df["lag_1_sq"] = df["lag_1"] ** 2




    return df


