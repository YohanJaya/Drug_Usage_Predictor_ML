import pandas as pd
import matplotlib.pyplot as plt

# filePath = 'data/hospital_drug_demand.csv'

def get_available_drugs(filePath):
    """
    Get list of unique drugs in the dataset.
    
    Args:
        filePath: Path to the CSV file
    
    Returns:
        List of unique drug names (sorted)
    """
    df = pd.read_csv(filePath, usecols=['Drug'])
    drugs = sorted(df['Drug'].unique())
    return drugs


def read_data(filePath):

    # Read all columns to see what features we have
    df = pd.read_csv(filePath, usecols = [0,2,3,4])

    #filter only 'Dispense' Action_Type
    df = df[df['Action_Type'] == 'Dispense']

    # Drop the 'Action_Type' column as it is no longer needed
    df = df.drop(columns=['Action_Type'])
    

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    #Group by Date and Drug, summing the Demand. make sure there are no any duplicate entries
    df = (
    df.groupby(["Date", "Drug"], as_index=False)
      .agg({"Demand": "sum"})
    )
    df = df.sort_values(["Drug", "Date"]).reset_index(drop=True)



    # Extract date features
    # Day of the week (0 = Monday, 6 = Sunday)
    df["day_of_week"] = df["Date"].dt.dayofweek

    # Week of the year (1-52)
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    # Month (1-12)
    df["month"] = df["Date"].dt.month

    # Weekend flag (1 = Saturday/Sunday, 0 = weekday)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Create lag features
    LAGS = [1, 7, 14, 28]
    for lag in LAGS:
        df[f"lag_{lag}"] = (
            df
            .groupby("Drug")["Demand"]
            .shift(lag)
        )

    
    # ROLLING FEATURES 
    df["rolling_mean_7"] = (
        df.groupby("Drug")["Demand"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    df["rolling_mean_14"] = (
        df.groupby("Drug")["Demand"]
        .shift(1)
        .rolling(window=14)
        .mean()
    )

    df["rolling_std_7"] = (
        df.groupby("Drug")["Demand"]
        .shift(1)
        .rolling(window=7)
        .std()
    )

    # One-hot encode 'Drug'
    df = pd.get_dummies(df, columns=['Drug'])


    return df


def read_data_single_drug(filePath, drug_name='Drug_1'):
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
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Create lag features (no need to group by Drug since it's single drug)
    LAGS = [1, 7, 14, 28]
    for lag in LAGS:
        df[f"lag_{lag}"] = df["Demand"].shift(lag)
    
    # ROLLING FEATURES
    df["rolling_mean_7"] = df["Demand"].shift(1).rolling(window=7).mean()
    df["rolling_mean_14"] = df["Demand"].shift(1).rolling(window=14).mean()
    df["rolling_std_7"] = df["Demand"].shift(1).rolling(window=7).std()
    
    return df


def add_weekly_features(df):
    """
    Add lag and rolling features for weekly aggregated data.
    
    Args:
        df: DataFrame with 'Demand' column (weekly aggregated)
    
    Returns:
        DataFrame with weekly lag and rolling features
    """
    df['lag_1'] = df['Demand'].shift(1)    # last week
    df['lag_2'] = df['Demand'].shift(2)    # 2 weeks ago
    df['lag_4'] = df['Demand'].shift(4)    # ~1 month ago
    df['lag_8'] = df['Demand'].shift(8)    # ~2 months ago

    df['rolling_mean_4'] = df['Demand'].shift(1).rolling(4).mean()
    df['rolling_mean_8'] = df['Demand'].shift(1).rolling(8).mean()

    df['rolling_std_4'] = df['Demand'].shift(1).rolling(4).std()

    return df


def read_data_weekly_single_drug(filePath, drug_name='Drug_1'):
    """
    Read and process weekly aggregated data for a single drug.
    
    This function aggregates daily demand into weekly demand, which can help:
    - Reduce noise in daily fluctuations
    - Capture longer-term patterns
    - Improve model stability
    
    Args:
        filePath: Path to the CSV file
        drug_name: Name of the drug to filter (e.g., 'Drug_1')
    
    Returns:
        DataFrame with weekly aggregated features for the specified drug
    """
    # Read data
    df = pd.read_csv(filePath, usecols=[0, 2, 3, 4])
    df = df[df['Action_Type'] == 'Dispense']
    df = df.drop(columns=['Action_Type'])

    df['Date'] = pd.to_datetime(df['Date'])

    # Filter one drug
    df = df[df['Drug'] == drug_name]

    # Create year-week (start of each week)
    df['year_week'] = df['Date'].dt.to_period('W').apply(lambda x: x.start_time)

    # Aggregate weekly demand (sum all daily demand in each week)
    df = (
        df.groupby(['year_week'], as_index=False)
          .agg({'Demand': 'sum'})
          .sort_values('year_week')
          .reset_index(drop=True)
    )

    # Calendar features
    df['week_of_year'] = df['year_week'].dt.isocalendar().week.astype(int)
    df['month'] = df['year_week'].dt.month
    
    # Add weekly-specific lag and rolling features
    df = add_weekly_features(df)

    return df


if __name__ == "__main__":
    df = read_data('data/hospital_drug_demand.csv')
    df = df.dropna().reset_index(drop=True)
    
    print(df.head(10))

