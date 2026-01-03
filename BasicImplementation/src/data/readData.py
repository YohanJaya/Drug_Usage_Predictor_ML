import pandas as pd
import matplotlib.pyplot as plt

# filePath = 'data/hospital_drug_demand.csv'

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

    return df


df = read_data('data/hospital_drug_demand.csv')

df = df.dropna().reset_index(drop=True)
print(df[df['Drug'] == 'Drug_10'].head(10))



