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

    #testing
    print(df.head(10))
    print(df.shape)
    print(df.duplicated(subset=["Date", "Drug"]).sum())


    return df



read_data('data/hospital_drug_demand.csv')