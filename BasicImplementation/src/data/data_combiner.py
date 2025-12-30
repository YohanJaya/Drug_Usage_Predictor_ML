import numpy as np
import pandas as pd
from pathlib import Path


csv_path = Path(__file__).parent / "Hyper.csv"

# Load the CSV
df = pd.read_csv(csv_path, usecols = [1,3,4,7], dtype={"StateHoliday" : str}, nrows=10000)

#rename the columns
df = df.rename(columns={
    "Sales": "drug_usage",
    "Customers": "PatientCount",
    "StateHoliday": "isHoliday",
    
})

#convert isHoliday to binary
df["isHoliday"] = df["isHoliday"].apply(lambda x: 0 if x == "0" else 1)

#create a date column
start_date = pd.to_datetime("2000-01-01")  # change to any start date
df["Date"] = pd.date_range(start=start_date, periods= 10000, freq="D")

#add emergency_cases column with random integers between 5 and 60
df["emergency_cases"] = np.random.randint(5, 60, size=len(df))

print(df.head())



# Save the DataFrame to a new CSV
output_path = Path(__file__).parent / "FinalData.csv"
df.to_csv(output_path, index=False)  # index=False prevents adding the row numbers as a column

print(f"Data saved to {output_path}")
