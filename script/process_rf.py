import numpy as np
import pandas as pd
from datetime import datetime
df_riskFreeRate = pd.read_csv("../data/riskFreeRate.csv") # 3-month T bill rate, unit is "% per year"

# Convert the "MCALDT" column to pandas Timestamp
df_riskFreeRate['MCALDT'] = pd.to_datetime(df_riskFreeRate['MCALDT'])

# Set the "MCALDT" column as the index of the DataFrame
df_riskFreeRate.set_index('MCALDT', inplace=True)

# Use resample to convert to daily frequency and forward fill the data to fill missing values
daily_df = df_riskFreeRate.resample("D").ffill()

# Reset the index to move "Date" back to a regular column
daily_df = daily_df.reset_index()

daily_df.to_csv('../data/riskFreeRate_processed.csv', index=False)
