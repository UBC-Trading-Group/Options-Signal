import numpy as np
import pandas as pd
from datetime import datetime
df_vix = pd.read_csv("../data/vix.csv") # 3-month T bill rate, unit is "% per year"

# Convert the "Date" column to pandas Timestamp
df_vix['Date'] = pd.to_datetime(df_vix['Date'])

df_vix.drop_duplicates(subset="Date", inplace=True)

# Set the "Date" column as the index of the DataFrame
df_vix.set_index('Date', inplace=True)
df_vix.dropna(subset=["vix"], inplace=True)

# Use resample to convert to daily frequency and forward fill the data to fill missing values
daily_df = df_vix.resample("D").ffill()

# Reset the index to move "Date" back to a regular column
daily_df = daily_df.reset_index()

daily_df.to_csv('../data/vix_processed.csv', index=False)
