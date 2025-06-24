import numpy as np
import pandas as pd

##download data from database
from database import DatabaseUBCTG
db = DatabaseUBCTG()
df = db.get_returns_universe(start="2022-01-01", end="2022-12-31")
df.to_csv("1y.csv", index=False)

# Convert the "Date" column to pandas Timestamp
df['Date'] = pd.to_datetime(df['Date'])

df.drop_duplicates(subset=["Date", "Permno"], inplace=True)

# Set the "Date" column as the index of the DataFrame
df.set_index('Date', inplace=True)
df.dropna(subset=["Returns"], inplace=True)

# Use resample to convert to daily frequency and forward fill the data to fill missing values
#daily_df = df.groupby('Permno').resample("D").ffill()
daily_df = df.groupby('Permno').resample("D")['Returns'].asfreq(fill_value=0)

# Use resample to convert to weekly frequency and calculate weekly returns
#weekly_returns = df.groupby('Permno').resample('W').ffill().groupby('Permno')['Returns'].agg(lambda x: (1 + x).prod() - 1)

# Reset the index to move "Date" back to a regular column
daily_df = daily_df.reset_index()

daily_df.to_csv('1y_processed.csv', index=False)
