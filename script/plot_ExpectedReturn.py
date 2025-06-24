import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

rf_rate_data = pd.read_csv("../data/riskFreeRate.csv") # 3-month T bill rate, unit is "% per year"
vix_data = pd.read_csv("../data/vix.csv") # vix, unit is "% per year"

vix_data['Date'] = pd.to_datetime(vix_data['Date'])
rf_rate_data['MCALDT'] = pd.to_datetime(rf_rate_data['MCALDT'])

rf_rate_data.set_index('MCALDT', inplace=True)
daily_rf_rate_data = rf_rate_data.resample('D').ffill()
data = pd.merge(vix_data, daily_rf_rate_data, left_on='Date', right_index=True, how='left')

data['ExpectedReturn'] = np.exp(data.loc[:,'TMYTM']/100/12) * (data.loc[:,'vix']/100)**2

plt.figure(figsize=(10,5))
plt.plot(data['Date'], data['ExpectedReturn'])
plt.xlim([pd.to_datetime('1995-06-01'), pd.to_datetime('2024-01-01')])
plt.xlabel('Date')
plt.ylabel('Annual Return')
plt.title('Expected Market Return Over Time')
plt.grid(True)
#plt.show()
plt.savefig('../fig/Expected_Return_Over_Time.png', bbox_inches='tight')


for i in range(90, len(data)):
    data.loc[i, 'RiskAversion'] = data.loc[i, 'ExpectedReturn'] / np.var(data.loc[i-90:i, 'ExpectedReturn'])

plt.clf()
plt.plot(data['Date'], data['RiskAversion'])
plt.xlim([pd.to_datetime('1995-06-01'), pd.to_datetime('2024-01-01')])
plt.xlabel('Date')
plt.ylabel('Risk Aversion')
plt.title('Risk Aversion Over Time')
plt.grid(True)
#plt.show()
plt.savefig('../fig/Risk_Aversion_Over_Time.png', bbox_inches='tight')

