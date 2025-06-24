import sys
sys.path.insert(0, '../../script/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from denoisingLib import *

# Read CSV files
df_spy = pd.read_csv("../../data/spy.csv") # stock price data
df_riskFreeRate = pd.read_csv("../../data/riskFreeRate_processed.csv") # 3-month T bill rate, unit is "% per year"
df_vix = pd.read_csv("../../data/vix_processed.csv") # vix, unit is "% per year"

# preprocessing vix
df_vix['Date'] = pd.to_datetime(df_vix['Date'])

# preprocessing riskFreeRate
df_riskFreeRate['MCALDT'] = pd.to_datetime(df_riskFreeRate['MCALDT'])

df_vix = df_vix.set_index('Date')
df_riskFreeRate = df_riskFreeRate.set_index('MCALDT')
df_vixrf = pd.merge(df_vix, df_riskFreeRate, left_index=True, right_index=True)
df_vixrf['Rfvix^2'] = np.exp(df_vixrf['TMYTM']/100/12) * (df_vixrf['vix']/100)**2

df_spy['date'] = pd.to_datetime(df_spy['date'])
df_spy['portfolio'] = 1
df_spy['portfolio_riskfree'] = 1
df_spy['market_weight'] = 1
df_spy['sharpe'] = 0
df_spy['market_sharpe'] = 0
for i in range(1, len(df_spy)-1):
    port = df_spy.loc[i-1, 'portfolio']
    port_rf = df_spy.loc[i-1, 'portfolio_riskfree']
    prc_yesterday = df_spy.loc[i-1, 'PRC']
    prc_today = df_spy.loc[i, 'PRC']
    date = df_spy.loc[i, 'date']

    # Correcting date-1
    date_yesterday = date - pd.DateOffset(days=1)
    rfvix2 = df_vixrf.loc[date_yesterday, 'Rfvix^2']
    rf = df_vixrf.loc[date_yesterday, 'TMYTM']
    #rfvix2 = rfvix2*5 if rfvix2*5 < 1 else 1
    rfvix2 = rfvix2*5
    df_spy.loc[i, 'market_weight'] = rfvix2
    df_spy.loc[i, 'portfolio'] = port*rfvix2*(prc_today/prc_yesterday) + port*(1-rfvix2)*(1+rf/100/252)
    df_spy.loc[i, 'portfolio_riskfree'] = port_rf*(1+rf/100/252)
    
    #convert asset values to returns
    df_spy.loc[i, 'returns_portfolio'] = (df_spy.loc[i, 'portfolio'] - df_spy.loc[i-1, 'portfolio'])/df_spy.loc[i-1, 'portfolio']
    df_spy.loc[i, 'returns_portfolio_riskfree'] = (df_spy.loc[i, 'portfolio_riskfree'] - df_spy.loc[i-1,
    'portfolio_riskfree'])/df_spy.loc[i-1, 'portfolio_riskfree']
    df_spy.loc[i, 'returns_PRC'] = (df_spy.loc[i, 'PRC'] - df_spy.loc[i-1, 'PRC'])/df_spy.loc[i-1, 'PRC']
    if i > 1260:
        df_spy.loc[i, 'sharpe'] = 252**0.5 * (df_spy.loc[i-1260:i,'returns_portfolio']-df_spy.loc[i-1260:i,'returns_portfolio_riskfree']).mean()/df_spy.loc[i-1260:i,'returns_portfolio'].std()
        df_spy.loc[i, 'market_sharpe'] = 252**0.5 * (df_spy.loc[i-1260:i,'returns_PRC']-df_spy.loc[i-1260:i,'returns_portfolio_riskfree']).mean()/df_spy.loc[i-1260:i,'returns_PRC'].std()
df_spy['returns_PRC'] = df_spy['returns_PRC']-df_spy['returns_portfolio_riskfree']
df_spy['returns_portfolio'] = df_spy['returns_portfolio']-df_spy['returns_portfolio_riskfree']

#1996 to ~2010
#print('sharpe SPY:', 252**0.5 * df_spy['returns_PRC'][:4000].mean()/df_spy['returns_PRC'][:4000].std())
#print('sharpe timing:', 252**0.5 * df_spy['returns_portfolio'][:4000].mean()/df_spy['returns_portfolio'][:4000].std())
#1996 to 2022
#sr_SPY = 252**0.5 * df_spy['returns_PRC'].mean()/df_spy['returns_PRC'].std()
#sr_portfolio = 252**0.5 * df_spy['returns_portfolio'].mean()/df_spy['returns_portfolio'].std()

fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 7))
ax.plot(df_spy['date'], df_spy['PRC']/62.14063, color='0', label='SPY')
ax.plot(df_spy['date'], df_spy['portfolio'], color='0.4', label='market timing')
ax.plot(df_spy['date'], df_spy['portfolio_riskfree'], color='0.8', label='risk free')
ax.plot(df_spy['date'], df_spy['market_weight'], label='market weight', color='blue')
ax.legend(loc='upper left')
ax.set_ylabel('portfolio')
ax.grid(True)


ax2.plot(df_spy['date'][1260:], df_spy['market_sharpe'][1260:], label='SPY', color='0')
ax2.plot(df_spy['date'][1260:], df_spy['sharpe'][1260:], label='market timing', color='0.4')
ax2.legend()
ax2.set_ylabel('5-yr Sharpe')
ax2.set_xlabel('Date')
ax2.set_xlim([pd.to_datetime('1996-01-02'), pd.to_datetime('2022-12-20')])
ax2.grid(True)

#plt.show()
plt.savefig('vix_market_timing.png', bbox_inches='tight')

