import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import linregress
from scipy.stats import t

def autocorrelation_time(x, threshold=0.1):
    # Calculate the autocorrelation function (ACF)
    n = len(x)
    mean_x = np.mean(x)
    acf = np.correlate(x - mean_x, x - mean_x, mode='full')[len(x)-1:] / (np.var(x) * n)
    print(acf)

    # Find the first index where ACF drops below the threshold
    for lag, value in enumerate(acf):
        if value < threshold:
            return lag

    # If the threshold is not reached, return a value indicating no convergence
    return -1

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

#calculate realized return after 1 month
for i in range(1, len(df_spy)-21):
    df_spy.loc[i, 'relized_return'] = (df_spy.loc[i+20, 'PRC'] - df_spy.loc[i, 'PRC']) / df_spy.loc[i, 'PRC']
    date = df_spy.loc[i, 'date']
    df_spy.loc[i, 'expected_return'] = df_vixrf.loc[date, 'Rfvix^2'] / 12

#calculate the auto-correlation of x
x = df_spy['expected_return'].dropna().values
y = df_spy['relized_return'].dropna().values
x = x[len(x)//2:] #out of sample data (i.e. since ~2012)
y = y[len(y)//2:] #out of sample data (i.e. since ~2012)
actime = autocorrelation_time(x, 1/2.718)
print("auto correlation time:", actime)

#linear regression sub-sampled by auto-correlation time
x = df_spy['expected_return'].dropna().values[::actime]
y = df_spy['relized_return'].dropna().values[::actime]
x = x[len(x)//2:]
y = y[len(y)//2:]
res = linregress(x, y)
xx = np.linspace(-0.01, 0.05, 20)
fit_line = res.slope*xx + res.intercept
print(res)

#tinv = lambda p, df: abs(t.ppf(p/2, df))
#ts = tinv(0.05, len(x)-2)
#print(f"slope (95%): {res.slope:.6f} +/- {ts*res.stderr:.6f}")
#print(f"intercept (95%): {res.intercept:.6f} +/- {ts*res.intercept_stderr:.6f}")

pvalue_slope = t.sf(abs(res.slope-1)/res.stderr, len(x)-2) * 2 #two sided
pvalue_intercept = t.sf(abs(res.intercept-0)/res.stderr, len(x)-2) * 2 #two sided
print(pvalue_slope, pvalue_intercept)


plt.figure(figsize=(10,5))
plt.plot(x, y, '.')
plt.plot(xx, fit_line, label=f"y=ax+b \
\na={res.slope:.3f} +/- {res.stderr:.3f} \
\nb={res.intercept:.3f} +/- {res.intercept_stderr:.3f} \
\np(b=0)={pvalue_intercept:.3f} \
\np(a=1)={pvalue_slope:.3f}")
plt.xlabel('expected_return')
plt.ylabel('relized_return (1 month)')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('expectedR_vs_realizedR.png')

