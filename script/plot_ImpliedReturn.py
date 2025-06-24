import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ImpliedReturnLib import *
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

def get_business_days(start_date, end_date):
    # Create a custom business day with the US Federal Holidays
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)
    business_day = CustomBusinessDay(holidays=holidays)

    # Generate a range of dates between start and end date (inclusive)
    date_range = pd.date_range(start=start_date, end=end_date, freq=business_day)

    return date_range



# Read CSV files
df_stockprice = pd.read_csv("../data/stock_mock_universe_4_stocks.csv") # stock price data
df_riskFreeRate = pd.read_csv("../data/riskFreeRate_processed.csv") # 3-month T bill rate, unit is "% per year"
df_vix = pd.read_csv("../data/vix_processed.csv") # vix, unit is "% per year

# preprocessing vix
df_vix['Date'] = pd.to_datetime(df_vix['Date'])

# preprocessing riskFreeRate
df_riskFreeRate['MCALDT'] = pd.to_datetime(df_riskFreeRate['MCALDT'])


start_date = "1997-01-04"
end_date = "2022-12-30"
business_days = get_business_days(start_date, end_date)
arr_return = np.zeros((4, len(business_days)))

r_mkt_prev = np.array([0,0,0,0])

for i in range(len(business_days)):
    print(business_days[i].strftime("%Y-%m-%d"))
    try:
        expected_return = ExpectedReturn(df_vix, df_riskFreeRate, business_days[i].strftime("%Y-%m-%d"))
        covariance_matrix = CovarianceMatrix(df_stockprice, business_days[i].strftime("%Y-%m-%d"))
        w_mkt = CapitalizationWeight(df_stockprice, business_days[i].strftime("%Y-%m-%d"))
        r_mkt = expected_return * covariance_matrix @ w_mkt / (w_mkt.T @ covariance_matrix @ w_mkt)
    except:
        r_mkt = r_mkt_prev
    r_mkt_prev = r_mkt
    arr_return[:, i] = r_mkt

plt.figure(figsize=(10,5))
plt.plot(business_days, arr_return[0], color='grey', label='XOM')
plt.plot(business_days, arr_return[1], color='brown', label='CHV')
plt.plot(business_days, arr_return[2], color='green', label='JNJ')
plt.plot(business_days, arr_return[3], color='orange', label='LLY')
plt.legend()
plt.xlim([pd.to_datetime('1995-06-01'), pd.to_datetime('2024-01-01')])
plt.xlabel('Date')
plt.ylabel('Implied Equilibrium Return (Annual)')
plt.title('Implied Equilibrium Return Over Time')
plt.grid(True)
plt.savefig('../fig/Implied_Equilibrium_Return_Over_Time_4stock.png', bbox_inches='tight')

for y in range(1997, 2023):
    plt.xlim([pd.to_datetime(str(y)+'-01-01'), pd.to_datetime(str(y+1)+'-01-01')])
    plt.savefig('../fig/r_mkt_4stock_per_year/Implied_Equilibrium_Return_Over_Time_4stock_'+str(y)+'.png', bbox_inches='tight')

