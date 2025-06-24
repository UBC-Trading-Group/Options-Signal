import pandas as pd
import numpy as np
from datetime import datetime

#extract options with specified date
df = pd.read_csv('../data/SPX/opprcd2007.csv')
df_selected = df[df['date']=='2007-01-03']
df_selected.loc[:, 'mid'] = (df_selected['best_bid'] + df_selected['best_offer']) / 2


#find the expiration date that is lower and closest to 30 days
list_of_expiration = df_selected['exdate'].to_numpy()
list_of_expiration = np.sort(np.unique(list_of_expiration))
date_cur = datetime.strptime('2007-01-01', "%Y-%m-%d")
for i in range(len(list_of_expiration)):
    date_expiration = datetime.strptime(list_of_expiration[i], "%Y-%m-%d")
    diff = date_expiration - date_cur
    if diff.days > 30:
        break
index_near_term = i-1
index_next_term = i
date_near_term = datetime.strptime(list_of_expiration[index_near_term], "%Y-%m-%d")
date_next_term = datetime.strptime(list_of_expiration[index_next_term], "%Y-%m-%d")
T_1 = (date_near_term - date_cur).days/365
T_2 = (date_next_term - date_cur).days/365
str_T_1 = list_of_expiration[index_near_term]
str_T_2 = list_of_expiration[index_next_term]

########################################
#Calculate near term variance
########################################
R_1 = 0 #to-do: substitute it to be bond-equivalent yield of US T-bills
dK = 1 #K interval
df_1 = df_selected[df_selected['exdate']==str_T_1]
df_1_call = df_1[df_1['cp_flag']=='C']
df_1_put = df_1[df_1['cp_flag']=='P']

# (optional) save the csv for the near-term call/put option
#df_1_call.to_csv('tmp_call.csv', index=False)
#df_1_put.to_csv('tmp_put.csv', index=False)

strikes = df_1.loc[:, 'strike_price'].to_numpy()
strikes = np.sort(np.unique(strikes))

#find forward price
min_mid_diff = float('inf')                 #minimum of mid-price difference between put and call
df_1_call.loc[:,'mid_diff'] = float('inf')  #mid-price difference between put and call
index_min_mid_diff = 0                      #index of a strike price where mid-price difference between put and call is minimum
for i in range(len(df_1_call)):
    call_mid = df_1_call.loc[df_1_call.index[i],'mid']
    put_mid = df_1_put.loc[df_1_put.index[i],'mid']
    mid_diff = abs(call_mid - put_mid)
    df_1_call.loc[df_1_put.index[i],'mid_diff'] = mid_diff
    if mid_diff < min_mid_diff:
        min_mid_diff = mid_diff
        index_min_mid_diff = i

#Forward price
F = df_1_call.loc[df_1_call.index[index_min_mid_diff],'strike_price'] + \
    np.exp(R_1*T_1)*(df_1_call.loc[df_1_call.index[index_min_mid_diff],'mid'] - df_1_put.loc[df_1_put.index[index_min_mid_diff],'mid'])

I = 0 #integral
for i in range(len(strikes)-1):
    I += min(df_1_put.loc[df_1_put['strike_price'] == strikes[i],'mid'].values[0], df_1_call.loc[df_1_call['strike_price']==
    strikes[i],'mid'].values[0]) / strikes[i]**2 * (strikes[i+1] - strikes[i])
I = I * 1000 #strike prices in WRDS is 1000 times the stock price


variance = 2/T_1 * np.exp(R_1*T_1) * I
VIX = 100 * variance**0.5
print(VIX) #VIX calculated from the near-term option


########################################
#Calculate next term variance
########################################

#similar to the calculation in near-term
