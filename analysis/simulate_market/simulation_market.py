import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculateCovariance(returns, time = -1):
    n_stock, n_record = returns.shape
    if time < 100 and time != -1:
        print("no enough records to calculate covariance matrix")
        return np.ones((n_stock,n_stock))
    returns = returns[:, time-100:time]

    # Step 1: Calculate the mean of each column (stock)
    mean_returns = np.mean(returns, axis=1, keepdims=True)

    # Step 2: Center the returns matrix
    centered_returns = returns - mean_returns

    # Step 3: exponential weight
    weight = np.ones(returns.shape)
    for i in range(1, weight.shape[1]):
        weight[:, i]  = weight[:, i-1] * 0.94

    # Step 4: Calculate the covariance matrix
    covariance_matrix = np.dot(centered_returns * weight, centered_returns.T) / np.sum(weight[0,:])

    return covariance_matrix

#mock returns
stock1 = 0.01 + np.random.rand(1, 100) * 0.2
stock2 = 2 * stock1
stock3 = 0.01 + np.random.rand(1, 100) * 0.02
stock4 = 2 * stock3

price_init = np.array([[1,2,3,4]]).T
#returns_init = np.random.rand(4, 100) * 0.01
returns_init = np.vstack((stock1,stock2,stock3,stock4))
risk_aversion = 5
n_horizon = 500
returns = returns_init
price = price_init
priceRecord = np.array([[1,2,3,4]]).T

covariance_matrix = calculateCovariance(returns)
print('covariance matrix:\n', covariance_matrix)
covarianceRecord = covariance_matrix.reshape((1,4,4))
print('covariance matrix Record:\n', covarianceRecord)
w_mkt = price / np.sum(price)
r_mkt = risk_aversion * covariance_matrix @ w_mkt
print('r_mkt:\n', r_mkt)

for i in range(n_horizon):
    covariance_matrix = calculateCovariance(returns)
    w_mkt = price / np.sum(price)
    r_mkt = risk_aversion * covariance_matrix @ w_mkt
#    print(r_mkt)

    price  = price * np.exp(r_mkt)
    priceRecord = np.append(priceRecord, price, axis=1)
    returns = np.append(returns, r_mkt, axis=1)
    covarianceRecord = np.append(covarianceRecord, [covariance_matrix], axis=0)

plt.plot(range(1+n_horizon), covarianceRecord[:,0,0])

#plt.plot(range(returns_init.shape[1]+n_horizon), np.log(returns[0]))
#plt.plot(range(returns_init.shape[1]+n_horizon), np.log(returns[1]))
#plt.plot(range(returns_init.shape[1]+n_horizon), np.log(returns[2]))
#plt.plot(range(returns_init.shape[1]+n_horizon), np.log(returns[3]))


#plt.plot(range(priceRecord.shape[1]), priceRecord[0]/np.sum(priceRecord, axis=0))
#plt.plot(range(priceRecord.shape[1]), priceRecord[1]/np.sum(priceRecord, axis=0))
#plt.plot(range(priceRecord.shape[1]), priceRecord[2]/np.sum(priceRecord, axis=0))
#plt.plot(range(priceRecord.shape[1]), priceRecord[3]/np.sum(priceRecord, axis=0))
plt.show()


