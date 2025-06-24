import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from denoisingLib import *

def RiskAversion(df_vix, df_riskFreeRate, date='2022-01-03'):
    # Function to calculate risk aversion based on equation (15) in Ian Martin 2017 paper
    # Parameters:
    #   - df_vix: DataFrame containing the pre-calculated SVIX data
    #   - df_riskFreeRate: Risk-free interest rate (e.g. 3 month T-bill)
    #   - date: Date for which risk aversion is calculated (default: '2022-01-03')
    # Returns:
    #   - risk aversion

    # Define a time window from start_dd to dd
    dd = datetime.strptime(date, "%Y-%m-%d")
    start_dd = dd - pd.DateOffset(days=90)
    
    # Filter the DataFrame to get the data within the specified window
    rf_window = df_riskFreeRate.loc[(df_riskFreeRate['MCALDT'] >= start_dd) & (df_riskFreeRate['MCALDT'] <= dd),'TMYTM'].values
    vix_window = df_vix.loc[(df_vix['Date'] >= start_dd) & (df_vix['Date'] <= dd),'vix'].values

    expected_return_wondow = np.exp(rf_window/100/12) * (vix_window/100)**2
    return expected_return_wondow[-1] / np.var(expected_return_wondow)

def ExpectedReturn(df_vix, df_riskFreeRate, date='2022-01-03'):
    # Function to calculate expected reutrn based on equation (15) in Ian Martin 2017 paper
    # Parameters:
    #   - df_vix: DataFrame containing the pre-calculated SVIX data
    #   - df_riskFreeRate: Risk-free interest rate (e.g. 3 month T-bill)
    #   - date: Date for which risk aversion is calculated (default: '2022-01-03')
    # Returns:
    #   - Expected Return

    rf_at_date = df_riskFreeRate.loc[df_riskFreeRate['MCALDT']==datetime.strptime(date, "%Y-%m-%d"), 'TMYTM'].values
    vix_at_date = df_vix.loc[df_vix['Date']==datetime.strptime(date, "%Y-%m-%d"), 'vix'].values
    return np.exp(rf_at_date/100/12) * (vix_at_date/100)**2

def CovarianceMatrix(df, date='2022-01-03', mode='ewma'):
    # Function to calculate the covariance matrix of stock returns
    # Parameters:
    #   - df: DataFrame containing stock data
    #   - date: Date for which the observation has reached (default: '2022-01-03')
    #   - mode: a string chosen from 'vanilla', 'ewma', 'denoise'
    # Returns:
    #   - Covariance matrix of stock returns
    
    # Extract unique stock identifiers
    unique_stocks = df['PERMNO'].unique().tolist()
    
    # Initialize an array to store prices
    list_prices = []

    # extract prices for each stock before specified date
    for i in range(len(unique_stocks)):
        prices = df[df['PERMNO'] == unique_stocks[i]].copy()  # Create a copy to avoid modifying the original DataFrame
        prices.loc[:, 'date'] = pd.to_datetime(prices['date'], format="%Y-%m-%d")
        list_prices.append(prices.loc[prices['date']<datetime.strptime(date, "%Y-%m-%d"), 'PRC'].values)

    # Stack prices into a 2D array of size
    array_prices = np.vstack(list_prices) # size: (n_stock, n_records)

    # Convert prices to log returns 
    returns = np.zeros((array_prices.shape[0], array_prices.shape[1]-1)) # size: (n_stock, n_records-1)
    for i in range(array_prices.shape[1] - 1, 0, -1):
        returns[:, i-1] = np.log(array_prices[:, i] / array_prices[:, i-1]) #log return
    
    # Calculate the covariance matrix
    # choose to toggle one method below
    if mode == 'vanilla':
        covariance_matrix = covariance_vanilla(returns)
    elif mode == 'ewma':
        covariance_matrix = covariance_ewma(returns)
    elif mode == 'denoise':
        covariance_matrix = covariance_denoise(returns)
    else:
        covariance_matrix = covariance_ewma(returns)

    return covariance_matrix


def covariance_vanilla(returns):
    # The most basic method to calculate the covariance matrix
    # Parameters: 
    #   - returns: an array of size (n_stock, n_records) that stores the stock returns
    # Returns:
    #   - Covariance matrix of stock returns

    covariance_matrix = np.cov(returns)
    return covariance_matrix

def covariance_ewma(returns, decay = 0.994):
    # Calculate the covariance matrix using exponentially weighted moving average model (EWMA)
    # Parameters: 
    #   - returns: an array of size (n_stock, n_records) that stores the stock returns
    # Returns:
    #   - Covariance matrix of stock returns

    # Step 1: exponential weight
    weight = np.ones(returns.shape)
    for i in range(1, weight.shape[1]):
        weight[:, i]  = weight[:, i-1] * decay #expponential decay per day

    # Step 2: Calculate the mean of each stock
    mean_returns = np.mean(returns, axis=1, keepdims=True)

    # Step 3: Center the returns matrix
    centered_returns = returns - mean_returns


    # Step 4: Calculate the covariance matrix
    covariance_matrix = np.dot(centered_returns * weight, centered_returns.T) / np.sum(weight[0,:])

    return covariance_matrix

def covariance_denoise(returns, decay=0.994):
    """
    Calculate the denoised covariance matrix of stock returns using basic method.

    Parameters:
        returns (array): An array of size (n_stock, n_records) that stores the stock returns.
        decay (float): The decay parameter for calculating T.

    Returns:
        array: Denoised covariance matrix of stock returns.
    """
    N = returns.shape[0]
    T = 1 / (1 - decay)
    q = T / N
    #print('N:', N, 'T:', T, 'q:', q)

    cov0 = covariance_ewma(returns, decay)
    corr0 = cov2corr(cov0)
    eVal0, eVec0 = getPCA(corr0)
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth=0.25)
    nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
    print('nFacts0:', nFacts0)
    #nFacts0 = int(0.7 * cov0.shape[0])
    corr1 = denoisedCorr(eVal0, eVec0, nFacts0)
    cov1 = corr2cov(corr1, np.diag(cov0) ** 0.5)
    return cov1

def CapitalizationWeight(df, date='2022-01-03'):
    # Function to calculate the capitalization weights of stocks
    # Parameters:
    #   - df: DataFrame containing stock data
    #   - date: Date for which the capitalization weights are calculated (default: '2022-01-03')
    # Returns:
    #   - Array of capitalization weights
    
    # Filter the DataFrame for the given date and sort by stock identifier
    df_at_date = df[df['date'] == date].sort_values('PERMNO')
    
    # Calculate the market capitalization for each stock
    df_at_date['cap'] = df_at_date['PRC'] * df_at_date['SHROUT']
    
    # Calculate the weight of each stock based on market capitalization
    df_at_date['weight'] = df_at_date['cap'] / sum(df_at_date['cap'])
    
    return df_at_date['weight'].values



if __name__ == "__main__":
    # Read CSV files
    df_stockprice = pd.read_csv("../data/stock_mock_universe_4_stocks.csv") # stock price data
    df_riskFreeRate = pd.read_csv("../data/riskFreeRate_processed.csv") # 3-month T bill rate, unit is "% per year"
    df_vix = pd.read_csv("../data/vix_processed.csv") # vix, unit is "% per year"

    # preprocessing vix
    df_vix['Date'] = pd.to_datetime(df_vix['Date'])
    
    # preprocessing riskFreeRate
    df_riskFreeRate['MCALDT'] = pd.to_datetime(df_riskFreeRate['MCALDT'])
    #df_riskFreeRate.set_index('MCALDT', inplace=True)
    
    # Set the risk aversion value
    risk_aversion = RiskAversion(df_vix, df_riskFreeRate, '2022-01-03')
    expected_return = ExpectedReturn(df_vix, df_riskFreeRate, '2022-01-03')
    
    # Calculate the covariance matrix of stock returns
    covariance_matrix = CovarianceMatrix(df_stockprice, '2022-01-03', mode='ewma')
    
    # Calculate the capitalization weights of stocks
    w_mkt = CapitalizationWeight(df_stockprice, '2022-01-03')
    
    # Calculate the market return
    #r_mkt1 = risk_aversion * covariance_matrix @ w_mkt
    r_mkt2 = expected_return * covariance_matrix @ w_mkt / (w_mkt.T @ covariance_matrix @ w_mkt)
    
    # Print the results
    print('risk_aversion\n', risk_aversion)
    print('w_mkt\n', w_mkt)  # Print the capitalization weights
    print('covariance_matrix\n', covariance_matrix)  # Print the covariance matrix

    # Print the implied equilibrium return, projected to 1 year
    #print('r_mkt1 with option-impled variance\n', r_mkt1*252)
    # Print the implied equilibrium return
    print('r_mkt2 with variance from history data\n', r_mkt2)
