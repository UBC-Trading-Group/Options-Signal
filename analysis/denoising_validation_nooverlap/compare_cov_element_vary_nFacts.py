import sys
sys.path.insert(0, '../../script/')
import numpy as np
import pandas as pd
from ImpliedReturnLib import *
from denoisingLib import *
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import linregress

def covariance_denoise_custom_nFacts(returns, decay=0.994, nFacts0=1):
    N = returns.shape[0]
    T = 1 / (1 - decay)
    q = T / N
    #print('N:', N, 'T:', T, 'q:', q)

    cov0 = covariance_ewma(returns, decay)
    corr0 = cov2corr(cov0)
    eVal0, eVec0 = getPCA(corr0)
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth=0.25)
    #nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
    #print('nFacts0:', nFacts0)
    corr1 = denoisedCorr(eVal0, eVec0, nFacts0)
    cov1 = corr2cov(corr1, np.diag(cov0) ** 0.5)
    return cov1

def remove_stocks_with_consecutive_zeros(matrix, max_consecutive_zeros):
    ind = []
    shape = matrix.shape
    for i in range(shape[0]):
        #print(np.max(np.convolve(matrix[i, :] == 0, np.ones(max_consecutive_zeros), mode='valid')))
        if np.any(np.convolve(matrix[i, :] == 0, np.ones(max_consecutive_zeros), mode='valid') == max_consecutive_zeros):
            ind += [i]
    print('n filtered stocks:', len(ind))
    # Filter matrix based on zero rows
    exclude_mask = np.ones(matrix.shape, dtype=bool)
    exclude_mask[ind, :] = False
    filtered_matrix = matrix[exclude_mask]
    filtered_matrix = filtered_matrix.reshape(-1, shape[1])
    return filtered_matrix
daily_df = pd.read_csv("1y_processed.csv")

# Pivot the DataFrame to create a 2D array
pivot_df = daily_df.pivot(index='Permno', columns='Date', values='Returns')

# Fill missing values with 0
pivot_df = pivot_df.fillna(0)

# Convert the pivot_df to a NumPy array
returns_all = pivot_df.to_numpy()
returns_all[returns_all>20] = 20

returns_all = remove_stocks_with_consecutive_zeros(returns_all, 7)
print(returns_all.shape)

# remove rows (stocks) that have too many zero-return days
is0_num = np.sum(returns_all==0, axis=1)
ind = np.where(is0_num>150)[0]
exclude_mask = np.ones(returns_all.shape, dtype=bool)
exclude_mask[ind, :] = False
shape = returns_all.shape
returns_all = returns_all[exclude_mask]
returns_all = returns_all.reshape(-1, shape[1])

#returns_all = returns_all[returns_all.var(axis=1) != 0]
returns_1to11 = returns_all[:,:180]
returns_2to12 = returns_all[:,180:]

## remove rows (stocks) that have zero variance
#var0_ind = np.where(np.logical_or(returns_1to11.var(axis=1)==0, returns_2to12.var(axis=1)==0))[0]
#exclude_mask1 = np.ones(returns_1to11.shape, dtype=bool)
#exclude_mask1[var0_ind, :] = False
#exclude_mask2 = np.ones(returns_2to12.shape, dtype=bool)
#exclude_mask2[var0_ind, :] = False
#shape1 = returns_1to11.shape
#shape2 = returns_2to12.shape
#returns_1to11 = returns_1to11[exclude_mask1]
#returns_2to12 = returns_2to12[exclude_mask2]
#returns_1to11 = returns_1to11.reshape(-1, shape1[1])
#returns_2to12 = returns_2to12.reshape(-1, shape2[1])


cov_1to11 = covariance_ewma(returns_1to11)
cov_2to12 = covariance_ewma(returns_2to12)

nFacts_list = [10, 100, 1000, 5000, len(cov_1to11)]
rvalue_list = []

for i in nFacts_list:
    print('processing:', i)
    if i==len(cov_1to11):
        cov_1to11_denoise = cov_1to11
    else:
        cov_1to11_denoise = covariance_denoise_custom_nFacts(returns_1to11, nFacts0=i)
    x = cov_1to11_denoise.flatten()
    y = cov_2to12.flatten()
    res = linregress(x, y)
    rvalue_list += [res.rvalue]

print(rvalue_list)
plt.plot(nFacts_list, rvalue_list, 'o')
plt.xlabel('retained eigenval')
plt.ylabel('rvalue')
plt.xscale("log")
#plt.legend()
#plt.grid(True)
#plt.show()
plt.savefig('nFact_vs_r.png')
plt.close()

cov0 = cov_1to11
corr0 = cov2corr(cov0)
eVal0, eVec0 = getPCA(corr0)
plt.hist(np.diag(eVal0), bins=1000)
plt.ylim(0, 50)
plt.xlim(-10, 500)
plt.xlabel('eigenval')
plt.ylabel('count')
plt.savefig('eval_distribution.png')


