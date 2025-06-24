import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../../script/')
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
    corr1 = denoisedCorr(eVal0, eVec0, nFacts0)
    cov1 = corr2cov(corr1, np.diag(cov0) ** 0.5)
    return cov1

daily_df = pd.read_csv("../denoising_validation/1y_processed.csv")

# Pivot the DataFrame to create a 2D array
pivot_df = daily_df.pivot(index='Permno', columns='Date', values='Returns')

# Fill missing values with 0
pivot_df = pivot_df.fillna(0)

# Convert the pivot_df to a NumPy array
returns_all = pivot_df.to_numpy()
returns_all[returns_all>20] = 20


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

# remove rows (stocks) that have more than 10 consecutive days with zero return
returns_all = remove_stocks_with_consecutive_zeros(returns_all, 7)
print(returns_all.shape)

# remove rows (stocks) that have too many zero-return days
is0_num = np.sum(returns_all==0, axis=1)
#plt.hist(is0_num, bins=100)
#plt.show()
ind = np.where(is0_num>150)[0]
exclude_mask = np.ones(returns_all.shape, dtype=bool)
exclude_mask[ind, :] = False
shape = returns_all.shape
returns_all = returns_all[exclude_mask]
returns_all = returns_all.reshape(-1, shape[1])
print(returns_all.shape)

return_1 = returns_all[:,:180]
return_2 = returns_all[:,180:]
cov_1 = covariance_ewma(return_1)



"""
Z = R^T * W, where 
Z (t x p) is the return in new coordinate
R (p x t) is the return matrix
W (p x p) is the eigenvector matrix, each column is a eigenvector
"""

off_list = []
nFacts = [1, 2, 3, 4, 5, 6,7,8,9, 10, 13, 15, 20, 100, 1000, 5000]
for i in range(len(nFacts)):
    print("processing nFacts=", nFacts[i])

    cov_modeled = covariance_denoise_custom_nFacts(return_1, nFacts0=nFacts[i])
    eVal1, eVec1 = getPCA(cov_modeled)
    Z1 = return_2.T @ eVec1
    zz1 = (Z1.T @ Z1)/180
    off_list += [np.sum(abs(zz1 - np.diag(np.diag(zz1))))]


plt.plot(nFacts, off_list, '.')
plt.xlabel('# eiganvec')
plt.ylabel('sum of abs off-diagonal elements')
plt.xscale('log')
#plt.legend()
#plt.xlim([-0.01, 0.5])
#plt.ylim([-0.01, 0.5])
#plt.grid(True)
#plt.show()
plt.savefig('offsum_vs_nFacts.png')

