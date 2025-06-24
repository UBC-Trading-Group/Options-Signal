import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../../script/')
from ImpliedReturnLib import *
from denoisingLib import *
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import linregress


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

# remove rows (stocks) that have more than 7 consecutive days with zero return
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

print('before')
cov_1_denoise = covariance_denoise(return_1)
print('after')

"""
Z = R^T * W, where 
Z (t x p) is the return in new coordinate
R (p x t) is the return matrix
W (p x p) is the eigenvector matrix, each column is a eigenvector
"""
eVal0, eVec0 = getPCA(cov_1)
Z0 = return_2.T @ eVec0
zz0 = (Z0.T @ Z0)/180

eVal1, eVec1 = getPCA(cov_1_denoise)
Z1 = return_2.T @ eVec1
zz1 = (Z1.T @ Z1)/180

plt.plot(np.diag(eVal0), np.diag(zz0), '.', label='original')
plt.plot(np.diag(eVal1), np.diag(zz1), '.', label='denoised')
plt.xlabel('$\lambda$')
#plt.ylabel('$R^T W$ (eigenvalue basis)')
plt.ylabel('$Z^T Z$')
plt.legend()
#plt.xlim([-0.01, 0.5])
#plt.ylim([-0.01, 0.5])
#plt.grid(True)
#plt.show()
plt.savefig('eigenval_vs_OSeigenval.png')


#compare the sum of off-diagonal values.
"""
Z^T * Z should have zero off-diagonals if eigenvectors can still diagonalize (i.e. representative of the most efficient
basis) the OOS covariance matrix 
OOS_cov = W (Z^T * Z) W^T
"""
weight_off0 = np.sum(abs(zz0 - np.diag(np.diag(zz0))))
weight_off1 = np.sum(abs(zz1 - np.diag(np.diag(zz1))))
print("model 1 vs model 2 (smaller is better)")
print(weight_off0, weight_off1)
