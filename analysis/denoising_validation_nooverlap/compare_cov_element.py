import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../../script/')
from ImpliedReturnLib import *
from denoisingLib import *
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import linregress

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
#var0_ind = np.where(np.logical_or(returns_1to11.var(axis=1)<0.000000001, returns_2to12.var(axis=1)<0.000000001))[0]
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

#diagonal_indices = np.where(np.logical_or(np.diagonal(cov_1to11) == 0, np.diagonal(cov_2to12) == 0))[0]
#
## Create a mask for rows and columns to exclude
#exclude_mask = np.ones(cov_1to11.shape, dtype=bool)
#
## Set the rows and columns to exclude based on the diagonal indices
#exclude_mask[diagonal_indices, :] = False
#exclude_mask[:, diagonal_indices] = False
#
## Apply the mask to get the filtered array
#cov_1to11 = cov_1to11[exclude_mask]
#cov_2to12 = cov_2to12[exclude_mask]
#size = int(np.sqrt(cov_1to11.size))
#cov_1to11 = cov_1to11.reshape((size, size))
#cov_2to12 = cov_2to12.reshape((size, size))

print('before')
cov_1to11_denoise = covariance_denoise(returns_1to11)
print('after')


x = cov_1to11.flatten()
y = cov_2to12.flatten()

## Filter the data based on your threshold values
#x_threshold = 1
#y_threshold = 1
#filtered_indices = (x < x_threshold) & (y < y_threshold)
#x = x[filtered_indices]
#y = y[filtered_indices]
res = linregress(x, y)
xx = np.linspace(-0.01, 0.05, 20)
fit_line = res.slope*xx + res.intercept
print(res)


#plt.figure(figsize=(10,5))
plt.plot(x, y, '.')
plt.plot(xx, fit_line, label=f"y=ax+b \
\na={res.slope:.3f} +/- {res.stderr:.3f} \
\nb={res.intercept:.3f} +/- {res.intercept_stderr:.3f} \
\nr={res.rvalue:.3f}")
plt.xlabel('$S_{ij}$')
plt.ylabel('$S_{ij}$ after 6 month')
plt.legend()
#plt.grid(True)
#plt.show()
plt.savefig('cov_element_6month_compare.png')
plt.close()


x = cov_1to11_denoise.flatten()
y = cov_2to12.flatten()
res = linregress(x, y)
xx = np.linspace(-0.01, 0.05, 20)
fit_line = res.slope*xx + res.intercept
print(res)


#plt.figure(figsize=(10,5))
plt.plot(x, y, '.')
plt.plot(xx, fit_line, label=f"y=ax+b \
\na={res.slope:.3f} +/- {res.stderr:.3f} \
\nb={res.intercept:.3f} +/- {res.intercept_stderr:.3f} \
\nr={res.rvalue:.3f}")
plt.xlabel('denoised $S_{ij}$')
plt.ylabel('$S_{ij}$ after 6 month')
plt.legend()
#plt.grid(True)
#plt.show()
plt.savefig('cov_element_6month_compare_denoised.png')
