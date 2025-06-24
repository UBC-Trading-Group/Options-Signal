import sys
sys.path.insert(0, '../../script/')
import numpy as np
import pandas as pd
from ImpliedReturnLib import *
from denoisingLib import *
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import linregress

daily_df = pd.read_csv("1y_processed.csv")

# Pivot the DataFrame to create a 2D array
pivot_df = daily_df.pivot(index='Permno', columns='Date', values='Returns')

# Fill missing values with 0
pivot_df = pivot_df.fillna(0)

# Convert the pivot_df to a NumPy array
returns_all = pivot_df.to_numpy()
returns_all[returns_all>20] = 20

#returns_all = returns_all[returns_all.var(axis=1) != 0]
returns_1to11 = returns_all[:,:365-30]
returns_2to12 = returns_all[:,30:]

# remove rows (stocks) that have zero variance
var0_ind = np.where(np.logical_or(returns_1to11.var(axis=1)==0, returns_2to12.var(axis=1)==0))[0]
exclude_mask1 = np.ones(returns_1to11.shape, dtype=bool)
exclude_mask1[var0_ind, :] = False
exclude_mask2 = np.ones(returns_2to12.shape, dtype=bool)
exclude_mask2[var0_ind, :] = False
shape1 = returns_1to11.shape
shape2 = returns_2to12.shape
returns_1to11 = returns_1to11[exclude_mask1]
returns_2to12 = returns_2to12[exclude_mask2]
returns_1to11 = returns_1to11.reshape(-1, shape1[1])
returns_2to12 = returns_2to12.reshape(-1, shape2[1])


cov_1to11 = covariance_ewma(returns_1to11)
cov_2to12 = covariance_ewma(returns_2to12)

corr_1to11 = cov2corr(cov_1to11)

plt.hist(corr_1to11.flatten(), bins=100)
#plt.ylim(0, 50)
#plt.xlim(-10, 500)
plt.xlabel('corr')
plt.ylabel('count')
plt.savefig('corr_distribution.png')


