import numpy as np
from scipy import linalg, stats

# Here is a simple toy model to investigate the eigenvalue distribution of a 10x10 sampled covariance matrix.
# We generate a population covariance and then use it to compute random variates with limited degrees of freedom.
# Then, we compare the expectation value for the sampled eigenvalues with the original set of population eigenvalues.

# First, we generate the population eigenvalues using a chi-squared distribution with one degree of freedom.
# This is simplistic but the important point is that the eigenvalues constitute a spectrum (it's not the identity).
# In practice, the largest eigenvalue should probably be much larger but this is just illustrative
eigenvalues = np.sort(stats.chi2.rvs(1, size=10))
#eigenvalues = np.array(range(1,11))
# The covariance is diagonal but as we only look at eigenvalues, it doesn't affect the result.
# We could for example rotate the covariance using a random rotation, but it's pointless for our purpose.
cov = np.diag(eigenvalues)
# We generate 10'000 random matrices with 20 degrees of freedom.
# This is similar to using 20 days return to compute a 10x10 covariance
df = 20
W = stats.wishart.rvs(df=df, scale=cov / df, size=10000)
# We calculate the eigenvalues for each matrix
L = np.vectorize(linalg.eigvalsh, signature='(n,n)->(n)')(W)
# We print the average eigenvalue and the original set
print(np.mean(L, axis=0))
print(eigenvalues)
