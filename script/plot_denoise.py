from denoisingLib import *
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import scipy.cluster.hierarchy as sch
 
def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]

alpha, nCols, nFact, q = 0.5, 20, 1, 2
cov=np.cov(np.random.normal(size=(nCols*q,nCols)),rowvar=0)
cov=alpha*cov+(1-alpha)*getRndCov(nCols,nFact) # noise+signal
cov = cluster_corr(cov)

denoised_cov = deNoiseCov(cov, q=q, bWidth=0.25)

# Print denoised covariance matrix
#print("Denoised Covariance Matrix:\n", denoised_cov)

# Visualize the original and denoised covariance matrices using heatmaps
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

sns.heatmap(cov, ax=axes[0], cmap="YlGnBu", center=0, vmin=np.min(cov), vmax=np.max(cov))
axes[0].set_title("Original Covariance Matrix")

sns.heatmap(denoised_cov, ax=axes[1], cmap="YlGnBu", center=0, vmin=np.min(cov), vmax=np.max(cov))
axes[1].set_title("Denoised Covariance Matrix")

sns.heatmap(denoised_cov - cov, ax=axes[2], cmap="YlGnBu", center=0, vmin=np.min(cov), vmax=np.max(cov))
axes[2].set_title("Denoised - Original")

plt.tight_layout()
plt.savefig('../fig/denoise_test.png')
plt.show()
