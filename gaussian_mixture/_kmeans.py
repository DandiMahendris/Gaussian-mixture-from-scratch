import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from time import time
import numpy as np
from scipy.special import logsumexp
from scipy import linalg, sparse

from ._base import BaseEstimator


def row_norms(X, squared=False):
    """Row-wise (squared) Eucladian norm of X.
    
    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse matrices and
    does not create an X.shape-sized temporary.
    
    Parameters
    ----------
    X : array-like
        The input array
    squared : bool, default=False
        If True, return squared norms.
        
    Returns
    ---------
    array-like
        the row-wise (squared) Eucladian norm of X.
    """
    
    if sparse.issparse(X):
        if not sparse.isspmatrix_csr(X):
            X = sparse.csr_matrix(X)
            
        ## Squared L2 norm of each row in CSR matrix X.
        norms = sq;
        

def kmeans_plusplus(
    X,
    n_clusters,
    *,
    sample_weight=None,
    x_squared_norms=None,
    random_state=None,
    n_local_trials=None,
):
    """Init n_cluters seed according to k-means++.
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to picks seeds form.
    
    n_clusters : int
        The number of centroids to initialize.
    
    sample_weight : array-like of shape (n_samples,),  default=None
        The weight of each observation in X. If None, all observations are assigned equal weight.
    
    x_squared_norms : array-like of shape (n_samples,), default=None
        Squared euclidean norm of each data point.
        
    random_state : int or RandomState instance, default=None
        Determines random number generator for centroid initialization. Pass an int for reproducible output
        across multiple function calls.
        
    Returns
    ---------
    centers : ndarray of shape (n_clusters, n_features)
        the initial centers of k-means.
        
    indices : ndarray of shape (n_clusters, )
        The index location of chosen centers in the data array of X. for a given index and center, 
        X[index] = center
        
    Examples
    ---------
    >>> from ._kmeans import kmeans_plusplus
    >>> import numpy as np
    >>> X = np.array([[1,2], [1,4], [1,0], [10,2], [10,4], [10,0]])
    
    >>> centers, indices = kmeans_plusplus(X, n_clusters=2, random_state=42)
    
    >>> centers
        array([[10, 2], [1, 0]])
    >>> indices
        array([3, 2])
    """
    ## If number of n_samples is less than n_clusters initialize
    if X.shape[0] < n_clusters:
        raise ValueError(
            f"n_samples={X.shape[0]} should be >= n_clusters={n_clusters}."
        )
    
    ## Check parameters
    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)