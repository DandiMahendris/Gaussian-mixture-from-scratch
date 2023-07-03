import numpy as np
from scipy import linalg

from ._base import BaseEstimator
from ._kmeans import row_norms

def _compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.
    
    Parameters
    ---------
    covariances : array-like
        The covariances matrix of the current components.
        
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.
        
    Returns
    -------
    precision_cholesky : array-like
        The Cholesky decomposition of sample precisions of the current components.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariances (for instance caused  by singleton) "
        "or collapsed sample). Try to decrease the number of components, "
        "or increase reg_covar."
    )
    
    if covariance_type == "full":
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(
                cov_chol, np.eye(n_features), lower = True
            ).T

    elif covariance_type == "tied":
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(
            cov_chol, np.eye(n_features), lower=True
        ).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1.0 / np.sqrt(covariances)
    
    return precisions_chol

def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.
    
    Returns
    -------
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == "full":
        n_components, _, _ = matrix_chol.shape
        log_det_chol = np.sum(
            np.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1
        )
    
    elif covariance_type == "tied":
        log_det_chol = np.sum(np.log(np.diag(matrix_chol)))

    elif covariance_type == "diag":
        log_det_chol = np.sum(np.log(matrix_chol), axis=1)

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol

def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    means : array-like of shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # The determinant of the precision matrix from the Cholesky decomposition
    # corresponds to the negative half of the determinant of the full precision
    # matrix.
    # In short: det(precision_chol) = - det(precision) / 2
    log_det = _compute_log_det_cholesky(precisions_chol, covariance_type, n_features)

    if covariance_type == "full":
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == "tied":
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == "diag":
        precisions = precisions_chol**2
        log_prob = (
            np.sum((means**2 * precisions), 1)
            - 2.0 * np.dot(X, (means * precisions).T)
            + np.dot(X**2, precisions.T)
        )

    elif covariance_type == "spherical":
        precisions = precisions_chol**2
        log_prob = (
            np.sum(means**2, 1) * precisions
            - 2 * np.dot(X, means.T * precisions)
            + np.outer(row_norms(X, squared=True), precisions)
        )
    # Since we are using the precision of the Cholesky decomposition,
    # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

def _estimate_gaussian_parameters_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.
    
    Returns
    ------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar
    
    return covariances

def _estimate_gaussian_parameters_tied(resp, X, nk, means, reg_covar):
    """Estimate the tied covariance matrices.
    
    Reutrns
    ---------
    covariances : array, shape (n_features, n_features)
        The tied covariance matrix of the components.
    """
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * means.T, means)
    covariances = avg_X2 - avg_means2
    covariances /= nk.sum()
    covariances.flat[:: len(covariances) + 1] += reg_covar
    
    return covariances

def _estimate_gaussian_parameters_diag(resp, X, nk, means, reg_covar):
    """Estimate the diag covariances matrices.
    
    Returns
    -------
    covariances : array, shape (n_components, n_features)
        The covariance vector of the current components.
    """
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means**2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar

def _estimate_gaussian_parameters_spherical(resp, X, nk, means, reg_covar):
    """Estimate the spherical covariances matrices.
    
    Returns
    -------
    variances : array, shape (n_components, )
        The variances values of each components.
    """
    
    return _estimate_gaussian_parameters_diag(resp, X, nk, means, reg_covar).means(1)

def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type):
    """Estimate the Gaussian paramters.
    
    Parameters
    ---------
    X : array-like of shape (n_samples, n_features)
    
    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.
        
    reg_covar : float
        The regularization added to the diagonal of the covariances matrices.
        
    covariance_type : {"full", "tied", "diag", "spherical"}
        The type of precisions matrices.
        
    Returns
    -------
    nk : array-like of shape (n_components, )
        The numbers of data samples in the current components.
        
    means : array-like of shape (n_components, n_features)
        The centers of the current components.
        
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends on the covariance_type.
    """
    
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = {
        "full": _estimate_gaussian_parameters_full,
        "tied": _estimate_gaussian_parameters_tied,
        "diag": _estimate_gaussian_parameters_diag,
        "spherical": _estimate_gaussian_parameters_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    
    return nk, means, covariances

class GaussianMixture(BaseEstimator):
    """Gaussian Mixture.
    
    Representation of a Gaussian Mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture distribution.
    
    Parameters
    -----------
    n_components : int, default=1
        the number of mixture components.
        
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        - 'full': each components has its own general covariance matrix.
        - 'tied': all components share the same general covariance matrix.
        - 'diag': each components has its own diagonal covariance matrix.
        - 'spherical': each components has its own single variance.
        
    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.
        
    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positives.
        
    max_iter : float, default=100
        The number of EM iterations to perform.
        
    n_init : int, default=1
        The number of initialization perform. The best result are kept.
        
    init_params : {'kmeans', 'random', 'k-means++'}, default='k-means++'
        The method used to initialize the weights, the means and the precisions.
        
        - 'kmeans': responsibilities are initialized using k-means.
        - 'random': responsibilities are initialized randomly.
        - 'k-means++': use the k-means++ method to initialize.
        
    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the parameters
        
    warm_start : bool, default=False
        If `warm_start` is True, the solution of the last fitting is used as initialization for the next call of fit()
        This can be speed up convergence when fit is called several times on similar problems.
        In that case, `n_init` is ignored and only a single intialization occurs upon the first call.
        
    verbose : int, default=0
        Enable verbose output, If 1 then its print the current initialization and each iteration step.
        If greater than 1 it prints also the log probability and the time needed for each step.
        
    verbose_interval : int, default=10
        Number of iterations done before the next print.
        
    Attributes
    ---------
    weights_ : array-like of shape (n_components,)
        The weight of each mixture components.
        
    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture components.
        
    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`.
        
            (n_components, )                        if `spherical`,
            (n_features, n_components)              if `tied`,
            (n_components, n_features)              if `diag`,
            (n_components, n_features, n_features)  if `full`
            
    precisions_ : array-like
        The precision matrices for each componen in the mixture. A precision matrix is the inverse of a covariance 
        matrix. A covariance matrix is symmetric positive definite so the mixture of Gaussian can be equivalently
        parameterized by the precision matrices. Storing the precision matrices instead of the covariance matrices
        make it more efficient to compute the log-likelihood.
        
    precisions_cholesky_ : array-like
        The cholesky decompositions of the precision matrices of each mixture component.
        
    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
        
    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.
        
    lower_bound : int
        lower bound value of the log-likelihood (of the training data with respect to the model)
        of the best fit of EM.
        
    n_features_in_ : int
        Number of features seen during fit.
        
    features_names_in : ndarray of shape (`n_features_in`,)
        Names of features seen during fit
        
    Examples
    -------
    >>> import numpy as np
    >>> from sklearn.mixture import GaussianMixture
    >>> X = np.array([[1,2], [1,4], [1,0], [10,2], [10,4], [10,0]])
    >>> gm = GaussianMixture(n_components=2, random_state=42).fit(X)
    >>> gm.means_
        array([[10., 2.], [1., 2.]])
    >>> gm.predict([[0, 0], [12, 3]])
        array([1, 0])
    """
    
    def __init__(
        self,
        n_components=1,
        *,
        covariance_type='full',
        tol = 1e-3,
        reg_covar = 1e-6,
        max_iter = 100,
        n_init = 1,
        init_params = 'k-means++',
        precisions_init = None,
        random_state = None,
        warm_start = False,
        verbose = 0,
        verbose_interval = 10,
    ):
        super().__init__(
            n_components= n_components,
            tol = tol,
            reg_covar = reg_covar,
            max_iter = max_iter,
            n_init = n_init,
            init_params = init_params,
            random_state = random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )
        self.covariance_type = covariance_type
        self.precisions_init = precisions_init
        
    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        
        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape
        
        weights, means, covariances = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type
        )
        
        weights /= n_samples
        
        self.weights_ = weights
        self.means_ = means
        
        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type
            )
        
        elif self.covariance_type == "full":
            self.precisions_cholesky_ = np.array(
                [
                    linalg.cholesky(prec_init, lower=True)
                    for prec_init in self.precisions_init
                ]
            )
        
        elif self.covariance_type == 'tied':
            self.precisions_cholesky_ = linalg.cholesky(
                self.precisions_init, lower = True
            )
        
        else:
            self.precisions_cholesky_ = np.sqrt(self.precisions_init)
            
    def _m_step(self, X, log_resp):
        """M step.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        
        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities of the point of each sample in X.
        """
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )
    
    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type
        )
        
    def _estimate_log_weights(self):
        return np.log(self.weights_)
    
    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm
    
    def _get_parameter(self):
        return (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        )
    
    def _set_parameter(self, params):
        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        ) = params
        
        # Attribute computation
        _, n_features = self.means_.shape
        
        if self.covariance_type == "full":
            self.precisions_ = np.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = np.dot(prec_chol, prec_chol.T)
                
        elif self.covariance_type == "tied":
            self.precisions_ = np.dot(
                self.precisions_cholesky_, self.precision_cholesky_.T
            )
            
        else:
            self.precisions_ = self.precisions_cholesky_**2
            
    def _n_parameters(self):
        """Return the number of free components in the model."""
        
        _, n_features = self.means_.shape
        if self.covariance_type == "full":
            cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "diag":
            cov_params = self.n_components * n_features
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "spherical":
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        
        return int(cov_params + mean_params + self.n_components - 1)
    
    def bic(self, X):
        """Bayesian information criterion.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        --------
        bic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + self._n_parameters() * np.log(X.shape[0])
    
    def aic(self, X):
        """Akaike Information Criterion.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()