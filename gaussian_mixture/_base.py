import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from time import time
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from inspect import isclass


class BaseEstimator(metaclass=ABCMeta):
    """Base class for Gaussian Mixture Model.
    """
    def __init__(
        self,
        n_components,
        tol,
        reg_covar,
        max_iter,
        n_init,
        init_params,
        random_state,
        warm_start,
        verbose,
        verbose_interval,
    ):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
    
    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parametes.
        """
        n_samples, n_features = X.shape
        
        if self.init_params == "random":
            # We initialize randomly
            resp = np.zeros((n_samples, n_features))
            cluster_centers_ind = np.random.choice(n_samples,
                                                   size = self.n_components,
                                                   replace=False)
            
            resp[cluster_centers_ind, np.arange(self.n_components)] = 1
            
        elif self.init_params == 'kmeans':
            # Initialize k-means
            resp = np.zeros((n_samples, self.n_components))
            label = (
                KMeans(
                    n_clusters=self.n_components, n_init=1, random_state=random_state
                )
                .fit(X)
                .labels_
            )
            resp[np.arange(n_samples), label] = 1
            
        else:
            # Initialize k-means++
            cluster_centers=[]
            dist_sample= np.zeros(n_samples)
            proba_sample= np.zeros(n_samples)
            
            # Choose c1 randomly from data
            c_0_ind = np.random.choice(n_samples)
            cluster_centers.append(X[c_0_ind])
            
            # print(f"cluster centers 1: {list(cluster_centers[0])}")
            
            # choose c2, c3, etc
            for j in range(1, self.n_components):
                # Compute distance between data to the closest available centroid
                for i, sample_i in enumerate(X):
                    # Find closest cluster centers
                    ind_i = self._closest_cluster_centers(sample_i,
                                                          cluster_centers)
                    
                    cluster_centers_i = cluster_centers[int(ind_i)]
                    
                    # Find the distance
                    dist_i = np.linalg.norm(sample_i - cluster_centers_i)
                    
                    # Append
                    dist_sample[i] = dist_i 
                    
                # Compute probability of a point
                proba_sample = (dist_sample**2) / np.sum(dist_sample**2)
                
                # Generate random cluster centers based on the probability
                c_j_ind = np.random.choice(n_samples, p=proba_sample)
                
                # Append the cluster centers
                cluster_centers.append(X[c_j_ind])
                
            cluster_centers = np.array(cluster_centers)
            
            # Assign data points to clusters
            resp = np.zeros((n_samples, self.n_components))
            for i, sample_i in enumerate(X):
                ind_i = self._closest_cluster_centers(sample_i, cluster_centers)
                resp[i, int(ind_i)] = 1
                
        self._initialize(X, resp)
                    
    def _closest_cluster_centers(self, sample, cluster_centers):
        """Returns the index of closest centroid (cluster centers) of the sample.
        
        Parametes
        ---------
        sample : array-like of shape (1, n_features)
        
        cluster_centers : {array-like} of shape (n_clusters, n_features)
        
        Returns
        --------
        closest_ind : int
            Index of closest centroid.
        """
        closest_i = 0
        closest_dist = float('inf')
        
        for i, cluster_center in enumerate(cluster_centers):
            # calculate distance
            distance_i = np.linalg.norm(sample - cluster_center)
            
            # Check for the minimal distance
            if distance_i < closest_dist:
                closest_dist = distance_i
                closest_i = i
            
        return closest_i
                
        
    @abstractmethod
    def _initialize(self, X, resp):
        """Initialize the model paramters of the derived class.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        
        resp : array-like of shape (n_samples, n_components)
        """
        pass
    
    def fit(self, X):
        """Estimate model parameter with the EM algorithm.
        
        The method fits the model ``n_init`` times and sets the parameters with which the model has the largest
        likelihood or lower bound. Within each trial, the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than ``tol``, otherwise, a ``ConvergenceWarning``
        is raised. If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single initialization is performed upon
        the first call. Upon consecutive calls, training starts where it left off.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features dimensional data points. Each row corresponds to single data points.
            
        Returns
        -------
        self : object
            The fitted mixture.
        """
        # Parameter are validated in fitted_object
        self.fit_predict(X)
        return self
    
    # @_fit_context(prefer_skip_nested_validation=True)
    def fit_predict(self, X):
        """Estimate model parameters using X and predict the labels for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.
        
        Returns
        ---------
        labels : array, shape (n_samples,)
            Component labes.
        """
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
            
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1
        
        max_lower_bound = -np.inf
        self.converged_ = False
    
        random_state = self.random_state
        
        n_samples, _ = X.shape
        
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)
            
            if do_init:
                self._initialize_parameters(X, random_state)
                
            lower_bound = -np.inf if do_init else self.lower_bound_
            
            if self.max_iter == 0:
                best_params = self._get_parameter()
                best_n_iter = 0
            
            else:
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound
                    
                    log_prob_norm, log_resp = self._e_step(X)
                    self._m_step(X, log_resp)
                    lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)
                    
                    change = lower_bound - prev_lower_bound
                    self._print_verbose_msg_iter_end(n_iter, change)
                    
                    if abs(change) < self.tol:
                        self.converged_ = True
                        break
                    
                self._print_verbose_msg_init_beg(lower_bound)
                
                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameter()
                    best_n_iter = n_iter
                    
            if not self.converged_ and self.max_iter > 0:
                warnings.warn(
                    "initialization % d did not converge. "
                    "Try different init parameters, "
                    "or increase max_iter, tol "
                    "or check for degenerate data. " % (init+1)
                )
            
            self._set_parameter(best_params)
            self.n_iter = best_n_iter
            self.lower_bound_ = max_lower_bound
            
            # Always do final e-step to guarantee that the labels returned by fit_predict(X) are always
            # consistent with fit(X).predict(X) for any value of max_iter and tol (and any random state)
            
            _, log_resp = self._e_step(X)
            
            return log_resp.argmax(axis=1)
            
            
    def _print_verbose_msg_init_beg(self, n_init):
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization %d" % n_init)
        elif self.verbose >=2:
            print("initialization %d" % n_init)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time
            
    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("  Iteration %d" % n_iter)
            elif self.verbose >= 2:
                cur_time = time()
                print(
                    "  Iteration %d\t time lapse %.5fs\t ll change %.5f"
                    % (n_iter, cur_time - self._iter_prev_time, diff_ll)
                )
                self._iter_prev_time = cur_time

    def _print_verbose_msg_init_end(self, ll):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Initialization converged: %s" % self.converged_)
        elif self.verbose >= 2:
            print(
                "Initialization converged: %s\t time lapse %.5fs\t ll %.5f"
                % (self.converged_, time() - self._init_prev_time, ll)
            )
     
    def _e_step(self, X):
        """E-Step.
        
        Parameters
        -----------
        X : array-like of shape (n_samples, n_features)
        
        Returns
        ---------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X
            
        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of posterior probabilities (or responsbilities) of the point of each sample in X.
        """ 
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp
    
    @abstractmethod
    def _m_step(self, X, log_resp):
        """M-step.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        
        log_resp : array-like of shape (n_samples, n_features)
            Logarithm of the posterior probabilities (or responsibilities) of the point of each sample in X.
        """
        pass
           
    @abstractmethod
    def _get_parameter(self):
        pass
    
    @abstractmethod
    def _set_parameter(self):
        pass
    
    def score_samples(self, X):
        """Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample in `X` under the current model.
        """

        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)

    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of `X` under the Gaussian mixture model.
        """
        return self.score_samples(X).mean()
    
    @abstractmethod
    def _estimate_log_weights(self):
        """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.

        Returns
        -------
        log_weight : array, shape (n_components, )
        """
        pass
    
    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights()
    
    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp
    
    def predict(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        
        return self._estimate_weighted_log_prob(X).argmax(axis=1)
    
    def predict_proba(self, X):
        """Evaluate the components' density for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Density of each Gaussian component for each sample in X.
        """
        
        _, log_resp = self._estimate_log_prob_resp(X)
        return np.exp(log_resp)
    
    def score_samples(self, X):
        """Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample in `X` under the current model.
        """
        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)
    
    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of `X` under the Gaussian mixture model.
        """
        
        return self.score_samples(X).mean()
    
    
     