#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dirichlet Multinomial Mixture with Variational Variable Selection
Following Dang et al. (2022) - Variational Inference Implementation

This implements the COORDINATE ASCENT variational inference algorithm
(without stochastic optimization - that comes later)

Key components:
1. Stick-breaking representation for infinite mixture
2. Per-sample feature selection (φ_ij)
3. Taylor expansion for intractable expectations
4. Coordinate ascent updates for all variational parameters

Author: Implementation following paper exactly
"""

import numpy as np
from scipy.special import digamma, gammaln, logsumexp, betaln
from scipy.spatial.distance import cosine
from sklearn.utils import check_array, check_random_state
from sklearn import cluster
from time import time


class NumericalStability:
    """Numerical stability constants"""
    EPS = 1e-10
    MAX_EXP = 500
    MIN_GAMMA = 1e-8
    MAX_GAMMA = 1e8

    @staticmethod
    def safe_log(x):
        return np.log(np.maximum(x, NumericalStability.EPS))

    @staticmethod
    def safe_exp(x):
        return np.exp(np.clip(x, -NumericalStability.MAX_EXP, NumericalStability.MAX_EXP))


class DMM_SVVS_Variational:
    """
    Dirichlet Multinomial Mixture with Variational Variable Selection

    Implements exact variational inference with coordinate ascent
    Following Dang et al. (2022) supplementary material

    Parameters
    ----------
    K_max : int
        Maximum number of clusters (truncation level)
    nu : float
        Concentration parameter for stick-breaking (Dirichlet process)
    zeta : float
        Prior concentration for alpha (cluster-specific parameters)
    eta : float
        Prior concentration for beta (common background parameters)
    xi_1, xi_2 : float
        Beta prior parameters for feature selection probabilities
    selection_prior : float
        Prior probability that a feature is selected (ε_j1)
    tol : float
        Convergence tolerance for ELBO
    max_iter : int
        Maximum iterations
    verbose : int
        Verbosity level
    random_state : int
        Random seed
    """

    def __init__(self,
                 K_max=10,
                 nu=0.1,
                 zeta=1.0,
                 eta=1.0,
                 xi_1=0.1,
                 xi_2=0.1,
                 selection_prior=0.3,
                 tol=1e-3,
                 max_iter=1000,
                 prune_threshold=0.02,
                 verbose=1,
                 random_state=42):

        # Model hyperparameters (from paper)
        self.K_max = K_max
        self.nu = nu
        self.zeta = zeta
        self.eta = eta
        self.xi_1 = xi_1
        self.xi_2 = xi_2
        self.selection_prior = selection_prior

        # Algorithm parameters
        self.tol = tol
        self.max_iter = max_iter
        self.prune_threshold = prune_threshold
        self.verbose = verbose
        self.random_state = random_state

        # To be initialized during fit
        self.N = None  # Number of samples
        self.S = None  # Number of features (OTUs)
        self.K = None  # Current number of clusters (may be < K_max after pruning)

        # Variational parameters (to be initialized)
        # Local parameters
        self.r = None      # (N, K) - cluster responsibilities
        self.f = None      # (N, S) - per-sample feature selection probabilities

        # Global parameters
        self.theta = None       # (K,) - stick-breaking Beta parameters (first)
        self.theta_prime = None # (K,) - stick-breaking Beta parameters (second)
        self.xi_star = None     # (S, 2) - selection prior parameters
        self.lambda_star = None # (K, S) - Dirichlet parameters for alpha
        self.iota_star = None   # (S,) - Dirichlet parameters for beta

        # Tracking
        self.elbo_history = []
        self.converged = False
        self.n_iter = 0

        # Cached expectations
        self._cache = {}

    def _initialize_parameters(self, X, random_state):
        """
        Initialize all variational parameters

        Strategy:
        1. Use k-means for initial cluster assignments
        2. Initialize alpha from cluster statistics
        3. Initialize beta from overall statistics
        4. Use prior values for stick-breaking and selection
        """
        self.N, self.S = X.shape
        self.K = self.K_max

        if self.verbose >= 1:
            print(f"Initializing parameters: N={self.N}, S={self.S}, K_max={self.K_max}")
            print(f"Hyperparameters: ν={self.nu}, ζ={self.zeta}, η={self.eta}, "
                  f"ξ=({self.xi_1}, {self.xi_2})")

        # ========================================
        # Local parameters
        # ========================================

        # Initialize cluster responsibilities from k-means
        self.r = self._initialize_responsibilities_kmeans(X, random_state)

        # Initialize feature selection: start with uniform
        # f_ij = P(φ_ij = 1) - probability that feature j is selected for sample i
        self.f = np.ones((self.N, self.S)) * self.selection_prior

        # ========================================
        # Global parameters
        # ========================================

        # Stick-breaking parameters
        # Initialize from prior
        self.theta = np.ones(self.K)
        self.theta_prime = np.ones(self.K) * self.nu

        # Selection prior parameters
        # ε_j ~ Beta(ξ*_j1, ξ*_j2)
        self.xi_star = np.ones((self.S, 2))
        self.xi_star[:, 0] = self.xi_1
        self.xi_star[:, 1] = self.xi_2

        # Dirichlet parameters for cluster-specific multinomial (alpha_k)
        # Initialize from cluster statistics
        self.lambda_star = np.zeros((self.K, self.S))
        for k in range(self.K):
            # Get samples assigned to cluster k
            cluster_mask = self.r[:, k] > 0.1
            if cluster_mask.sum() > 0:
                # Mean counts for this cluster
                cluster_mean = X[cluster_mask].mean(axis=0) + 0.5
                # Normalize to probability
                cluster_prob = cluster_mean / cluster_mean.sum()
                # Initialize with moderate concentration
                self.lambda_star[k] = cluster_prob * (self.zeta * self.S)
            else:
                # Fallback: use overall mean
                self.lambda_star[k] = self.zeta

        # Ensure positive
        self.lambda_star = np.maximum(self.lambda_star, 0.1)

        # Dirichlet parameters for background multinomial (beta)
        # Initialize from overall statistics
        overall_mean = X.mean(axis=0) + 0.5
        overall_prob = overall_mean / overall_mean.sum()
        self.iota_star = overall_prob * (self.eta * self.S)
        self.iota_star = np.maximum(self.iota_star, 0.1)

        # Clear cache
        self._cache = {}

    def _initialize_responsibilities_kmeans(self, X, random_state):
        """
        Initialize cluster responsibilities using k-means

        Returns
        -------
        r : array (N, K)
            Initial hard assignments (one-hot encoded)
        """
        N = X.shape[0]

        if self.K == 1:
            return np.ones((N, 1))

        try:
            # Log-transform for k-means (better for count data)
            X_log = np.log1p(X)

            # Run k-means
            kmeans = cluster.KMeans(
                n_clusters=min(self.K, N),
                n_init=10,
                max_iter=100,
                random_state=random_state
            )
            labels = kmeans.fit_predict(X_log)

            # Vectorized one-hot encoding
            r = np.zeros((N, self.K))
            r[np.arange(N), labels] = 1.0

        except Exception as e:
            if self.verbose >= 1:
                print(f"  K-means failed ({e}), using random initialization")
            # Random soft assignment
            r = random_state.rand(N, self.K)
            r /= r.sum(axis=1, keepdims=True)

        return r

    # ========================================
    # Expectation computations (cached)
    # ========================================

    def _compute_E_alpha(self, k):
        """
        Compute E[α_k] under q(α_k | λ*_k)

        For Dirichlet(λ*_k), E[α_kj] = λ*_kj / sum(λ*_k)
        """
        key = f'E_alpha_{k}'
        if key not in self._cache:
            lambda_sum = self.lambda_star[k].sum()
            self._cache[key] = self.lambda_star[k] / lambda_sum
        return self._cache[key]

    def _compute_E_log_alpha(self, k):
        """
        Compute E[log α_kj] under q(α_k | λ*_k)

        For Dirichlet(λ*_k), E[log α_kj] = ψ(λ*_kj) - ψ(sum(λ*_k))
        """
        key = f'E_log_alpha_{k}'
        if key not in self._cache:
            lambda_sum = self.lambda_star[k].sum()
            self._cache[key] = digamma(self.lambda_star[k]) - digamma(lambda_sum)
        return self._cache[key]

    def _compute_E_beta(self):
        """Compute E[β] under q(β | ι*)"""
        if 'E_beta' not in self._cache:
            iota_sum = self.iota_star.sum()
            self._cache['E_beta'] = self.iota_star / iota_sum
        return self._cache['E_beta']

    def _compute_E_log_beta(self):
        """Compute E[log β_j] under q(β | ι*)"""
        if 'E_log_beta' not in self._cache:
            iota_sum = self.iota_star.sum()
            self._cache['E_log_beta'] = digamma(self.iota_star) - digamma(iota_sum)
        return self._cache['E_log_beta']

    def _compute_E_log_gamma(self, k):
        """
        Compute E[log γ_k] under q(γ_k | θ_k, θ'_k)

        For Beta(θ_k, θ'_k), E[log γ_k] = ψ(θ_k) - ψ(θ_k + θ'_k)
        """
        return digamma(self.theta[k]) - digamma(self.theta[k] + self.theta_prime[k])

    def _compute_E_log_1minus_gamma(self, k):
        """
        Compute E[log(1 - γ_k)] under q(γ_k | θ_k, θ'_k)

        For Beta(θ_k, θ'_k), E[log(1-γ_k)] = ψ(θ'_k) - ψ(θ_k + θ'_k)
        """
        return digamma(self.theta_prime[k]) - digamma(self.theta[k] + self.theta_prime[k])

    def _compute_E_log_pi(self):
        """
        Compute E[log π_k] for all k using stick-breaking.

        log π_k = log γ_k + sum_{k'<k} log(1 - γ_k')

        Vectorized with cumsum: O(K) instead of O(K²).
        """
        if 'E_log_pi' not in self._cache:
            sum_theta = self.theta + self.theta_prime
            E_log_gamma  = digamma(self.theta)       - digamma(sum_theta)  # (K,)
            E_log_1mgamma = digamma(self.theta_prime) - digamma(sum_theta)  # (K,)
            # cumsum shifted by 1: prefix sum of log(1-γ_{k'}) for k' < k
            cum = np.concatenate([[0.0], np.cumsum(E_log_1mgamma[:-1])])
            self._cache['E_log_pi'] = E_log_gamma + cum
        return self._cache['E_log_pi']

    def _clear_cache(self):
        """Clear expectation cache (call after parameter updates)"""
        self._cache = {}

    # ========================================
    # Coordinate ascent updates
    # ========================================

    def _update_r(self, X):
        """
        Update cluster responsibilities r_ik.

        log r_ik ∝ E[log π_k] + sum_j f_ij * log_likelihood_k(X_ij)

        Fully vectorized over N, K, S.
        """
        EPS = NumericalStability.EPS
        E_log_pi = self._compute_E_log_pi()                        # (K,)

        # Precompute all cluster expectations at once — (K, S)
        alpha    = np.array([self._compute_E_alpha(k)     for k in range(self.K)])  # (K, S)
        E_log_alpha = np.array([self._compute_E_log_alpha(k) for k in range(self.K)])  # (K, S)
        beta     = self._compute_E_beta()                          # (S,)
        E_log_beta = self._compute_E_log_beta()                    # (S,)

        # Expand dims for broadcasting: X (N,1,S), alpha (1,K,S)
        X3       = X[:, None, :]                                   # (N, 1, S)
        alpha3   = alpha[None, :, :]                               # (1, K, S)
        Ela3     = E_log_alpha[None, :, :]                         # (1, K, S)

        # --- Selected-feature log-likelihood (Taylor expansion) ---
        # Base term: log Γ(X+α) - log Γ(α)
        log_sel  = gammaln(X3 + alpha3) - gammaln(alpha3)          # (N, K, S)
        psi_sel  = digamma(alpha3 + X3) - digamma(alpha3)          # (N, K, S)
        corr_sel = Ela3 - np.log(alpha3 + EPS)                     # (1, K, S)
        log_sel += alpha3 * psi_sel * corr_sel                     # (N, K, S)

        # --- Unselected-feature log-likelihood (Taylor expansion) ---
        beta2    = beta[None, None, :]                             # (1, 1, S)
        Elb2     = E_log_beta[None, None, :]                       # (1, 1, S)
        log_unsel = gammaln(X3 + beta2) - gammaln(beta2)           # (N, 1, S)
        psi_unsel = digamma(beta2 + X3) - digamma(beta2)           # (N, 1, S)
        corr_unsel = Elb2 - np.log(beta2 + EPS)                    # (1, 1, S)
        log_unsel += beta2 * psi_unsel * corr_unsel                # (N, 1, S)

        # Weighted sum over S (feature dimension)
        f3       = self.f[:, None, :]                              # (N, 1, S)
        log_like = (f3 * log_sel + (1 - f3) * log_unsel).sum(axis=2)  # (N, K)

        # --- Normalizing constant terms ---
        alpha_sum = alpha.sum(axis=1)                              # (K,)
        X_i_sum   = X.sum(axis=1)                                  # (N,)

        log_norm_sel   = (gammaln(alpha_sum[None, :])
                          - gammaln(X_i_sum[:, None] + alpha_sum[None, :]))  # (N, K)

        beta_sum       = beta.sum()
        log_norm_unsel = gammaln(beta_sum) - gammaln(X_i_sum + beta_sum)     # (N,)

        f_i_avg  = self.f.mean(axis=1)                             # (N,)
        log_norm = (f_i_avg[:, None] * log_norm_sel
                    + (1 - f_i_avg[:, None]) * log_norm_unsel[:, None])      # (N, K)

        log_r = E_log_pi[None, :] + log_like + log_norm           # (N, K)

        # Normalize in log-space
        log_r -= logsumexp(log_r, axis=1, keepdims=True)
        self.r = np.exp(log_r)

        # Numerical stability
        self.r = np.maximum(self.r, EPS)
        self.r /= self.r.sum(axis=1, keepdims=True)

    def _update_f(self, X):
        """
        Update feature selection probabilities f_ij = P(φ_ij = 1 | ...).

        Fully vectorized over N, K, S.
        """
        EPS = NumericalStability.EPS

        xi_sum  = self.xi_star.sum(axis=1)                         # (S,)
        E_xi_1  = digamma(self.xi_star[:, 0]) - digamma(xi_sum)   # (S,)
        E_xi_2  = digamma(self.xi_star[:, 1]) - digamma(xi_sum)   # (S,)

        # Precompute cluster expectations — (K, S)
        alpha       = np.array([self._compute_E_alpha(k)     for k in range(self.K)])
        E_log_alpha = np.array([self._compute_E_log_alpha(k) for k in range(self.K)])
        beta        = self._compute_E_beta()                       # (S,)
        E_log_beta  = self._compute_E_log_beta()                   # (S,)

        # --- Selected: log-likelihood for each (i, k, j) ---
        X3      = X[:, None, :]                                    # (N, 1, S)
        alpha3  = alpha[None, :, :]                                # (1, K, S)
        Ela3    = E_log_alpha[None, :, :]                          # (1, K, S)

        log_sel  = gammaln(X3 + alpha3) - gammaln(alpha3)          # (N, K, S)
        psi_sel  = digamma(alpha3 + X3) - digamma(alpha3)          # (N, K, S)
        log_sel += alpha3 * (psi_sel * (Ela3 - np.log(alpha3 + EPS)))  # (N, K, S)

        # Sum over k weighted by r: result (N, S)
        # r: (N, K) → einsum 'ik,iks->is'
        r_sum_sel = np.einsum('ik,iks->is', self.r, log_sel)       # (N, S)

        # --- Unselected: log-likelihood (N, S), no k-dependence ---
        # sum_k r_ik * log_unsel_ij = log_unsel_ij  (since sum_k r_ik = 1)
        beta2     = beta[None, :]                                  # (1, S)
        Elb2      = E_log_beta[None, :]                            # (1, S)
        log_unsel = gammaln(X + beta2) - gammaln(beta2)            # (N, S)
        psi_unsel = digamma(beta2 + X) - digamma(beta2)            # (N, S)
        log_unsel += beta2 * (psi_unsel * (Elb2 - np.log(beta2 + EPS)))  # (N, S)

        log_prob_sel   = E_xi_1[None, :] + r_sum_sel              # (N, S)
        log_prob_unsel = E_xi_2[None, :] + log_unsel              # (N, S)

        # Numerically stable sigmoid
        max_log       = np.maximum(log_prob_sel, log_prob_unsel)
        prob_sel      = np.exp(log_prob_sel   - max_log)
        prob_unsel    = np.exp(log_prob_unsel - max_log)
        self.f        = prob_sel / (prob_sel + prob_unsel + EPS)   # (N, S)

        self.f = np.clip(self.f, EPS, 1 - EPS)

    def _update_theta(self):
        """
        Update stick-breaking parameters θ_k and θ'_k.

        θ_k  = 1 + sum_i r_ik
        θ'_k = ν + sum_{k'>k} sum_i r_ik'

        Vectorized with cumsum: O(K) instead of O(K²).
        """
        r_k = self.r.sum(axis=0)                                   # (K,)
        self.theta = 1.0 + r_k
        # theta'[k] = nu + sum_{k'>k} r_k' = nu + (total - cumsum[k])
        cumsum_r = np.cumsum(r_k)                                  # (K,)
        self.theta_prime = self.nu + (cumsum_r[-1] - cumsum_r)
        self.theta_prime[-1] = self.nu                             # last cluster

        self.theta       = np.maximum(self.theta,       NumericalStability.EPS)
        self.theta_prime = np.maximum(self.theta_prime, NumericalStability.EPS)

    def _update_xi_star(self):
        """
        Update selection prior parameters ξ*

        ξ*_j1 = ξ_1 + sum_i f_ij
        ξ*_j2 = ξ_2 + sum_i (1 - f_ij)
        """
        self.xi_star[:, 0] = self.xi_1 + self.f.sum(axis=0)
        self.xi_star[:, 1] = self.xi_2 + (1 - self.f).sum(axis=0)

        # Ensure positive
        self.xi_star = np.maximum(self.xi_star, NumericalStability.EPS)

    def _update_lambda_star(self, X):
        """
        Update Dirichlet parameters λ*_k for cluster-specific alpha.

        Vectorized over N and S for each k.
        """
        X_i_sum = X.sum(axis=1)                                    # (N,) — hoisted outside k-loop

        for k in range(self.K):
            alpha_k   = self._compute_E_alpha(k)                   # (S,)
            alpha_sum = alpha_k.sum()                              # scalar — hoisted outside N,S loops

            # grad_matrix[i, j] = ψ(α_sum) - ψ(X_sum_i + α_sum) + ψ(α_kj + X_ij) - ψ(α_kj)
            grad_matrix = (
                digamma(alpha_sum)
                - digamma(X_i_sum[:, None] + alpha_sum)            # (N, 1) broadcast → (N, S)
                + digamma(alpha_k[None, :] + X)                    # (N, S)
                - digamma(alpha_k[None, :])                        # (1, S)
            )                                                      # (N, S)

            # gradient[j] = sum_i r_ik * f_ij * α_kj * grad[i,j]
            # = α_kj * (r[:, k:k+1] * f * grad_matrix).sum(axis=0)
            weighted = self.r[:, k:k+1] * self.f * grad_matrix    # (N, S)
            gradient = alpha_k * weighted.sum(axis=0)              # (S,)

            self.lambda_star[k] = np.maximum(self.zeta + gradient, 0.1)

    def _update_iota_star(self, X):
        """
        Update Dirichlet parameters ι* for background beta.

        Vectorized over N and S.
        """
        beta     = self._compute_E_beta()                          # (S,)
        beta_sum = beta.sum()                                      # scalar — hoisted outside loops
        X_i_sum  = X.sum(axis=1)                                   # (N,) — hoisted outside loops

        # grad_matrix[i, j] = ψ(β_sum) - ψ(X_sum_i + β_sum) + ψ(β_j + X_ij) - ψ(β_j)
        grad_matrix = (
            digamma(beta_sum)
            - digamma(X_i_sum[:, None] + beta_sum)                 # (N, 1) → (N, S)
            + digamma(beta[None, :] + X)                           # (N, S)
            - digamma(beta[None, :])                               # (1, S)
        )                                                          # (N, S)

        # gradient[j] = sum_i (1 - f_ij) * β_j * grad[i,j]
        gradient = beta * ((1 - self.f) * grad_matrix).sum(axis=0)  # (S,)

        self.iota_star = np.maximum(self.eta + gradient, 0.1)

    # ========================================
    # ELBO computation
    # ========================================

    def _compute_elbo(self, X):
        """
        Compute Evidence Lower Bound (ELBO).

        Fully vectorized over N, K, S.
        """
        EPS = NumericalStability.EPS
        E_log_pi = self._compute_E_log_pi()                        # (K,)

        alpha = np.array([self._compute_E_alpha(k) for k in range(self.K)])  # (K, S)
        beta  = self._compute_E_beta()                             # (S,)

        # Log-likelihoods — (N, K, S)
        X3       = X[:, None, :]                                   # (N, 1, S)
        alpha3   = alpha[None, :, :]                               # (1, K, S)
        beta3    = beta[None, None, :]                             # (1, 1, S)

        log_like_sel   = gammaln(X3 + alpha3) - gammaln(alpha3)   # (N, K, S)
        log_like_unsel = gammaln(X3 + beta3)  - gammaln(beta3)    # (N, 1, S)

        f3       = self.f[:, None, :]                              # (N, 1, S)
        log_like = (f3 * log_like_sel + (1 - f3) * log_like_unsel).sum(axis=2)  # (N, K)

        # E[log π_k] + log-likelihood, weighted by r
        elbo = np.sum(self.r * (E_log_pi[None, :] + log_like))

        # Entropy of q(Z)
        elbo -= np.sum(self.r * np.log(np.maximum(self.r, EPS)))

        return elbo

    # ========================================
    # Pruning and model selection
    # ========================================

    def _prune_empty_clusters(self):
        """
        Remove clusters with very low weight.

        Stick-breaking weights computed with cumprod: O(K) instead of O(K²).
        Returns True if any cluster was removed.
        """
        E_gamma = self.theta / (self.theta + self.theta_prime)     # (K,)
        # π_k = γ_k * prod_{k'<k} (1 - γ_{k'})
        log_weights = np.log(np.maximum(E_gamma, NumericalStability.EPS))
        log_1mg     = np.log(np.maximum(1 - E_gamma, NumericalStability.EPS))
        log_weights += np.concatenate([[0.0], np.cumsum(log_1mg[:-1])])
        weights = np.exp(log_weights)

        # Find clusters to keep
        keep = weights > self.prune_threshold

        # Also keep clusters with significant samples
        cluster_sizes = self.r.sum(axis=0)
        keep = keep | (cluster_sizes > 1.0)

        # Keep at least 1 cluster
        if keep.sum() < 1:
            keep[np.argmax(weights)] = True

        n_keep = keep.sum()

        if n_keep < self.K:
            if self.verbose >= 1:
                print(f"  Pruning: {self.K} → {n_keep} clusters")

            # Update all cluster-dependent parameters
            self.K = n_keep
            self.r = self.r[:, keep]
            self.theta = self.theta[keep]
            self.theta_prime = self.theta_prime[keep]
            self.lambda_star = self.lambda_star[keep]

            # Renormalize responsibilities
            self.r /= self.r.sum(axis=1, keepdims=True)

            self._clear_cache()
            return True

        return False

    # ========================================
    # Main fitting algorithm
    # ========================================

    def fit(self, X):
        """
        Fit the model using coordinate ascent variational inference

        Parameters
        ----------
        X : array-like, shape (N, S)
            Count data matrix (samples × features)

        Returns
        -------
        self
        """
        X = check_array(X, dtype=np.float64)
        random_state = check_random_state(self.random_state)

        # Initialize
        self._initialize_parameters(X, random_state)

        if self.verbose >= 1:
            print(f"\nStarting coordinate ascent variational inference")
            print(f"=" * 70)

        start_time = time()

        # Coordinate ascent loop
        for iteration in range(1, self.max_iter + 1):
            self.n_iter = iteration

            # Clear expectation cache
            self._clear_cache()

            # ========================================
            # Coordinate ascent updates
            # ========================================

            # Update local parameters
            self._update_r(X)
            self._update_f(X)

            # Update global parameters
            self._update_theta()
            self._update_xi_star()
            self._update_lambda_star(X)
            self._update_iota_star(X)

            # Prune empty clusters (not too frequently)
            if iteration > 50 and iteration % 20 == 0:
                self._prune_empty_clusters()

            # Compute ELBO and check convergence
            if iteration % 10 == 0:
                elbo = self._compute_elbo(X)
                self.elbo_history.append(elbo)

                if self.verbose >= 1:
                    elapsed = time() - start_time
                    print(f"Iter {iteration:4d}: ELBO = {elbo:12.2f}, K = {self.K}, "
                          f"Time = {elapsed:.1f}s")

                # Check convergence
                if len(self.elbo_history) >= 3:
                    recent_elbos = self.elbo_history[-3:]
                    elbo_changes = [abs(recent_elbos[i] - recent_elbos[i-1]) /
                                   (abs(recent_elbos[i]) + 1e-10)
                                   for i in range(1, len(recent_elbos))]

                    if all(change < self.tol for change in elbo_changes):
                        self.converged = True
                        if self.verbose >= 1:
                            print(f"\n✓ Converged at iteration {iteration}")
                        break

        # Final pruning
        self._prune_empty_clusters()

        # Compute final mixing weights
        self.weights_ = self._compute_mixing_weights()

        if self.verbose >= 1:
            total_time = time() - start_time
            print(f"\nFinal results:")
            print(f"  Number of clusters: {self.K}")
            print(f"  Mixing weights: {self.weights_}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Converged: {self.converged}")

        return self

    def _compute_mixing_weights(self):
        """
        Compute mixing weights π_k from stick-breaking.

        Vectorized with log-cumprod: O(K) instead of O(K²).
        """
        E_gamma = self.theta / (self.theta + self.theta_prime)     # (K,)
        log_weights = np.log(np.maximum(E_gamma, NumericalStability.EPS))
        log_1mg     = np.log(np.maximum(1 - E_gamma, NumericalStability.EPS))
        log_weights += np.concatenate([[0.0], np.cumsum(log_1mg[:-1])])
        return np.exp(log_weights)

    def predict(self, X):
        """
        Predict cluster assignments for new data

        Parameters
        ----------
        X : array-like, shape (N_new, S)
            New count data

        Returns
        -------
        labels : array, shape (N_new,)
            Cluster assignments
        """
        X = check_array(X, dtype=np.float64)

        # Run E-step to get responsibilities
        N_new = X.shape[0]

        # Use current f as prior for new samples (could be improved)
        f_new = np.ones((N_new, self.S)) * self.f.mean(axis=0, keepdims=True)

        # Store original
        N_orig, f_orig = self.N, self.f

        # Temporarily set to new data
        self.N = N_new
        self.f = f_new
        self.r = np.zeros((N_new, self.K))

        # Run one E-step
        self._clear_cache()
        self._update_r(X)

        # Get predictions
        labels = self.r.argmax(axis=1)

        # Restore original
        self.N, self.f = N_orig, f_orig

        return labels

    def get_selected_features(self, threshold=0.5):
        """
        Get features selected for each sample

        Parameters
        ----------
        threshold : float
            Probability threshold for selection

        Returns
        -------
        selected : dict
            Dictionary mapping sample indices to selected feature indices
        """
        selected = {}
        for i in range(self.N):
            idx = np.where(self.f[i] > threshold)[0]
            if len(idx) > 0:
                selected[i] = {
                    'indices': idx.tolist(),
                    'probabilities': self.f[i, idx].tolist()
                }
        return selected

    def get_cluster_signatures(self, threshold=0.5):
        """
        Get important features for each cluster

        Parameters
        ----------
        threshold : float
            Selection probability threshold

        Returns
        -------
        signatures : dict
            Dictionary mapping cluster indices to feature info
        """
        signatures = {}

        for k in range(self.K):
            # Get samples in this cluster
            cluster_samples = self.r[:, k] > 0.5

            if cluster_samples.sum() > 0:
                # Average selection probability for samples in this cluster
                avg_selection = self.f[cluster_samples].mean(axis=0)

                # Get highly selected features
                important = np.where(avg_selection > threshold)[0]

                if len(important) > 0:
                    # Get alpha values for these features
                    alpha_k = self._compute_E_alpha(k)

                    signatures[k] = {
                        'features': important.tolist(),
                        'selection_probs': avg_selection[important].tolist(),
                        'alpha_values': alpha_k[important].tolist(),
                        'n_samples': cluster_samples.sum()
                    }

        return signatures


# ========================================
# Testing
# ========================================

if __name__ == "__main__":
    print("Testing DMM-SVVS Variational Inference")
    print("=" * 70)

    np.random.seed(42)

    # Generate synthetic data
    N, S, K_true = 100, 50, 3

    X = np.zeros((N, S))
    true_labels = np.random.choice(K_true, size=N)

    # Generate cluster-specific distributions
    for k in range(K_true):
        # Cluster-specific probabilities
        probs = np.random.gamma(0.5, 0.5, size=S)
        probs /= probs.sum()

        # Generate counts for samples in this cluster
        mask = true_labels == k
        for i in np.where(mask)[0]:
            X[i] = np.random.multinomial(500, probs)

    print(f"Generated data: N={N}, S={S}, K_true={K_true}")
    print(f"Cluster sizes: {np.bincount(true_labels)}\n")

    # Fit model
    model = DMM_SVVS_Variational(
        K_max=10,
        nu=0.1,
        zeta=1.0,
        eta=1.0,
        max_iter=200,
        verbose=1,
        random_state=42
    )

    model.fit(X)

    # Evaluate
    from sklearn.metrics import adjusted_rand_score
    pred_labels = model.predict(X)
    ari = adjusted_rand_score(true_labels, pred_labels)

    print(f"\n{'=' * 70}")
    print(f"Results:")
    print(f"  True K: {K_true}")
    print(f"  Estimated K: {model.K}")
    print(f"  ARI: {ari:.3f}")
    print(f"  Final weights: {model.weights_}")

    # Show cluster signatures
    print(f"\nCluster signatures:")
    signatures = model.get_cluster_signatures(threshold=0.6)
    for k, sig in signatures.items():
        print(f"  Cluster {k}: {len(sig['features'])} important features, "
              f"{sig['n_samples']} samples")
