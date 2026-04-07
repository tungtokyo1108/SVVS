#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS Variational Inference — Version 2.1
==============================================

Key change over v2: per-cluster feature selection  f ∈ [0,1]^{K×S}
------------------------------------------------------------------------

In v2, f ∈ [0,1]^{N×S} assigns a selection probability to every
(sample, feature) pair.  This is theoretically incorrect: whether a
feature is discriminative should be a property of the *cluster*, not
of individual samples.  It also introduces O(NS) noise parameters.

In v2.1 we use f ∈ [0,1]^{K×S}, so f_kj is the probability that
feature j is informative for cluster k.  This means:

  - Each cluster can select a *different* subset of features.
  - Features that discriminate cluster k from the background get
    f_kj → 1 for that cluster; features irrelevant to k get f_kj → 0.
  - The responsibility update uses f[k,:] when evaluating cluster k,
    so clusters are pulled toward samples that match their *own*
    selected feature profile — this is the mechanism that can improve ARI.

Mathematical changes
--------------------
  f      : (K, S)  [was (N, S)]
  xi_star: (K, S, 2) Beta posterior params  [was (S, 2)]

  Expected log-likelihood (N, K):
    ll[i,k] = Σ_j f[k,j] · x_ij · E[log α_kj]
            + Σ_j (1−f[k,j]) · x_ij · E[log β_j]
    = X @ (f[k,:] * E_log_α[k,:] + (1−f[k,:]) * E_log_β).T
    Computed efficiently as:
      combined[k,j] = f[k,j]*E_log_α[k,j] + (1−f[k,j])*E_log_β[j]  (K,S)
      ll = X @ combined.T                                              (N,K)

  Lambda update (exact conjugate):
    λ_kj = ζ + Σ_i r_ik · f_kj · x_ij
         = ζ + f[k,:] * (r[:,k] @ X)   (element-wise, vectorized over k)

  Iota update:
    ι_j = η + Σ_k Σ_i r_ik · (1−f_kj) · x_ij
        = η + Σ_k (1−f[k,:]) * (r[:,k] @ X)
        = η + ((1 − f) * rX).sum(axis=0)   where rX = r.T @ X  (K,S)

  f update (per-cluster per-feature log-odds):
    For cluster k, feature j:
      log_odds_kj = E[log(ξ*_k1j / ξ*_k2j)]
                  + (r[:,k] @ X)_j · E[log α_kj]    (cluster gain)
                  − (r[:,k] @ X)_j · E[log β_j]     (background)
    Vectorized: rX = r.T @ X  (K,S)
      log_odds = E_xi1 + rX * E_log_α − rX * E_log_β   (K,S)

  xi_star update:
    ξ*_k1j = ξ_1 + N · f_kj
    ξ*_k2j = ξ_2 + N · (1 − f_kj)

All other components (stick-breaking, pruning, ELBO, convergence) are
identical to v2.

Author: v2.1 — per-cluster feature selection extension of v2
"""

import numpy as np
from scipy.special import digamma, logsumexp, expit
from sklearn.utils import check_array, check_random_state
from sklearn import cluster
from time import time


class NumericalStability:
    EPS     = 1e-10
    MAX_EXP = 500

    @staticmethod
    def safe_log(x):
        return np.log(np.maximum(x, NumericalStability.EPS))


class DMM_SVVS_Variational_v2_1:
    """
    Dirichlet Multinomial Mixture with per-Cluster Variational Variable
    Selection — v2.1

    Identical to v2 except that the feature selection matrix f has shape
    (K, S) instead of (N, S).  Each cluster maintains its own feature
    relevance profile, allowing the model to identify which OTUs/features
    specifically characterise each cluster.

    Parameters
    ----------
    K_max : int
        Maximum / initial number of clusters (truncation level).
    nu : float or 'auto'
        DP concentration parameter for stick-breaking.
        'auto' → nu = 1/K_max.
    zeta : float
        Prior concentration for cluster-specific Dirichlet (alpha_k).
    eta : float
        Prior concentration for background Dirichlet (beta).
    xi_1, xi_2 : float
        Beta(xi_1, xi_2) prior on per-cluster-per-feature selection f_kj.
        Default 1.0/1.0 → uniform (neutral) prior.
    selection_prior : float
        Initial warm-start value for f (all f_kj initialised to this).
    tol : float
        Relative ELBO convergence tolerance.
    max_iter : int
        Maximum CAVI iterations.
    prune_threshold : float
        Cluster is pruned if weight < prune_threshold AND size < min_cluster_size.
    min_clusters : int or None
        Hard floor on number of active clusters.  None → max(2, K_max // 5).
    prune_start : int
        Iteration at which pruning begins.
    prune_every : int
        Prune every this many iterations.
    verbose : int
    random_state : int
    """

    def __init__(self,
                 K_max=10,
                 nu='auto',
                 zeta=1.0,
                 eta=1.0,
                 xi_1=1.0,
                 xi_2=1.0,
                 selection_prior=0.3,
                 tol=1e-4,
                 max_iter=500,
                 prune_threshold=0.02,
                 min_clusters=None,
                 prune_start=10,
                 prune_every=5,
                 verbose=1,
                 random_state=42):

        self.K_max           = K_max
        self.nu_input        = nu
        self.zeta            = zeta
        self.eta             = eta
        self.xi_1            = xi_1
        self.xi_2            = xi_2
        self.selection_prior = selection_prior
        self.tol             = tol
        self.max_iter        = max_iter
        self.prune_threshold = prune_threshold
        self.min_clusters    = min_clusters
        self.prune_start     = prune_start
        self.prune_every     = prune_every
        self.verbose         = verbose
        self.random_state    = random_state

        # Runtime state
        self.N = self.S = self.K = None
        self.r = self.f = None            # f is (K, S) in v2.1
        self.theta = self.theta_prime = None
        self.xi_star = None               # (K, S, 2) in v2.1
        self.lambda_star = self.iota_star = None
        self.elbo_history = []
        self.converged = False
        self.n_iter = 0
        self._cache = {}
        self._pruned_at_least_once = False

    # ─────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────

    def _resolve_nu(self):
        if self.nu_input == 'auto':
            return 1.0 / self.K_max
        nu = float(self.nu_input)
        if nu > 2.0 and self.verbose >= 1:
            print(f"  [Warning] nu={nu:.1f} is large — stick-breaking will spread mass "
                  f"across all {self.K_max} clusters and resist pruning. "
                  f"Consider nu='auto' or nu<1 for real data.")
        return nu

    def _initialize_parameters(self, X, random_state):
        self.N, self.S = X.shape
        self.K = self.K_max
        self.nu = self._resolve_nu()

        self._min_clusters = (
            max(2, self.K_max // 5)
            if self.min_clusters is None
            else int(self.min_clusters)
        )
        self._min_cluster_size = max(1.0, self.N / (5.0 * self.K_max))

        if self.verbose >= 1:
            print(f"Initializing: N={self.N}, S={self.S}, K_max={self.K_max}")
            print(f"  ν={self.nu:.4f}, ζ={self.zeta}, η={self.eta}, "
                  f"ξ=({self.xi_1},{self.xi_2}), min_K={self._min_clusters}, "
                  f"min_cluster_size={self._min_cluster_size:.1f}")

        # Responsibilities from k-means
        self.r = self._init_responsibilities_kmeans(X, random_state)

        # Per-cluster feature selection warm-start — shape (K, S)
        self.f = np.full((self.K, self.S), self.selection_prior)

        # Stick-breaking: symmetric init
        self.theta       = np.ones(self.K)
        self.theta_prime = np.ones(self.K) * self.nu

        # Beta prior params: (K, S, 2)
        self.xi_star = np.empty((self.K, self.S, 2))
        self.xi_star[:, :, 0] = self.xi_1
        self.xi_star[:, :, 1] = self.xi_2

        # Cluster-specific Dirichlet params: init from k-means statistics
        self.lambda_star = np.zeros((self.K, self.S))
        for k in range(self.K):
            mask = self.r[:, k] > 0.1
            if mask.sum() > 0:
                mu = X[mask].mean(axis=0) + 0.5
                self.lambda_star[k] = mu / mu.sum() * (self.zeta * self.S)
            else:
                self.lambda_star[k] = self.zeta
        self.lambda_star = np.maximum(self.lambda_star, 0.1)

        # Background Dirichlet params
        mu_all = X.mean(axis=0) + 0.5
        self.iota_star = mu_all / mu_all.sum() * (self.eta * self.S)
        self.iota_star = np.maximum(self.iota_star, 0.1)

        self._cache = {}
        self._pruned_at_least_once = False

    def _init_responsibilities_kmeans(self, X, random_state):
        N = X.shape[0]
        if self.K == 1:
            return np.ones((N, 1))
        try:
            X_log  = np.log1p(X)
            kmeans = cluster.KMeans(
                n_clusters=min(self.K, N),
                n_init=10, max_iter=100,
                random_state=random_state
            )
            labels = kmeans.fit_predict(X_log)
            r = np.zeros((N, self.K))
            r[np.arange(N), labels] = 1.0
        except Exception as e:
            if self.verbose >= 1:
                print(f"  K-means failed ({e}), using random init")
            r = random_state.rand(N, self.K)
            r /= r.sum(axis=1, keepdims=True)
        return r

    # ─────────────────────────────────────────────────────────────────────
    # Cached expectations
    # ─────────────────────────────────────────────────────────────────────

    def _E_log_alpha(self):
        """(K, S): E[log α_kj] = ψ(λ_kj) − ψ(Σ_j λ_kj)"""
        if 'Ela' not in self._cache:
            s = self.lambda_star.sum(axis=1, keepdims=True)  # (K, 1)
            self._cache['Ela'] = digamma(self.lambda_star) - digamma(s)
        return self._cache['Ela']

    def _E_log_beta(self):
        """(S,): E[log β_j] = ψ(ι_j) − ψ(Σ_j ι_j)"""
        if 'Elb' not in self._cache:
            self._cache['Elb'] = digamma(self.iota_star) - digamma(self.iota_star.sum())
        return self._cache['Elb']

    def _E_log_pi(self):
        """(K,): E[log π_k] via vectorized cumsum."""
        if 'Elpi' not in self._cache:
            st   = self.theta + self.theta_prime
            elg  = digamma(self.theta)       - digamma(st)
            el1g = digamma(self.theta_prime) - digamma(st)
            cum  = np.concatenate([[0.0], np.cumsum(el1g[:-1])])
            self._cache['Elpi'] = elg + cum
        return self._cache['Elpi']

    def _clear_cache(self):
        self._cache = {}

    # ─────────────────────────────────────────────────────────────────────
    # Expected log-likelihood  — per-cluster f
    # ─────────────────────────────────────────────────────────────────────

    def _expected_log_lik(self, X):
        """
        (N, K) expected log-likelihood using per-cluster feature selection.

        For cluster k:
          ll[i,k] = Σ_j [ f[k,j]·x_ij·E[log α_kj]
                         + (1−f[k,j])·x_ij·E[log β_j] ]

        combined[k,j] = f[k,j]*E[log α_kj] + (1−f[k,j])*E[log β_j]  (K,S)
        ll = X @ combined.T                                             (N,K)
        """
        E_log_a  = self._E_log_alpha()   # (K, S)
        E_log_b  = self._E_log_beta()    # (S,)

        # combined: each cluster uses its own f[k,:] to blend cluster vs bg
        combined = (self.f * E_log_a
                    + (1.0 - self.f) * E_log_b[None, :])   # (K, S)
        return X @ combined.T                               # (N, K)

    # ─────────────────────────────────────────────────────────────────────
    # CAVI update steps
    # ─────────────────────────────────────────────────────────────────────

    def _update_r(self, X):
        EPS      = NumericalStability.EPS
        E_log_pi = self._E_log_pi()
        ll       = self._expected_log_lik(X)   # (N, K)

        log_r  = E_log_pi[None, :] + ll
        log_r -= logsumexp(log_r, axis=1, keepdims=True)
        self.r = np.exp(log_r)
        self.r = np.maximum(self.r, EPS)
        self.r /= self.r.sum(axis=1, keepdims=True)

    def _update_f(self, X):
        """
        Per-cluster per-feature selection update  (K, S).

        For cluster k, feature j, the CAVI log-odds is:
          log_odds[k,j] = E[log(ξ*_k1j / ξ*_k2j)]
                        + (r[:,k] @ X)_j · E[log α_kj]     (cluster gain)
                        − (r[:,k] @ X)_j · E[log β_j]      (background)

        rX[k,j] = Σ_i r_ik · x_ij  =  r.T @ X             (K, S)
        The log-odds compares how well cluster k's profile explains
        feature j's counts vs. the background — per cluster.
        """
        EPS = NumericalStability.EPS

        xi_sum = self.xi_star.sum(axis=2)                                 # (K, S)
        E_xi1  = digamma(self.xi_star[:, :, 0]) - digamma(xi_sum)        # (K, S)
        E_xi2  = digamma(self.xi_star[:, :, 1]) - digamma(xi_sum)        # (K, S)

        E_log_alpha = self._E_log_alpha()   # (K, S)
        E_log_beta  = self._E_log_beta()    # (S,)

        # r-weighted sufficient statistics per cluster: (K, S)
        rX = self.r.T @ X                  # (K, N) @ (N, S) = (K, S)

        # cluster gain vs background per (k, j)
        E_log_b  = E_log_beta[None, :]                      # (1, S) broadcast
        log_odds = np.clip(
            E_xi1 + rX * (E_log_alpha - E_log_b) - E_xi2,
            -500, 500
        )
        self.f = np.clip(expit(log_odds), EPS, 1 - EPS)    # (K, S)

    def _update_theta(self):
        r_k              = self.r.sum(axis=0)
        self.theta       = np.maximum(1.0 + r_k, NumericalStability.EPS)
        cumsum_r         = np.cumsum(r_k)
        self.theta_prime = np.maximum(
            self.nu + (cumsum_r[-1] - cumsum_r), NumericalStability.EPS
        )
        self.theta_prime[-1] = max(self.nu, NumericalStability.EPS)

    def _update_xi_star(self):
        """
        Beta posterior for per-cluster per-feature selection.

        ξ*_k1j = ξ_1 + N · f_kj
        ξ*_k2j = ξ_2 + N · (1 − f_kj)
        """
        self.xi_star[:, :, 0] = np.maximum(
            self.xi_1 + self.N * self.f, NumericalStability.EPS)
        self.xi_star[:, :, 1] = np.maximum(
            self.xi_2 + self.N * (1.0 - self.f), NumericalStability.EPS)

    def _update_lambda_star(self, X):
        """
        Exact conjugate CAVI update with per-cluster feature selection:
          λ_kj = ζ + Σ_i r_ik · f_kj · x_ij
               = ζ + f[k,:] * (r[:,k] @ X)
               = ζ + f * rX                       (K, S) element-wise
        """
        rX = self.r.T @ X                      # (K, S)
        self.lambda_star = np.maximum(
            self.zeta + self.f * rX,            # (K, S) element-wise
            0.1
        )

    def _update_iota_star(self, X):
        """
        Background Dirichlet update summing over all clusters:
          ι_j = η + Σ_k Σ_i r_ik · (1−f_kj) · x_ij
              = η + Σ_k (1−f[k,:]) * rX[k,:]
              = η + ((1 − f) * rX).sum(axis=0)   (S,)
        """
        rX = self.r.T @ X                      # (K, S)
        self.iota_star = np.maximum(
            self.eta + ((1.0 - self.f) * rX).sum(axis=0),
            0.1
        )

    # ─────────────────────────────────────────────────────────────────────
    # ELBO
    # ─────────────────────────────────────────────────────────────────────

    def _compute_elbo(self, X):
        EPS      = NumericalStability.EPS
        E_log_pi = self._E_log_pi()
        ll       = self._expected_log_lik(X)   # (N, K)

        elbo  = float(np.sum(self.r * (E_log_pi[None, :] + ll)))
        elbo -= float(np.sum(self.r * np.log(np.maximum(self.r, EPS))))
        return elbo

    # ─────────────────────────────────────────────────────────────────────
    # Pruning
    # ─────────────────────────────────────────────────────────────────────

    def _compute_weights(self):
        E_gamma  = self.theta / (self.theta + self.theta_prime)
        log_w    = np.log(np.maximum(E_gamma, NumericalStability.EPS))
        log_1mg  = np.log(np.maximum(1 - E_gamma, NumericalStability.EPS))
        log_w   += np.concatenate([[0.0], np.cumsum(log_1mg[:-1])])
        return np.exp(log_w)

    def _prune_empty_clusters(self):
        weights       = self._compute_weights()
        cluster_sizes = self.r.sum(axis=0)

        keep = (weights > self.prune_threshold) & \
               (cluster_sizes > self._min_cluster_size)

        n_keep = keep.sum()
        if n_keep < self._min_clusters:
            top_idx       = np.argsort(weights)[-self._min_clusters:]
            keep          = np.zeros(self.K, dtype=bool)
            keep[top_idx] = True
            n_keep        = self._min_clusters

        if n_keep < self.K:
            if self.verbose >= 1:
                removed = self.K - n_keep
                print(f"  Pruning: {self.K} → {n_keep} clusters "
                      f"(removed {removed}; min_K={self._min_clusters})")
            self.K           = n_keep
            self.r           = self.r[:, keep]
            self.theta       = self.theta[keep]
            self.theta_prime = self.theta_prime[keep]
            self.lambda_star = self.lambda_star[keep]
            self.f           = self.f[keep]           # (K, S) — prune rows
            self.xi_star     = self.xi_star[keep]     # (K, S, 2) — prune rows
            self.r          /= self.r.sum(axis=1, keepdims=True)
            self._clear_cache()
            self._pruned_at_least_once = True
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    # Main fit loop
    # ─────────────────────────────────────────────────────────────────────

    def fit(self, X):
        X            = check_array(X, dtype=np.float64)
        random_state = check_random_state(self.random_state)

        self._initialize_parameters(X, random_state)

        if self.verbose >= 1:
            print(f"\nStarting CAVI  (v2.1 — per-cluster feature selection)")
            print("=" * 70)

        t0 = time()

        for iteration in range(1, self.max_iter + 1):
            self.n_iter = iteration
            self._clear_cache()

            # ── CAVI updates ──────────────────────────────────────────────
            self._update_r(X)
            self._update_f(X)
            self._update_theta()
            self._update_xi_star()
            self._update_lambda_star(X)
            self._update_iota_star(X)

            # ── Pruning ───────────────────────────────────────────────────
            if iteration >= self.prune_start and iteration % self.prune_every == 0:
                self._prune_empty_clusters()

            # ── ELBO + convergence ────────────────────────────────────────
            if iteration % 10 == 0:
                elbo = self._compute_elbo(X)
                self.elbo_history.append(elbo)

                if self.verbose >= 1:
                    print(f"Iter {iteration:4d}: ELBO = {elbo:14.2f}, "
                          f"K = {self.K}, Time = {time()-t0:.1f}s")

                if self._pruned_at_least_once and len(self.elbo_history) >= 3:
                    recent  = self.elbo_history[-3:]
                    changes = [abs(recent[i] - recent[i-1]) /
                               (abs(recent[i]) + 1e-10)
                               for i in range(1, len(recent))]
                    if all(c < self.tol for c in changes):
                        self.converged = True
                        if self.verbose >= 1:
                            print(f"\n✓ Converged at iteration {iteration}")
                        break

        # Final pruning pass
        self._prune_empty_clusters()
        self.weights_ = self._compute_weights()

        if self.verbose >= 1:
            print(f"\nFinal results:")
            print(f"  Number of clusters : {self.K}")
            print(f"  Mixing weights     : {np.round(self.weights_, 4)}")
            print(f"  Total time         : {time()-t0:.2f}s")
            print(f"  Converged          : {self.converged}")

        return self

    # ─────────────────────────────────────────────────────────────────────
    # Prediction and inspection
    # ─────────────────────────────────────────────────────────────────────

    def predict(self, X):
        X        = check_array(X, dtype=np.float64)
        N_new    = X.shape[0]

        # Save and temporarily substitute state for new data
        r_orig, N_orig = self.r, self.N
        self.N = N_new
        self.r = np.ones((N_new, self.K)) / self.K
        self._clear_cache()

        ll       = self._expected_log_lik(X)   # uses self.f (K,S) unchanged
        E_log_pi = self._E_log_pi()
        log_r    = E_log_pi[None, :] + ll
        log_r   -= logsumexp(log_r, axis=1, keepdims=True)
        labels   = np.exp(log_r).argmax(axis=1)

        self.r, self.N = r_orig, N_orig
        self._clear_cache()
        return labels

    def get_selected_features(self, threshold=0.5):
        """
        Return per-cluster selected feature indices.

        Returns
        -------
        dict  {cluster_k: {'indices': [...], 'probabilities': [...]}}
        """
        selected = {}
        for k in range(self.K):
            idx = np.where(self.f[k] > threshold)[0]
            if len(idx) > 0:
                selected[k] = {
                    'indices':       idx.tolist(),
                    'probabilities': self.f[k, idx].tolist(),
                }
        return selected

    def get_cluster_signatures(self, threshold=0.5):
        """
        Return per-cluster feature signatures combining selection and alpha.

        Returns
        -------
        dict  {cluster_k: {'features', 'selection_probs', 'alpha_values',
                            'n_samples'}}
        """
        signatures = {}
        lam_sums = self.lambda_star.sum(axis=1)          # (K,)
        for k in range(self.K):
            idx = np.where(self.f[k] > threshold)[0]
            if len(idx) > 0:
                alpha_k = self.lambda_star[k] / lam_sums[k]
                signatures[k] = {
                    'features':        idx.tolist(),
                    'selection_probs': self.f[k, idx].tolist(),
                    'alpha_values':    alpha_k[idx].tolist(),
                    'n_samples':       int((self.r[:, k] > 0.5).sum()),
                }
        return signatures


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print("Smoke-test — DMM_SVVS_Variational_v2.1  (per-cluster feature selection)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    N, S, K_true = 150, 200, 3

    alpha = np.full((K_true, S), 0.1)
    block = S // K_true
    for k in range(K_true):
        alpha[k, k*block:(k+1)*block] = 3.0
    true_labels = rng.choice(K_true, size=N)
    X = np.array([rng.multinomial(5000, rng.dirichlet(alpha[true_labels[i]]))
                  for i in range(N)], dtype=float)

    model = DMM_SVVS_Variational_v2_1(
        K_max=10, nu='auto', max_iter=300, verbose=1, random_state=42
    )
    model.fit(X)

    pred = model.predict(X)
    print(f"\nARI = {adjusted_rand_score(true_labels, pred):.3f}")
    print(f"NMI = {normalized_mutual_info_score(true_labels, pred):.3f}")
    print(f"K estimated = {model.K}  (true K = {K_true})")

    print("\nPer-cluster selected features (threshold=0.5):")
    sigs = model.get_cluster_signatures(threshold=0.5)
    for k, sig in sigs.items():
        print(f"  Cluster {k}: {len(sig['features'])} features selected, "
              f"n_samples={sig['n_samples']}")
