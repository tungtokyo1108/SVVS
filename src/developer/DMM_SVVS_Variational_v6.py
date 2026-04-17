#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS Variational Inference — Version 6 (SSL Feature Selection)

Changes from v2 → v6  (single focused change):
  The per-feature selection indicator is replaced by a
  Spike-and-Slab LASSO (SSL) prior, following Yao, Xie & Xu (JMLR 2025).

  v2 used:  Beta(ξ₁, ξ₂) prior on scalar selection probability f_j ∈ (0,1).

  v6 uses:  SSL prior on the feature importance indicator ξ_j ∈ {0,1}:
              π(ξ_j | θ) = Bernoulli(θ)
              θ ~ Beta(1, βθ)   with  βθ = S^(1+κ) / log(S)

            The posterior slab-inclusion probability ξ*_j is derived from
            the same log-odds signal used in v2's _update_f, but the
            selection prior contribution is now the SSL Laplace-based term
            instead of the Beta digamma term:

              log_odds_j = [v2 data signal]
                         + log(λ₁/λ₀) + (λ₀ − λ₁)·signal_j
                         + log(θ_ssl / (1 − θ_ssl))

            where signal_j = max(raw_log_odds_from_data, 0) (the part of
            the log-odds that is driven by data, kept non-negative as a
            proxy for |μ_j| in the SSL formulation).

  SSL prior motivation:
    - Spike: Laplace(g_j; λ₀) — heavy shrinkage to zero (noise features)
    - Slab:  Laplace(g_j; λ₁) — weak regularisation (informative features)
    - λ₀ ≫ λ₁ so the spike is tight and the slab is diffuse.

  Selection decision:  feature j is "selected" if  ξ*_j > 0.5

  All other components — DMM likelihood, stick-breaking DP prior, cluster
  responsibilities r, Dirichlet parameters λ*, ι*, pruning logic — are
  identical to v2.

Key new hyperparameters
-----------------------
  lambda0   : float  (default 50)
      Spike rate. Large ⟹ strong shrinkage.
  lambda1   : float  (default 1)
      Slab rate. Small ⟹ weakly regularised non-zero.
  kappa     : float  (default 0.1)
      Controls the Beta hyperparameter βθ = S^(1+κ)/log(S).
  init_xi   : float  (default 0.3)
      Initial slab-inclusion probability warm-start.

Removed hyperparameters (v2-specific)
--------------------------------------
  xi_1, xi_2      — replaced by lambda0, lambda1, kappa
  selection_prior — replaced by init_xi

Author: v6 extension of v2 following Yao, Xie & Xu (JMLR 2025)
"""

import numpy as np
from scipy.special import digamma, logsumexp, expit
from sklearn.utils import check_array, check_random_state
from sklearn import cluster
from time import time


# ─────────────────────────────────────────────────────────────────────────────
# Numerical utilities  (identical to v2)
# ─────────────────────────────────────────────────────────────────────────────

class NumericalStability:
    EPS     = 1e-10
    MAX_EXP = 500

    @staticmethod
    def safe_log(x):
        return np.log(np.maximum(x, NumericalStability.EPS))

    @staticmethod
    def safe_exp(x):
        return np.exp(np.clip(x, -NumericalStability.MAX_EXP,
                               NumericalStability.MAX_EXP))


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class DMM_SVVS_Variational_v6:
    """
    Dirichlet Multinomial Mixture with SSL Feature Selection — v6.

    The ONLY difference from v2 is the feature-selection prior:
    a Spike-and-Slab LASSO (SSL) replaces the Beta prior.

    The data-driven log-odds signal from v2 (_update_f) is preserved exactly.
    The SSL changes only the PRIOR contribution to that log-odds:
      v2:  prior term = E[log(ξ₁_j/ξ₂_j)]  (Beta digamma difference)
      v6:  prior term = log(λ₁/λ₀) + (λ₀ − λ₁)·signal_j
                      + log(θ_ssl/(1 − θ_ssl))

    Parameters
    ----------
    K_max : int
        Maximum / initial number of clusters (truncation level).
    nu : float or 'auto'
        DP concentration for stick-breaking.
        'auto' → 1/K_max  (recommended).
    zeta : float
        Prior concentration for cluster Dirichlet (alpha_k).
    eta : float
        Prior concentration for background Dirichlet (beta).

    --- SSL-specific (replaces xi_1, xi_2, selection_prior from v2) ---
    lambda0 : float
        Spike Laplace rate (λ₀ ≫ λ₁). Default 50.
    lambda1 : float
        Slab  Laplace rate (λ₁ ≪ λ₀). Default 1.
    kappa : float
        Exponent for βθ = S^(1+κ) / log(S).  Default 0.1.
    init_xi : float
        Initial slab-inclusion probability ξ*_j ∈ (0,1).  Default 0.3.
    -------------------------------------------------------------------

    tol : float
        Relative ELBO convergence tolerance.  Default 1e-4.
    max_iter : int
        Maximum CAVI iterations.
    prune_threshold : float
        Weight threshold for cluster pruning.
    min_clusters : int or None
        Minimum clusters to keep.  None → max(2, K_max // 5).
    prune_start : int
        Iteration at which pruning begins.  Default 10.
    prune_every : int
        Prune every this many iterations.  Default 5.
    verbose : int
    random_state : int
    """

    def __init__(self,
                 K_max=10,
                 nu='auto',
                 zeta=1.0,
                 eta=1.0,
                 # ── SSL hyperparameters (replaces xi_1/xi_2/selection_prior) ──
                 lambda0=50.0,
                 lambda1=1.0,
                 kappa=0.1,
                 init_xi=0.3,
                 # ── shared with v2 ────────────────────────────────────────────
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
        # SSL
        self.lambda0         = lambda0
        self.lambda1         = lambda1
        self.kappa           = kappa
        self.init_xi         = init_xi
        # shared
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
        self.r = None                       # (N, K) responsibilities
        # --- SSL state (replaces self.f and self.xi_star from v2) ---
        self.xi_post    = None              # (S,)  posterior slab probability ξ*_j
        self.theta_ssl  = None              # scalar posterior mean of θ
        self.beta_theta = None              # scalar βθ hyperparameter
        # Cluster / background Dirichlet
        self.theta      = None              # (K,) stick-breaking first parameter
        self.theta_prime= None              # (K,) stick-breaking second parameter
        self.lambda_star= None              # (K, S) cluster Dirichlet
        self.iota_star  = None              # (S,)  background Dirichlet
        self.elbo_history = []
        self.converged    = False
        self.n_iter       = 0
        self._cache       = {}
        self._pruned_at_least_once = False

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_nu(self):
        if self.nu_input == 'auto':
            return 1.0 / self.K_max
        nu = float(self.nu_input)
        if nu > 2.0 and self.verbose >= 1:
            print(f"  [Warning] nu={nu:.1f} is large — stick-breaking will spread mass "
                  f"across all {self.K_max} clusters and resist pruning. "
                  f"Consider nu='auto' or nu<1.")
        return nu

    def _initialize_parameters(self, X, random_state):
        self.N, self.S = X.shape
        self.K  = self.K_max
        self.nu = self._resolve_nu()

        self._min_clusters   = (max(2, self.K_max // 5)
                                if self.min_clusters is None
                                else int(self.min_clusters))
        self._min_cluster_size = max(1.0, self.N / (5.0 * self.K_max))

        # ── SSL hyperparameter βθ = S^(1+κ) / log(S) ──────────────────────
        self.beta_theta = (self.S ** (1.0 + self.kappa)
                           / np.log(max(self.S, 2.0)))

        if self.verbose >= 1:
            print(f"Initializing v6: N={self.N}, S={self.S}, K_max={self.K_max}")
            print(f"  ν={self.nu:.4f}, ζ={self.zeta}, η={self.eta}")
            print(f"  SSL: λ₀={self.lambda0}, λ₁={self.lambda1}, "
                  f"κ={self.kappa}, βθ={self.beta_theta:.2f}")
            print(f"  min_K={self._min_clusters}, "
                  f"min_cluster_size={self._min_cluster_size:.1f}")

        # Responsibilities from k-means (identical to v2)
        self.r = self._init_responsibilities_kmeans(X, random_state)

        # ── SSL state initialisation ──────────────────────────────────────
        # ξ*_j : warm-start at init_xi
        self.xi_post = np.full(self.S, np.clip(self.init_xi, 1e-6, 1 - 1e-6))

        # θ_ssl (slab inclusion rate): prior mean = 1/(1+βθ)
        self.theta_ssl = 1.0 / (1.0 + self.beta_theta)

        # Stick-breaking (identical to v2)
        self.theta        = np.ones(self.K)
        self.theta_prime  = np.ones(self.K) * self.nu

        # Cluster Dirichlet from k-means statistics (identical to v2)
        self.lambda_star = np.zeros((self.K, self.S))
        for k in range(self.K):
            mask = self.r[:, k] > 0.1
            if mask.sum() > 0:
                mu = X[mask].mean(axis=0) + 0.5
                self.lambda_star[k] = mu / mu.sum() * (self.zeta * self.S)
            else:
                self.lambda_star[k] = self.zeta
        self.lambda_star = np.maximum(self.lambda_star, 0.1)

        # Background Dirichlet (identical to v2)
        mu_all = X.mean(axis=0) + 0.5
        self.iota_star = mu_all / mu_all.sum() * (self.eta * self.S)
        self.iota_star = np.maximum(self.iota_star, 0.1)

        self._cache = {}
        self._pruned_at_least_once = False

    def _init_responsibilities_kmeans(self, X, random_state):
        """Identical to v2."""
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

    # ─────────────────────────────────────────────────────────────────────────
    # Cached expectations  (identical to v2)
    # ─────────────────────────────────────────────────────────────────────────

    def _E_log_alpha(self):
        if 'Ela' not in self._cache:
            s = self.lambda_star.sum(axis=1, keepdims=True)
            self._cache['Ela'] = digamma(self.lambda_star) - digamma(s)
        return self._cache['Ela']

    def _E_log_beta(self):
        if 'Elb' not in self._cache:
            self._cache['Elb'] = (digamma(self.iota_star)
                                  - digamma(self.iota_star.sum()))
        return self._cache['Elb']

    def _E_log_pi(self):
        if 'Elpi' not in self._cache:
            st   = self.theta + self.theta_prime
            elg  = digamma(self.theta)       - digamma(st)
            el1g = digamma(self.theta_prime) - digamma(st)
            cum  = np.concatenate([[0.0], np.cumsum(el1g[:-1])])
            self._cache['Elpi'] = elg + cum
        return self._cache['Elpi']

    def _clear_cache(self):
        self._cache = {}

    # ─────────────────────────────────────────────────────────────────────────
    # SSL utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _effective_selection(self):
        """
        Return effective per-feature selection weight f_j = ξ*_j ∈ (0,1).

        This is the posterior slab-inclusion probability, directly analogous
        to f.mean(axis=0) from v2.
        """
        return self.xi_post   # (S,)

    # ─────────────────────────────────────────────────────────────────────────
    # CAVI update steps
    # ─────────────────────────────────────────────────────────────────────────

    def _expected_log_lik(self, X):
        """
        (N, K) expected log-likelihood — identical structure to v2.

        E[log p(x_i | z_i=k)] = Σ_j [ f_j · x_ij · E[log α_kj]
                                      + (1−f_j) · x_ij · E[log β_j] ]
        where f_j = ξ*_j  (SSL posterior slab probability).
        """
        E_log_a = self._E_log_alpha()          # (K, S)
        E_log_b = self._E_log_beta()           # (S,)
        f_j     = self._effective_selection()  # (S,)
        combined = (f_j[None, :] * E_log_a
                    + (1.0 - f_j)[None, :] * E_log_b[None, :])  # (K, S)
        return X @ combined.T                  # (N, K)

    def _update_r(self, X):
        """Identical to v2 (responsibilities update)."""
        EPS      = NumericalStability.EPS
        E_log_pi = self._E_log_pi()
        ll       = self._expected_log_lik(X)
        log_r    = E_log_pi[None, :] + ll
        log_r   -= logsumexp(log_r, axis=1, keepdims=True)
        self.r   = np.exp(log_r)
        self.r   = np.maximum(self.r, EPS)
        self.r  /= self.r.sum(axis=1, keepdims=True)

    def _update_xi_post(self, X):
        """
        Update posterior slab-inclusion probability ξ*_j  (replaces
        _update_f + _update_xi_star from v2).

        The data-driven signal is computed identically to v2's _update_f:
            data_signal_j = Σ_i Σ_k r_ik · x_ij · E[log α_kj]
                          − Σ_i       x_ij · E[log β_j]

        In v2 the prior contribution to the log-odds came from the Beta:
            prior_term_v2 = E[log ξ₁_j] − E[log ξ₂_j]  (Beta digamma)

        In v6 the prior is the SSL Laplace prior.  For the DMM context the
        natural measure of feature importance is the raw data-signal above.
        We use it as a proxy for |μ_j| in the Laplace prior:
            signal_j = max(data_signal_j, 0)   (non-negative, like |μ|)

        The SSL log-odds prior contribution is:
            log(λ₁/λ₀) + (λ₀ − λ₁) · signal_j + log(θ_ssl/(1−θ_ssl))

        Derivation: for the spike-slab Laplace mixture the posterior odds
        of ξ_j=1 (slab) vs ξ_j=0 (spike) satisfy
            log p(ξ_j=1|g_j) − log p(ξ_j=0|g_j)
              = log(λ₁/λ₀) + (λ₀−λ₁)|g_j| + log(θ/(1−θ))
        (from the Laplace density ratio and the Bernoulli prior).
        Here g_j is the inferred feature importance; we proxy |g_j|
        with the non-negative data signal.

        Full log-odds:
            log_odds_j = data_signal_j                   ← same as v2
                       + log(λ₁/λ₀)                     ← SSL spike-slab ratio
                       + (λ₀ − λ₁) · signal_j           ← SSL Laplace term
                       + log(θ_ssl / (1 − θ_ssl))        ← SSL inclusion rate
        """
        EPS = NumericalStability.EPS

        E_log_alpha = self._E_log_alpha()   # (K, S)
        E_log_beta  = self._E_log_beta()    # (S,)

        # ── Data signal (identical to v2 _update_f) ───────────────────────
        # r-weighted E[log α]: (N,K)@(K,S) = (N,S)
        el_alpha_ni = self.r @ E_log_alpha  # (N, S)

        log_ps_raw = (X * el_alpha_ni).sum(axis=0)          # (S,)
        log_pu_raw = (X * E_log_beta[None, :]).sum(axis=0)  # (S,)

        data_signal = log_ps_raw - log_pu_raw               # (S,) signed

        # ── SSL prior contribution ────────────────────────────────────────
        # Proxy for |g_j|: non-negative part of the data signal
        signal_j = np.maximum(data_signal, 0.0)             # (S,) ≥ 0

        # log(λ₁/λ₀): negative because λ₀ > λ₁ — penalises slab by default
        log_lambda_ratio = np.log(self.lambda1 / max(self.lambda0, EPS))

        # (λ₀ − λ₁)·signal_j: positive for informative features, rewards slab
        laplace_diff_term = (self.lambda0 - self.lambda1) * signal_j

        # Prior log-odds from θ_ssl
        safe_theta    = np.clip(self.theta_ssl, EPS, 1.0 - EPS)
        prior_logodds = np.log(safe_theta / (1.0 - safe_theta))

        # ── Full log-odds ─────────────────────────────────────────────────
        log_odds = (data_signal
                    + log_lambda_ratio
                    + laplace_diff_term
                    + prior_logodds)                         # (S,)
        log_odds = np.clip(log_odds, -500.0, 500.0)

        self.xi_post = np.clip(expit(log_odds), EPS, 1.0 - EPS)  # (S,)

    def _update_theta_ssl(self):
        """
        Update posterior mean of slab-inclusion rate θ.

        θ ~ Beta(1, βθ)  prior.
        Posterior:  θ | ξ  ~  Beta(1 + Σ_j ξ*_j,  βθ + S − Σ_j ξ*_j)
        Posterior mean = (1 + Σ_j ξ*_j) / (1 + βθ + S)
        """
        sum_xi = self.xi_post.sum()
        self.theta_ssl = (1.0 + sum_xi) / (1.0 + self.beta_theta + self.S)
        self.theta_ssl = np.clip(self.theta_ssl,
                                 NumericalStability.EPS,
                                 1.0 - NumericalStability.EPS)

    def _update_theta(self):
        """Stick-breaking update — identical to v2."""
        r_k              = self.r.sum(axis=0)
        self.theta       = np.maximum(1.0 + r_k, NumericalStability.EPS)
        cumsum_r         = np.cumsum(r_k)
        self.theta_prime = np.maximum(
            self.nu + (cumsum_r[-1] - cumsum_r), NumericalStability.EPS
        )
        self.theta_prime[-1] = max(self.nu, NumericalStability.EPS)

    def _update_lambda_star(self, X):
        """
        Cluster Dirichlet update — identical in structure to v2.

          λ_kj = ζ + Σ_i r_ik · f_j · x_ij
        where f_j = ξ*_j  (SSL slab probability).
        """
        f_j  = self._effective_selection()   # (S,)
        fX   = f_j[None, :] * X             # (N, S)
        self.lambda_star = np.maximum(
            self.zeta + self.r.T @ fX,       # (K, S)
            0.1
        )

    def _update_iota_star(self, X):
        """
        Background Dirichlet update — identical in structure to v2.

          ι_j = η + Σ_i (1 − f_j) · x_ij
        where f_j = ξ*_j.
        """
        f_j = self._effective_selection()    # (S,)
        self.iota_star = np.maximum(
            self.eta + ((1.0 - f_j)[None, :] * X).sum(axis=0),
            0.1
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ELBO
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_elbo(self, X):
        """
        ELBO — extends v2 by adding the SSL prior / entropy terms.

        ELBO = E[log p(X|Z,ξ)] + E[log p(Z|π)] − E[log q(Z)]
             + E[log p(ξ|θ)] − H[q(ξ)] + E[log p(θ)] terms
        """
        EPS      = NumericalStability.EPS
        E_log_pi = self._E_log_pi()
        ll       = self._expected_log_lik(X)

        # Data + assignment terms (identical to v2)
        elbo  = float(np.sum(self.r * (E_log_pi[None, :] + ll)))
        elbo -= float(np.sum(self.r * np.log(np.maximum(self.r, EPS))))

        # SSL Bernoulli prior: E[log p(ξ|θ_ssl)]
        safe_theta = np.clip(self.theta_ssl, EPS, 1.0 - EPS)
        xi = self.xi_post
        elbo += float(np.sum(
            xi * np.log(safe_theta)
            + (1.0 - xi) * np.log(1.0 - safe_theta)
        ))

        # Bernoulli entropy H[q(ξ_j)] = −ξ*·log ξ* − (1−ξ*)·log(1−ξ*)
        safe_xi = np.clip(xi, EPS, 1.0 - EPS)
        elbo += float(np.sum(
            -safe_xi * np.log(safe_xi)
            - (1.0 - safe_xi) * np.log(1.0 - safe_xi)
        ))

        # θ Beta prior: E[log p(θ)] ≈ (βθ − 1)·log(1 − θ_ssl)
        elbo += (self.beta_theta - 1.0) * np.log(1.0 - safe_theta)

        return elbo

    # ─────────────────────────────────────────────────────────────────────────
    # Pruning  (identical logic to v2)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_weights(self):
        E_gamma  = self.theta / (self.theta + self.theta_prime)
        log_w    = np.log(np.maximum(E_gamma, NumericalStability.EPS))
        log_1mg  = np.log(np.maximum(1 - E_gamma, NumericalStability.EPS))
        log_w   += np.concatenate([[0.0], np.cumsum(log_1mg[:-1])])
        return np.exp(log_w)

    def _prune_empty_clusters(self):
        """
        AND-logic cluster pruning — identical to v2.
        Keeps clusters where BOTH weight > threshold AND size > min_cluster_size.
        """
        weights       = self._compute_weights()
        cluster_sizes = self.r.sum(axis=0)

        keep = ((weights > self.prune_threshold) &
                (cluster_sizes > self._min_cluster_size))

        n_keep = keep.sum()
        if n_keep < self._min_clusters:
            top_idx        = np.argsort(weights)[-self._min_clusters:]
            keep           = np.zeros(self.K, dtype=bool)
            keep[top_idx]  = True
            n_keep         = self._min_clusters

        if n_keep < self.K:
            if self.verbose >= 1:
                removed = self.K - n_keep
                print(f"  Pruning: {self.K} → {n_keep} clusters "
                      f"(removed {removed}; min_K={self._min_clusters})")
            self.K            = n_keep
            self.r            = self.r[:, keep]
            self.theta        = self.theta[keep]
            self.theta_prime  = self.theta_prime[keep]
            self.lambda_star  = self.lambda_star[keep]
            self.r           /= self.r.sum(axis=1, keepdims=True)
            self._clear_cache()
            self._pruned_at_least_once = True
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Main fit loop
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, X):
        """
        Fit via coordinate ascent variational inference.

        CAVI order per iteration:
          1. r          — responsibilities  (identical to v2)
          2. xi_post    — SSL slab-inclusion posterior  (replaces f + xi_star)
          3. theta_ssl  — SSL slab-inclusion rate
          4. theta / theta_prime — stick-breaking  (identical to v2)
          5. lambda_star — cluster Dirichlet  (identical to v2)
          6. iota_star   — background Dirichlet  (identical to v2)
        """
        X            = check_array(X, dtype=np.float64)
        random_state = check_random_state(self.random_state)
        self._initialize_parameters(X, random_state)

        if self.verbose >= 1:
            print(f"\nStarting CAVI  (v6 — SSL feature selection)")
            print("=" * 70)

        t0 = time()

        for iteration in range(1, self.max_iter + 1):
            self.n_iter = iteration
            self._clear_cache()

            # ── 1. Responsibilities ───────────────────────────────────────
            self._update_r(X)

            # ── 2-3. SSL feature-selection block ─────────────────────────
            self._update_xi_post(X)       # slab-inclusion posterior
            self._update_theta_ssl()      # slab-rate posterior

            # ── 4. Stick-breaking ─────────────────────────────────────────
            self._update_theta()

            # ── 5-6. Dirichlet parameters ─────────────────────────────────
            self._update_lambda_star(X)
            self._update_iota_star(X)

            # ── Pruning ───────────────────────────────────────────────────
            if (iteration >= self.prune_start
                    and iteration % self.prune_every == 0):
                self._prune_empty_clusters()

            # ── ELBO + convergence ────────────────────────────────────────
            if iteration % 10 == 0:
                elbo = self._compute_elbo(X)
                self.elbo_history.append(elbo)

                if self.verbose >= 1:
                    n_sel = int((self.xi_post > 0.5).sum())
                    print(f"Iter {iteration:4d}: ELBO = {elbo:14.2f}, "
                          f"K = {self.K}, "
                          f"selected features = {n_sel}/{self.S}, "
                          f"θ_ssl = {self.theta_ssl:.4f}, "
                          f"Time = {time()-t0:.1f}s")

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

        # Convenience attributes
        self.selected_features_ = np.where(self.xi_post > 0.5)[0]
        self.xi_post_            = self.xi_post.copy()

        if self.verbose >= 1:
            print(f"\nFinal results (v6):")
            print(f"  Clusters           : {self.K}")
            print(f"  Mixing weights     : {np.round(self.weights_, 4)}")
            print(f"  Selected features  : {len(self.selected_features_)} / {self.S}")
            print(f"  Slab rate θ_ssl    : {self.theta_ssl:.4f}")
            print(f"  Total time         : {time()-t0:.2f}s")
            print(f"  Converged          : {self.converged}")

        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction and inspection  (v2-compatible interface)
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, X):
        """Predict cluster labels for X."""
        X     = check_array(X, dtype=np.float64)
        N_new = X.shape[0]
        # Temporarily swap state for prediction (mirrors v2 pattern)
        xi_orig       = self.xi_post
        r_orig, N_orig = self.r, self.N
        self.N = N_new
        self.r = np.zeros((N_new, self.K))
        self._clear_cache()
        self._update_r(X)
        labels = self.r.argmax(axis=1)
        self.r, self.N = r_orig, N_orig
        self.xi_post   = xi_orig
        self._clear_cache()
        return labels

    def get_selected_features(self, threshold=0.5):
        """
        Return indices and SSL slab probabilities of selected features.

        A feature j is selected if ξ*_j > threshold (posterior probability
        that the feature belongs to the slab, i.e., is discriminative).
        """
        idx = np.where(self.xi_post > threshold)[0]
        return {
            'indices':  idx.tolist(),
            'xi_post':  self.xi_post[idx].tolist(),
        }

    def get_cluster_signatures(self, threshold=0.5):
        """
        Per-cluster summary of selected features.
        Uses the global SSL selection ξ*_j (joint-SSL: one sparsity pattern
        shared across all clusters, consistent with Yao et al. 2025).
        """
        signatures = {}
        for k in range(self.K):
            mask = self.r[:, k] > 0.5
            if mask.sum() > 0:
                important = np.where(self.xi_post > threshold)[0]
                if len(important) > 0:
                    lam_sum = self.lambda_star[k].sum()
                    alpha_k = self.lambda_star[k] / lam_sum
                    signatures[k] = {
                        'features':     important.tolist(),
                        'xi_post':      self.xi_post[important].tolist(),
                        'alpha_values': alpha_k[important].tolist(),
                        'n_samples':    int(mask.sum()),
                    }
        return signatures

    def summary(self):
        """Print a concise summary of fitted model."""
        print("=" * 60)
        print("DMM-SVVS v6  (Spike-and-Slab LASSO feature selection)")
        print("=" * 60)
        print(f"  Samples            : {self.N}")
        print(f"  Features (S)       : {self.S}")
        print(f"  Clusters (K)       : {self.K}")
        print(f"  Selected features  : {len(self.selected_features_)}")
        print(f"  Slab inclusion θ   : {self.theta_ssl:.4f}")
        print(f"  λ₀ (spike rate)    : {self.lambda0}")
        print(f"  λ₁ (slab  rate)    : {self.lambda1}")
        print(f"  βθ                 : {self.beta_theta:.2f}")
        print(f"  ELBO (final)       : "
              f"{self.elbo_history[-1] if self.elbo_history else 'n/a':.2f}")
        print(f"  Converged          : {self.converged}")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print("Smoke-test — DMM_SVVS_Variational_v6  (SSL feature selection)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    N, S, K_true = 150, 200, 3

    # Block-structured DMM data with only the first block*K_true features
    # being discriminative  (rest are noise → SSL should suppress them)
    alpha = np.full((K_true, S), 0.05)         # near-uniform noise features
    block = S // K_true
    for k in range(K_true):
        alpha[k, k*block:(k+1)*block] = 3.0   # discriminative block

    true_labels = rng.choice(K_true, size=N)
    X = np.array([rng.multinomial(5000, rng.dirichlet(alpha[true_labels[i]]))
                  for i in range(N)], dtype=float)

    model = DMM_SVVS_Variational_v6(
        K_max    = 10,
        nu       = 'auto',
        lambda0  = 50.0,
        lambda1  = 1.0,
        kappa    = 0.1,
        init_xi  = 0.3,
        max_iter = 300,
        verbose  = 1,
        random_state = 42
    )
    model.fit(X)

    pred = model.predict(X)
    print(f"\nARI = {adjusted_rand_score(true_labels, pred):.3f}")
    print(f"NMI = {normalized_mutual_info_score(true_labels, pred):.3f}")
    print(f"K estimated = {model.K}  (true K = {K_true})")

    model.summary()

    # Check that SSL finds roughly the right features
    sel = model.get_selected_features(threshold=0.5)
    true_sel = set(range(block * K_true))       # first block*K_true are discriminative
    found    = set(sel['indices'])
    print(f"\nTrue discriminative features : {len(true_sel)}")
    print(f"SSL selected features        : {len(found)}")
    print(f"Overlap (precision/recall)   : "
          f"{len(found & true_sel)}/{len(found)} sel, "
          f"{len(found & true_sel)}/{len(true_sel)} true")
