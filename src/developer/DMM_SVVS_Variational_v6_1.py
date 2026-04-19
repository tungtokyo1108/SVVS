#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS Variational Inference — Version 6.1
Spike-and-Slab LASSO (SSL) Feature Selection for DMM Clustering

═══════════════════════════════════════════════════════════════════════
DEEP UNDERSTANDING OF YAO, XIE & XU (JMLR 2025) APPLIED TO DMM
═══════════════════════════════════════════════════════════════════════

PAPER CORE IDEAS (Section 2.4):
────────────────────────────────
The paper proposes a Bayesian sparse Gaussian mixture where cluster
mean vectors μ₁,...,μ_K ∈ ℝᵖ share a COMMON sparsity pattern via the
joint-SSL prior:

  1. JOINT SPARSITY: One binary indicator ξ_j ∈ {0,1} per feature j
     governs ALL clusters simultaneously ("joint-SSL").

  2. SSL PRIOR: π(μ_kj|λ₀,λ₁,ξ_j) = (1-ξ_j)·Lap(μ_kj;λ₀) + ξ_j·Lap(μ_kj;λ₁)
     λ₀ ≫ λ₁: spike ≈ point mass at 0, slab allows large values.

  3. HYPERPRIOR: θ ~ Beta(1,βθ), βθ = p^(1+κ)/log(p) → E[θ] ≈ 0 for large p.

  4. POSTERIOR LOG-ODDS:
       log-odds(ξ_j=1) = log(λ₁/λ₀) + (λ₀-λ₁)|μ_kj| + log(θ/(1-θ))

DMM ADAPTATION (v6.1 — corrected):
────────────────────────────────────
For DMM (count data, Dirichlet-Multinomial), we replace |μ_kj| with a
BASELINE-CORRECTED between-cluster discriminative signal G_excess_j:

  G_raw_j  = Σ_k r̄_k · |E[log α_kj] - E[log β_j]|   [raw cluster-vs-bg signal]
  G_excess_j = max(G_raw_j - null_baseline, 0)          [corrected, ≈ 0 under H0]

WHY BASELINE CORRECTION:
  In Gaussian: |μ_kj| = 0 exactly for null features (spike prior shrinks to zero).
  In DMM: E[log α_kj] ≠ E[log β_j] even for null features (estimated from finite
  data with different subsets), so raw G_j >> 0 for ALL features, breaking the
  SSL threshold mechanism. The null baseline (high percentile of G_raw) corrects
  this: null features → G_excess ≈ 0; discriminative features → G_excess >> 0.

KEY FIXES vs v6 and earlier v6.1 attempts:
  - Correct null baseline: percentile adapts to expected sparsity fraction
  - EMA damping of xi_post: prevents oscillations from shifting baseline
  - Burn-in phase: first burn_in iters run without SSL gating to let
    clustering converge before SSL selection begins
  - Convergence: tracks xi_post change, not ELBO (which is non-monotone
    due to the baseline percentile computation)
  - Convergence check: no longer gated on pruning having occurred

CAVI UPDATE ORDER:
  1. r   — responsibilities [uses ξ*, α, β, π]
  2. ξ*  — SSL slab-inclusion [uses G_excess, θ_ssl; EMA-damped]
  3. θ   — slab rate [exact Beta posterior]
  4. θ/θ'— stick-breaking [uses r]
  5. λ*  — cluster Dirichlet [gated by ξ* after burn-in]
  6. ι*  — background Dirichlet [gated by 1-ξ* after burn-in]

Author: v6.1 corrected — faithful adaptation of Yao, Xie & Xu (JMLR 2025)
        for Dirichlet-Multinomial Mixture (DMM)
"""

import numpy as np
from scipy.special import digamma, logsumexp, expit, gammaln
from sklearn.utils import check_array, check_random_state
from sklearn import cluster
from time import time


# ─────────────────────────────────────────────────────────────────────────────
# Numerical utilities
# ─────────────────────────────────────────────────────────────────────────────

class _NS:
    EPS     = 1e-10
    MAX_EXP = 500.0

    @staticmethod
    def log(x):
        return np.log(np.maximum(x, _NS.EPS))

    @staticmethod
    def clip_exp(x):
        return np.exp(np.clip(x, -_NS.MAX_EXP, _NS.MAX_EXP))


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class DMM_SVVS_Variational_v6_1:
    """
    Dirichlet-Multinomial Mixture with Joint Spike-and-Slab LASSO Feature Selection.
    Version 6.1 — faithful adaptation of Yao, Xie & Xu (JMLR 2025) for DMM.

    Parameters
    ──────────
    K_max : int
        Maximum number of clusters (stick-breaking truncation). Default 10.
    nu : float or 'auto'
        DP concentration parameter. 'auto' → 1/K_max.
    zeta : float
        Prior concentration for cluster Dirichlet α_k. Default 1.0.
    eta : float
        Prior concentration for background Dirichlet β. Default 1.0.

    SSL hyperparameters (Yao et al. 2025, Section 2.4):
    lambda0 : float
        Spike Laplace rate λ₀. Default 100.0.
    lambda1 : float
        Slab Laplace rate λ₁ (≪ λ₀). Default 1.0.
    kappa : float
        Sparsity pressure exponent: βθ = S^(1+κ)/log(S). Default 0.1.
    xi_ema : float
        EMA decay for xi_post updates ∈ (0,1]. 1.0 = no damping. Default 0.5.
    burn_in : int
        Iterations without SSL gating (clustering-only warm-up). Default 20.

    tol : float
        xi_post change convergence tolerance. Default 1e-3.
    max_iter : int
        Maximum CAVI iterations. Default 300.
    prune_threshold : float
        Mixing weight threshold for cluster pruning. Default 0.02.
    min_clusters : int or None
        Minimum clusters to keep. None → max(2, K_max//5).
    prune_start : int
        Iteration to begin pruning. Default 10.
    prune_every : int
        Prune every N iterations. Default 5.
    verbose : int
        Verbosity (0=silent, 1=standard, 2=detailed).
    random_state : int
    """

    def __init__(self,
                 K_max=10,
                 nu='auto',
                 zeta=1.0,
                 eta=1.0,
                 lambda0=100.0,
                 lambda1=1.0,
                 kappa=0.1,
                 xi_ema=0.5,
                 burn_in=20,
                 tol=1e-3,
                 max_iter=300,
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
        self.lambda0         = lambda0
        self.lambda1         = lambda1
        self.kappa           = kappa
        self.xi_ema          = xi_ema
        self.burn_in         = burn_in
        self.tol             = tol
        self.max_iter        = max_iter
        self.prune_threshold = prune_threshold
        self.min_clusters    = min_clusters
        self.prune_start     = prune_start
        self.prune_every     = prune_every
        self.verbose         = verbose
        self.random_state    = random_state

        # Runtime state (set during fit)
        self.N = self.S = self.K = self.nu = None

        # Variational parameters
        self.r           = None   # (N, K)
        self.xi_post     = None   # (S,)   posterior slab-inclusion probability
        self.theta_ssl   = None   # scalar posterior E[θ]
        self.beta_theta  = None   # scalar βθ hyperparameter
        self.theta       = None   # (K,)   stick-breaking α params
        self.theta_prime = None   # (K,)   stick-breaking β params
        self.lambda_star = None   # (K, S) cluster Dirichlet params
        self.iota_star   = None   # (S,)   background Dirichlet params

        self.elbo_history  = []
        self.converged     = False
        self.n_iter        = 0
        self._cache        = {}
        self._null_baseline = None   # frozen at burn-in end; None = not set yet

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_nu(self):
        if self.nu_input == 'auto':
            return 1.0 / self.K_max
        nu = float(self.nu_input)
        if nu > 2.0 and self.verbose >= 1:
            print(f"  [Warning] nu={nu:.1f} is large — may resist pruning.")
        return nu

    def _initialize_parameters(self, X, random_state):
        self.N, self.S = X.shape
        self.K         = self.K_max
        self.nu        = self._resolve_nu()

        self._min_clusters     = (max(2, self.K_max // 5)
                                  if self.min_clusters is None
                                  else int(self.min_clusters))
        self._min_cluster_size = max(1.0, self.N / (5.0 * self.K_max))

        # βθ = S^(1+κ)/log(S)  (Yao et al. Eq. 8)
        S_safe = max(self.S, 2)
        self.beta_theta = (S_safe ** (1.0 + self.kappa)) / np.log(S_safe)

        if self.verbose >= 1:
            prior_mean = 1.0 / (1.0 + self.beta_theta)
            print(f"\nInitializing DMM-SVVS v6.1 (Joint-SSL Feature Selection)")
            print(f"  N={self.N}, S={self.S}, K_max={self.K_max}")
            print(f"  ν={self.nu:.4f}, ζ={self.zeta}, η={self.eta}")
            print(f"  SSL: λ₀={self.lambda0}, λ₁={self.lambda1}, κ={self.kappa}")
            print(f"  βθ={self.beta_theta:.2f}  →  prior E[θ]={prior_mean:.6f}")
            print(f"  burn_in={self.burn_in}, xi_ema={self.xi_ema}")
            print(f"  min_K={self._min_clusters}, "
                  f"min_cluster_size={self._min_cluster_size:.1f}")

        # K-means warm start
        self.r = self._init_responsibilities_kmeans(X, random_state)

        # SSL: start xi_post near the prior mean (very sparse)
        prior_mean_xi  = 1.0 / (1.0 + self.beta_theta)
        init_xi        = max(prior_mean_xi, 1e-3)   # at least 1e-3
        self.xi_post   = np.full(self.S, np.clip(init_xi, _NS.EPS, 1.0 - _NS.EPS))
        self.theta_ssl = np.clip(prior_mean_xi, _NS.EPS, 1.0 - _NS.EPS)

        # Stick-breaking
        self.theta       = np.ones(self.K)
        self.theta_prime = np.ones(self.K) * self.nu

        # Cluster Dirichlet from k-means statistics (NO xi gating yet)
        self.lambda_star = np.zeros((self.K, self.S))
        for k in range(self.K):
            mask = self.r[:, k] > 0.1
            if mask.sum() > 0:
                mu = X[mask].mean(axis=0) + 0.5
                self.lambda_star[k] = mu / mu.sum() * (self.zeta * self.S)
            else:
                self.lambda_star[k] = np.full(self.S, self.zeta)
        self.lambda_star = np.maximum(self.lambda_star, 0.1)

        # Background Dirichlet from marginal
        mu_all = X.mean(axis=0) + 0.5
        self.iota_star = mu_all / mu_all.sum() * (self.eta * self.S)
        self.iota_star = np.maximum(self.iota_star, 0.1)

        self._cache         = {}
        self._null_baseline = None   # frozen at burn-in end; reset each fit()

    def _init_responsibilities_kmeans(self, X, random_state):
        N = X.shape[0]
        if self.K == 1:
            return np.ones((N, 1))
        try:
            X_log  = np.log1p(X)
            kmeans = cluster.KMeans(
                n_clusters=min(self.K, N),
                n_init=10, max_iter=200,
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
    # Cached expectations
    # ─────────────────────────────────────────────────────────────────────────

    def _E_log_alpha(self):
        if 'Ela' not in self._cache:
            s = self.lambda_star.sum(axis=1, keepdims=True)
            self._cache['Ela'] = digamma(self.lambda_star) - digamma(s)
        return self._cache['Ela']   # (K, S)

    def _E_log_beta(self):
        if 'Elb' not in self._cache:
            self._cache['Elb'] = (digamma(self.iota_star)
                                  - digamma(self.iota_star.sum()))
        return self._cache['Elb']   # (S,)

    def _E_log_pi(self):
        if 'Elpi' not in self._cache:
            st   = self.theta + self.theta_prime
            elg  = digamma(self.theta)       - digamma(st)
            el1g = digamma(self.theta_prime) - digamma(st)
            cum  = np.concatenate([[0.0], np.cumsum(el1g[:-1])])
            self._cache['Elpi'] = elg + cum
        return self._cache['Elpi']   # (K,)

    def _clear_cache(self):
        self._cache = {}

    # ─────────────────────────────────────────────────────────────────────────
    # SSL core: discriminative signal G_excess_j
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_discriminative_signal(self):
        """
        G_excess_j: baseline-corrected joint discriminative signal.

        G_raw_j    = Σ_k r̄_k · |E[log α_kj] - E[log β_j]|
        G_excess_j = max(G_raw_j - _null_baseline, 0)

        The null baseline is computed ONCE at the end of burn-in and then
        FROZEN for all subsequent SSL-ON iterations. This prevents the
        baseline from shifting each iteration, which was the root cause of
        oscillation in the selected-feature count and non-convergence.

        Returns
        -------
        G_excess : (S,) non-negative array, near-zero for noise features.
        G_raw    : (S,) raw signal (for diagnostics).
        """
        E_log_a = self._E_log_alpha()   # (K, S)
        E_log_b = self._E_log_beta()    # (S,)
        r_bar   = self.r.mean(axis=0)   # (K,)

        # Raw joint signal: cluster-weighted |log α_kj - log β_j|
        G_raw = r_bar @ np.abs(E_log_a - E_log_b[None, :])   # (S,)

        # Use the frozen baseline set at burn-in end; fall back to adaptive
        # if not yet set (during burn-in itself, baseline doesn't matter
        # because xi_post is not used for gating yet).
        if self._null_baseline is None:
            # Adaptive estimate during burn-in (not used for gating)
            frac_signal   = float(np.clip(self.theta_ssl * 3.0, 0.05, 0.40))
            q_null        = float(np.clip(100.0 * (1.0 - frac_signal), 50.0, 90.0))
            null_baseline = float(np.percentile(G_raw, q_null))
        else:
            null_baseline = self._null_baseline   # frozen after burn-in

        G_excess = np.maximum(G_raw - null_baseline, 0.0)
        return G_excess, G_raw

    # ─────────────────────────────────────────────────────────────────────────
    # CAVI update steps
    # ─────────────────────────────────────────────────────────────────────────

    def _expected_log_lik(self, X, use_xi=True):
        """
        E_q[log p(x_i | z_i=k)]:
          = Σ_j x_ij · [ξ*_j·E[log α_kj] + (1-ξ*_j)·E[log β_j]]  if use_xi
          = Σ_j x_ij · E[log α_kj]                                  if not use_xi

        use_xi=False during burn-in so clustering is not corrupted by
        poorly-initialised ξ*.
        Shape: (N, K)
        """
        E_log_a = self._E_log_alpha()   # (K, S)
        if use_xi:
            E_log_b = self._E_log_beta()   # (S,)
            xi      = self.xi_post         # (S,)
            combined = xi[None, :] * E_log_a + (1.0 - xi)[None, :] * E_log_b[None, :]
        else:
            combined = E_log_a   # (K, S): use full cluster-specific log-probs
        return X @ combined.T   # (N, K)

    def _update_r(self, X, use_xi=True):
        E_log_pi = self._E_log_pi()
        ll       = self._expected_log_lik(X, use_xi=use_xi)
        log_r    = E_log_pi[None, :] + ll
        log_r   -= logsumexp(log_r, axis=1, keepdims=True)
        self.r   = np.exp(log_r)
        self.r   = np.maximum(self.r, _NS.EPS)
        self.r  /= self.r.sum(axis=1, keepdims=True)

    def _update_xi_post(self):
        """
        SSL update: xi_post_j = sigmoid(log(λ₁/λ₀) + (λ₀-λ₁)·G_excess_j + log(θ/(1-θ))).

        The log-odds formula depends only on the FROZEN G_excess (via _null_baseline)
        and the current θ_ssl — neither of which depends on xi_post itself. This
        means the update is a direct closed-form solve (no oscillation possible):
          xi* = sigmoid(const_per_feature + log(θ/(1-θ)))
        The EMA is retained only for the pre-freeze warm-up phase where G_excess
        is still shifting; after freezing, xi_new is accepted directly (EMA=1).
        """
        G_excess, _ = self._compute_discriminative_signal()   # (S,)

        log_lambda_ratio = np.log(self.lambda1 / max(self.lambda0, _NS.EPS))
        laplace_reward   = (self.lambda0 - self.lambda1) * G_excess
        safe_theta       = np.clip(self.theta_ssl, _NS.EPS, 1.0 - _NS.EPS)
        prior_logodds    = np.log(safe_theta / (1.0 - safe_theta))

        log_odds = log_lambda_ratio + laplace_reward + prior_logodds
        log_odds = np.clip(log_odds, -500.0, 500.0)
        xi_new   = np.clip(expit(log_odds), _NS.EPS, 1.0 - _NS.EPS)

        if self._null_baseline is None:
            # Pre-freeze: use EMA to damp shifting-baseline noise
            alpha        = float(np.clip(self.xi_ema, _NS.EPS, 1.0))
            self.xi_post = alpha * xi_new + (1.0 - alpha) * self.xi_post
        else:
            # Post-freeze: baseline is fixed, log_odds is deterministic given θ_ssl.
            # Accept directly — no oscillation possible from this update alone.
            self.xi_post = xi_new

        self.xi_post = np.clip(self.xi_post, _NS.EPS, 1.0 - _NS.EPS)

    def _update_theta_ssl(self):
        """
        Exact Beta posterior: θ | ξ ~ Beta(1+Σξ*_j, βθ+S-Σξ*_j).
        E[θ|ξ] = (1+Σξ*_j) / (1+βθ+S).
        """
        sum_xi         = float(self.xi_post.sum())
        a_post         = 1.0 + sum_xi
        b_post         = self.beta_theta + self.S - sum_xi
        self.theta_ssl = np.clip(a_post / (a_post + b_post), _NS.EPS, 1.0 - _NS.EPS)

    def _update_theta(self):
        r_k              = self.r.sum(axis=0)
        self.theta       = np.maximum(1.0 + r_k, _NS.EPS)
        cumsum_r         = np.cumsum(r_k)
        self.theta_prime = np.maximum(
            self.nu + (cumsum_r[-1] - cumsum_r), _NS.EPS
        )
        self.theta_prime[-1] = max(self.nu, _NS.EPS)

    def _update_lambda_star(self, X):
        """
        λ*_kj = ζ + Σ_i r_ik · x_ij   (always ungated)

        We intentionally do NOT gate lambda_star by xi_post. Gating creates
        a feedback collapse: low xi → poor cluster Dirichlet estimates →
        low G_raw → even lower xi → full collapse. By keeping lambda_star
        always updated from all features, clusters remain well-estimated
        regardless of the SSL selection state, and G_raw stays informative.
        The SSL selection only influences WHICH features drive assignment
        (via _expected_log_lik) and background modelling (via iota_star).
        """
        self.lambda_star = np.maximum(self.zeta + self.r.T @ X, 0.1)

    def _update_iota_star(self, X, use_xi=True):
        """
        ι*_j = η + Σ_i (1-ξ*_j)·x_ij  (after burn-in)
        ι*_j = η + Σ_i x_ij            (burn-in)
        """
        if use_xi:
            xi = self.xi_post
            self.iota_star = np.maximum(
                self.eta + ((1.0 - xi)[None, :] * X).sum(axis=0), 0.1
            )
        else:
            self.iota_star = np.maximum(self.eta + X.sum(axis=0), 0.1)

    # ─────────────────────────────────────────────────────────────────────────
    # ELBO computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_elbo(self, X):
        """
        ELBO = E_q[log p(X|Z,α,β,ξ)] + E_q[log p(Z|π)] - H_q[Z]
              + E_q[log p(ξ|θ)] - H_q[ξ] + E_q[log p(θ)]
        """
        EPS      = _NS.EPS
        xi       = self.xi_post
        E_log_pi = self._E_log_pi()
        ll       = self._expected_log_lik(X, use_xi=True)

        elbo  = float(np.sum(self.r * (E_log_pi[None, :] + ll)))
        elbo -= float(np.sum(self.r * np.log(np.maximum(self.r, EPS))))

        safe_theta = np.clip(self.theta_ssl, EPS, 1.0 - EPS)
        elbo += float(np.sum(
            xi * np.log(safe_theta) + (1.0 - xi) * np.log(1.0 - safe_theta)
        ))

        safe_xi = np.clip(xi, EPS, 1.0 - EPS)
        elbo += float(np.sum(
            -safe_xi * np.log(safe_xi)
            - (1.0 - safe_xi) * np.log(1.0 - safe_xi)
        ))

        elbo += float((self.beta_theta - 1.0) * np.log(1.0 - safe_theta))
        return elbo

    # ─────────────────────────────────────────────────────────────────────────
    # Pruning
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_weights(self):
        E_gamma  = self.theta / (self.theta + self.theta_prime)
        log_w    = _NS.log(E_gamma)
        log_1mg  = _NS.log(1 - E_gamma)
        log_w   += np.concatenate([[0.0], np.cumsum(log_1mg[:-1])])
        return np.exp(log_w)

    def _prune_empty_clusters(self):
        weights       = self._compute_weights()
        cluster_sizes = self.r.sum(axis=0)

        keep = ((weights > self.prune_threshold) &
                (cluster_sizes > self._min_cluster_size))

        n_keep = keep.sum()
        if n_keep < self._min_clusters:
            top_idx       = np.argsort(weights)[-self._min_clusters:]
            keep          = np.zeros(self.K, dtype=bool)
            keep[top_idx] = True
            n_keep        = self._min_clusters

        if n_keep < self.K:
            if self.verbose >= 1:
                print(f"  Pruning: {self.K} → {n_keep} clusters "
                      f"(removed {self.K - n_keep}; min_K={self._min_clusters})")
            self.K            = n_keep
            self.r            = self.r[:, keep]
            self.theta        = self.theta[keep]
            self.theta_prime  = self.theta_prime[keep]
            self.lambda_star  = self.lambda_star[keep]
            self.r           /= self.r.sum(axis=1, keepdims=True)
            self._clear_cache()
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Main fit loop
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, X):
        """
        Fit via CAVI with burn-in phase and EMA-stabilised SSL selection.

        Parameters
        ──────────
        X : array-like, shape (N, S)   count data matrix

        Returns
        ───────
        self
        """
        X            = check_array(X, dtype=np.float64)
        random_state = check_random_state(self.random_state)
        self._initialize_parameters(X, random_state)

        if self.verbose >= 1:
            print(f"\nStarting CAVI (DMM-SVVS v6.1 — Joint-SSL Feature Selection)")
            print("=" * 70)

        t0 = time()

        # xi_post running average: accumulate after burn-in to average out any
        # residual period-2 limit cycles in the CAVI. After ssl_avg_start iters
        # of SSL-ON, we declare convergence and set xi_post to the time average.
        # This is the correct Bayesian interpretation: the ergodic average of the
        # CAVI trajectory estimates the posterior expectation even if individual
        # iterates oscillate.
        _xi_sum       = np.zeros(self.S)
        _xi_count     = 0
        ssl_avg_start = self.burn_in + 10   # start averaging 10 iters after SSL-ON

        # Baseline freeze: computed once after ssl_avg_start, held fixed.
        ssl_warmup = 10   # iters of SSL-gated updates before freezing baseline

        # θ_ssl window for convergence: check relative change over last window
        _theta_window = []
        _conv_window  = 10   # number of SSL iters to average θ_ssl over

        for iteration in range(1, self.max_iter + 1):
            self.n_iter = iteration
            self._clear_cache()

            ssl_active = (iteration > self.burn_in)

            # Freeze the null baseline after ssl_warmup SSL-ON iterations.
            # By this point lambda_star has been updated under SSL gating,
            # so G_raw is at its operational scale.
            if ssl_active and self._null_baseline is None:
                iters_since_ssl = iteration - self.burn_in
                if iters_since_ssl >= ssl_warmup:
                    _, G_raw_init       = self._compute_discriminative_signal()
                    frac_signal         = float(np.clip(self.theta_ssl * 3.0, 0.05, 0.40))
                    q_null              = float(np.clip(100.0 * (1.0 - frac_signal), 50.0, 90.0))
                    self._null_baseline = float(np.percentile(G_raw_init, q_null))
                    if self.verbose >= 1:
                        print(f"  [SSL-ON] Null baseline frozen at iter {iteration}: "
                              f"{self._null_baseline:.4f} (q={q_null:.0f}th pct)")

            # ── CAVI steps ────────────────────────────────────────────────────
            self._update_r(X, use_xi=ssl_active)
            self._update_xi_post()
            self._update_theta_ssl()
            self._update_theta()
            self._update_lambda_star(X)
            self._update_iota_star(X, use_xi=ssl_active)

            # Pruning (only after burn-in, before averaging starts)
            if (ssl_active
                    and iteration < ssl_avg_start
                    and iteration >= self.prune_start
                    and iteration % self.prune_every == 0):
                pruned = self._prune_empty_clusters()
                if pruned:
                    # Reset xi running sum after structural change
                    _xi_sum   = np.zeros(self.S)
                    _xi_count = 0

            # ── Accumulate xi_post running average ────────────────────────────
            # After ssl_avg_start we average xi_post across iterations to get
            # a stable estimate that is immune to period-2 limit cycles.
            if iteration >= ssl_avg_start:
                _xi_sum   += self.xi_post
                _xi_count += 1
                _theta_window.append(float(self.theta_ssl))

            # ── Logging every 10 iters ────────────────────────────────────────
            if iteration % 10 == 0:
                elbo  = self._compute_elbo(X)
                self.elbo_history.append(elbo)
                # Report the running average xi for display (if available)
                if _xi_count > 0:
                    xi_avg = _xi_sum / _xi_count
                    n_sel  = int((xi_avg > 0.5).sum())
                else:
                    n_sel  = int((self.xi_post > 0.5).sum())
                _, G_raw = self._compute_discriminative_signal()
                G_max    = float(G_raw.max())

                phase = "BURN-IN" if not ssl_active else "SSL-ON"
                if self.verbose >= 1:
                    print(f"Iter {iteration:4d} [{phase}]: "
                          f"ELBO={elbo:14.2f}  K={self.K}  "
                          f"selected={n_sel}/{self.S}  "
                          f"θ_ssl={self.theta_ssl:.5f}  "
                          f"G_max={G_max:.3f}  t={time()-t0:.1f}s")

            # ── Convergence: check once we have enough averaged samples ────────
            # Declare convergence when the time-averaged θ_ssl is stable over
            # _conv_window SSL iterations. The averaged xi is then our final answer.
            if _xi_count >= _conv_window and _xi_count % 5 == 0:
                theta_avg_new = float(np.mean(_theta_window[-_conv_window:]))
                theta_avg_old = float(np.mean(_theta_window[-2*_conv_window:-_conv_window])) \
                                if len(_theta_window) >= 2 * _conv_window else None
                if theta_avg_old is not None:
                    theta_change = abs(theta_avg_new - theta_avg_old) / (abs(theta_avg_old) + 1e-12)
                    if theta_change < self.tol:
                        self.converged = True
                        if self.verbose >= 1:
                            if iteration % 10 != 0:
                                elbo = self._compute_elbo(X)
                                self.elbo_history.append(elbo)
                            xi_avg = _xi_sum / _xi_count
                            n_sel_avg = int((xi_avg > 0.5).sum())
                            print(f"\nConverged at iteration {iteration} "
                                  f"(avg θ_ssl change={theta_change:.2e}, "
                                  f"avg selected={n_sel_avg}).")
                        break

        # ── Set xi_post to time average (eliminates limit-cycle noise) ────────
        if _xi_count > 0:
            self.xi_post = np.clip(_xi_sum / _xi_count, _NS.EPS, 1.0 - _NS.EPS)

        # Final pruning and bookkeeping
        self._prune_empty_clusters()
        self.weights_           = self._compute_weights()
        self.selected_features_ = np.where(self.xi_post > 0.5)[0]
        self.xi_post_           = self.xi_post.copy()

        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print(f"Final Results (DMM-SVVS v6.1):")
            print(f"  Clusters (K)       : {self.K}")
            print(f"  Mixing weights     : {np.round(self.weights_[:min(self.K,6)], 4)}")
            print(f"  Selected features  : {len(self.selected_features_)} / {self.S}")
            print(f"  θ_ssl (slab rate)  : {self.theta_ssl:.6f}")
            print(f"  βθ (sparsity prior): {self.beta_theta:.2f}")
            print(f"  ELBO (final)       : "
                  f"{self.elbo_history[-1] if self.elbo_history else 'N/A':.2f}")
            print(f"  Converged          : {self.converged}")
            print(f"  Total time         : {time()-t0:.2f}s")

        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, X):
        """Return MAP cluster labels for X using fitted variational parameters."""
        X      = check_array(X, dtype=np.float64)
        N_new  = X.shape[0]
        r_save = self.r.copy()
        N_save = self.N

        self.N = N_new
        self.r = np.zeros((N_new, self.K))
        self._clear_cache()
        self._update_r(X, use_xi=True)
        labels = self.r.argmax(axis=1)

        self.r   = r_save
        self.N   = N_save
        self._clear_cache()
        return labels

    # ─────────────────────────────────────────────────────────────────────────
    # Inspection and reporting
    # ─────────────────────────────────────────────────────────────────────────

    def get_selected_features(self, threshold=0.5):
        """
        Return features with ξ*_j > threshold.

        Returns dict with 'indices', 'xi_post', 'discriminative_G' (raw G_raw values).
        """
        _, G_raw = self._compute_discriminative_signal()
        idx = np.where(self.xi_post > threshold)[0]
        return {
            'indices'          : idx.tolist(),
            'xi_post'          : self.xi_post[idx].tolist(),
            'discriminative_G' : G_raw[idx].tolist(),
        }

    def get_cluster_signatures(self, threshold=0.5):
        """Per-cluster Dirichlet parameters restricted to selected features."""
        _, G_raw  = self._compute_discriminative_signal()
        important = np.where(self.xi_post > threshold)[0]
        signatures = {}
        for k in range(self.K):
            mask = self.r[:, k] > 0.5
            if mask.sum() > 0 and len(important) > 0:
                lam_sum = self.lambda_star[k, important].sum()
                alpha_k = (self.lambda_star[k, important] / lam_sum
                           if lam_sum > 0
                           else np.ones(len(important)) / len(important))
                signatures[k] = {
                    'features'             : important.tolist(),
                    'xi_post'              : self.xi_post[important].tolist(),
                    'discriminative_G'     : G_raw[important].tolist(),
                    'cluster_alpha_normed' : alpha_k.tolist(),
                    'n_samples'            : int(mask.sum()),
                    'mixing_weight'        : float(self.weights_[k]),
                }
        return signatures

    def summary(self):
        print("\n" + "=" * 65)
        print("  DMM-SVVS v6.1  —  Spike-and-Slab LASSO Feature Selection")
        print("  (Yao, Xie & Xu, JMLR 2025 — Joint-SSL for DMM)")
        print("=" * 65)
        print(f"  Samples (N)        : {self.N}")
        print(f"  Features (S)       : {self.S}")
        print(f"  Clusters (K)       : {self.K}")
        print(f"  Selected features  : {len(self.selected_features_)}")
        print(f"  θ_ssl (slab rate)  : {self.theta_ssl:.6f}")
        print(f"  λ₀ (spike rate)    : {self.lambda0}")
        print(f"  λ₁ (slab  rate)    : {self.lambda1}")
        print(f"  κ                  : {self.kappa}")
        print(f"  βθ                 : {self.beta_theta:.2f}")
        n_sel = len(self.selected_features_)
        print(f"  Sparsity           : {1 - n_sel/self.S:.3f} "
              f"({self.S - n_sel}/{self.S} noise features)")
        print(f"  ELBO (final)       : "
              f"{self.elbo_history[-1] if self.elbo_history else 'N/A':.2f}")
        print(f"  Converged          : {self.converged}")
        print(f"  Iterations         : {self.n_iter}")
        print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print("=" * 70)
    print("DMM-SVVS v6.1 — Joint-SSL Feature Selection (Yao et al. 2025)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    N, S, K_true, s = 150, 200, 3, 20

    print(f"\nSimulation: N={N}, S={S}, K_true={K_true}, s={s} discriminative")
    print(f"Sparsity: {s}/{S} = {s/S:.1%}")

    # Sparse Dirichlet structure: each cluster dominates a distinct feature block
    alpha = np.full((K_true, S), 0.1)
    block = s // K_true
    for k in range(K_true):
        alpha[k, k*block:(k+1)*block] = 5.0

    true_labels = rng.choice(K_true, size=N)
    X = np.array([
        rng.multinomial(3000, rng.dirichlet(alpha[true_labels[i]]))
        for i in range(N)
    ], dtype=float)

    print(f"Data shape: {X.shape}")
    print(f"True discriminative features: 0 to {s-1}")

    model = DMM_SVVS_Variational_v6_1(
        K_max    = 8,
        nu       = 'auto',
        zeta     = 1.0,
        eta      = 1.0,
        lambda0  = 100.0,
        lambda1  = 1.0,
        kappa    = 0.1,
        xi_ema   = 0.5,
        burn_in  = 20,
        max_iter = 300,
        tol      = 1e-3,
        verbose  = 1,
        random_state = 42
    )
    model.fit(X)

    pred = model.predict(X)
    ari  = adjusted_rand_score(true_labels, pred)
    nmi  = normalized_mutual_info_score(true_labels, pred)

    print(f"\n{'─'*50}")
    print(f"Clustering Performance:")
    print(f"  ARI = {ari:.3f}")
    print(f"  NMI = {nmi:.3f}")
    print(f"  K estimated = {model.K}  (true K = {K_true})")

    sel       = model.get_selected_features(threshold=0.5)
    true_disc = set(range(s))
    found     = set(sel['indices'])
    tp = len(found & true_disc)
    fp = len(found - true_disc)
    fn = len(true_disc - found)
    pr = tp / max(len(found), 1)
    rc = tp / max(len(true_disc), 1)
    f1 = 2*pr*rc / max(pr+rc, _NS.EPS)

    print(f"\nFeature Selection (SSL Joint-Sparsity):")
    print(f"  True discriminative : {len(true_disc)} features (indices 0-{s-1})")
    print(f"  SSL selected        : {len(found)} features")
    print(f"  True Positives      : {tp}")
    print(f"  False Positives     : {fp}")
    print(f"  False Negatives     : {fn}")
    print(f"  Precision           : {pr:.3f}")
    print(f"  Recall              : {rc:.3f}")
    print(f"  F1-score            : {f1:.3f}")

    top10_idx = np.argsort(model.xi_post)[::-1][:10]
    _, G_raw  = model._compute_discriminative_signal()
    print(f"\nTop-10 features by SSL slab probability (ξ*_j):")
    print(f"  {'Feature':>8}  {'ξ*_j':>8}  {'G_raw':>8}  {'True?':>6}")
    print(f"  {'─'*40}")
    for idx in top10_idx:
        print(f"  {idx:>8}  {model.xi_post[idx]:>8.4f}  "
              f"{G_raw[idx]:>8.4f}  {'✓' if idx in true_disc else '✗':>6}")

    model.summary()
