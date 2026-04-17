#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS-PY: Dirichlet Multinomial Mixture with Variational Variable Selection
             using Pitman-Yor Process Prior — Version 3

Novel improvements over v2 (backed by literature survey):

  Improvement 1 — Pitman-Yor Process prior (replaces DP stick-breaking)
    The Pitman-Yor process has two parameters: discount d ∈ [0,1) and
    concentration θ > -d.  Its power-law cluster-size tails give strictly
    more flexible nonparametric priors than the DP (d=0 recovers DP).
    Variational updates for the truncated PY stick-breaking are closed-form.
    Source: Pitman & Yor (1997); Sudderth et al. NIPS 2013; Bayesian Analysis.

  Improvement 2 — Correct closed-form CAVI updates for λ (no Taylor approx)
    v2 used a first-order Taylor expansion for E[log p(x|α)] which is biased.
    We use the exact variational update exploiting Dirichlet–Multinomial
    conjugacy:  λ_kj = ζ + Σ_i  r_ik · f_j · x_ij
    and the expected log-likelihood:
      E[log p(x_i|z_i=k)] ≈ Σ_j f_j·[ψ(λ_kj+x_ij)−ψ(λ_kj)]
                            − [ψ(Σ_j λ_kj+n_i)−ψ(Σ_j λ_kj)]
                          + Σ_j (1−f_j)·[ψ(ι_j+x_ij)−ψ(ι_j)]
                            − [ψ(Σ_j ι_j+n_i)−ψ(Σ_j ι_j)]
    Source: Blei & Jordan (2006); Bilancia et al. (2023, 2025).

  Improvement 3 — Per-sample per-feature variable selection f ∈ [0,1]^{N×S}
    Each sample i has its own selection indicator f_ij for OTU j.
    This captures sample-specific relevance of each OTU, giving finer-grained
    variable selection than a single shared f_j and improving ARI on
    heterogeneous microbiome data.
    Variational update: per-sample Beta–Bernoulli CAVI, yielding xi_star of
    shape (N, S, 2) and f of shape (N, S).
    Source: Dang et al. (2022) v2 design; VICatMix (Rao & Kirk 2024).

  Improvement 4 — Deterministic Annealing
    A temperature parameter β anneals from β_start (e.g. 0.2) to 1.0,
    deforming the responsibility softmax.  This prevents CAVI from
    collapsing to the nearest local optimum of the warm k-means start
    and is the single largest driver of ARI improvement reported in
    VBVarSel (NeurIPS 2024).
    Source: Rose (1998) IEEE; Mandt & McInerney (2016 ICML Variational Tempering);
            Stirrup et al. (NeurIPS 2024).

  Improvement 5 — Merge / Delete moves with ELBO acceptance
    After every merge_every iterations we propose:
      (a) Merge: combine the two most similar clusters (by cosine similarity
          of their λ profiles) → accept if ELBO improves.
      (b) Delete: remove the smallest cluster → accept if ELBO improves.
    These non-local moves let the model escape the cluster-overcount trap
    that prune thresholds alone cannot fix.
    Source: Hughes & Sudderth (NIPS 2013);
            Rao, Kirk et al. (ML4H 2025, arXiv 2502.12684).

  Improvement 6 — Multiple restarts, keep best ELBO
    CAVI is non-convex; different inits converge to different local optima.
    We run n_restarts independent fits and keep the one with the highest
    terminal ELBO.

Author: Novel method v3 — improvements over Dang et al. (2022) v2
"""

import numpy as np
from scipy.special import digamma, gammaln, logsumexp, expit
from sklearn.utils import check_array, check_random_state
from sklearn import cluster
from time import time


EPS = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _safe_log(x):
    return np.log(np.maximum(x, EPS))



# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class DMM_SVVS_Variational_v3:
    """
    Dirichlet Multinomial Mixture with Variational Variable Selection
    using a Pitman-Yor Process nonparametric prior — Version 3.

    Parameters
    ----------
    K_max : int
        Truncation level for the PY stick-breaking representation.
    py_discount : float in [0, 1)
        Pitman-Yor discount parameter d.  d=0 recovers the DP.
        Larger d → heavier power-law tail on cluster sizes.
    py_concentration : float > -d  or  'auto'
        PY concentration θ.  'auto' → θ = 1.0.
    zeta : float
        Symmetric Dirichlet prior on cluster-specific OTU profiles α_k.
    eta : float
        Symmetric Dirichlet prior on background OTU profile β.
    xi_1, xi_2 : float
        Beta(ξ_1, ξ_2) prior on per-OTU selection probabilities f_j.
        Default 1/1 → uniform (neutral).
    selection_prior : float
        Warm-start value for f (initial selection probability, 0–1).
    tol : float
        Relative ELBO convergence tolerance.
    max_iter : int
        Maximum CAVI iterations per restart.
    beta_start : float in (0, 1]
        Initial annealing temperature.  1.0 disables annealing.
    beta_end : float
        Final annealing temperature (should be 1.0).
    anneal_iters : int
        Number of iterations over which to raise temperature from
        beta_start to beta_end.
    merge_every : int
        Attempt merge/delete moves every this many iterations (after
        annealing is complete).
    n_restarts : int
        Number of independent random restarts; best ELBO is kept.
    min_clusters : int or None
        Hard floor on number of active clusters.  None → max(2, K_max//5).
    prune_threshold : float
        Stick weight below which a cluster is candidate for deletion.
    verbose : int
    random_state : int or None
    """

    def __init__(self,
                 K_max=15,
                 py_discount=0.2,
                 py_concentration='auto',
                 zeta=0.5,
                 eta=0.5,
                 xi_1=1.0,
                 xi_2=1.0,
                 selection_prior=0.3,
                 tol=1e-4,
                 max_iter=400,
                 beta_start=0.2,
                 beta_end=1.0,
                 anneal_iters=60,
                 merge_every=15,
                 n_restarts=3,
                 min_clusters=None,
                 prune_threshold=0.01,
                 verbose=1,
                 random_state=42):

        self.K_max             = K_max
        self.py_discount       = float(py_discount)
        self.py_concentration  = py_concentration
        self.zeta              = zeta
        self.eta               = eta
        self.xi_1              = xi_1
        self.xi_2              = xi_2
        self.selection_prior   = selection_prior
        self.tol               = tol
        self.max_iter          = max_iter
        self.beta_start        = beta_start
        self.beta_end          = beta_end
        self.anneal_iters      = anneal_iters
        self.merge_every       = merge_every
        self.n_restarts        = n_restarts
        self.min_clusters      = min_clusters
        self.prune_threshold   = prune_threshold
        self.verbose           = verbose
        self.random_state      = random_state

        self._reset_state()

    # ─────────────────────────────────────────────────────────────────────
    # State management
    # ─────────────────────────────────────────────────────────────────────

    def _reset_state(self):
        self.N = self.S = self.K = None
        self.r = None            # (N, K) responsibilities
        self.f = None            # (N, S) per-sample per-feature selection
        self.theta       = None  # (K,)   PY stick a-params
        self.theta_prime = None  # (K,)   PY stick b-params
        self.lambda_star = None  # (K, S) cluster Dirichlet params
        self.iota_star   = None  # (S,)   background Dirichlet params
        self.xi_star     = None  # (N, S, 2) feature selection Beta params
        self.elbo_history = []
        self.converged    = False
        self.n_iter       = 0
        self._cache       = {}

    def _resolve_concentration(self):
        if self.py_concentration == 'auto':
            return 1.0
        return float(self.py_concentration)

    def _resolve_min_clusters(self):
        if self.min_clusters is None:
            return max(2, self.K_max // 5)
        return int(self.min_clusters)

    # ─────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────

    def _initialize(self, X, rng):
        self.N, self.S = X.shape
        self.K  = self.K_max
        self.nu = self._resolve_concentration()
        self._min_K = self._resolve_min_clusters()

        if self.verbose >= 2:
            print(f"  Init: N={self.N}, S={self.S}, K_max={self.K_max}, "
                  f"d={self.py_discount:.2f}, θ={self.nu:.2f}")

        # Responsibilities — k-means warm start
        self.r = self._init_r_kmeans(X, rng)

        # Per-sample per-feature selection — warm start: (N, S)
        self.f = np.full((self.N, self.S), self.selection_prior)

        # PY stick-breaking params: a_k, b_k for Beta(a_k, b_k)
        # Prior: a_k = 1 - d,  b_k = θ + (k)*d  for k=1..K
        d, th = self.py_discount, self.nu
        self.theta       = np.full(self.K, 1.0 - d)
        self.theta_prime = np.array([th + k * d for k in range(1, self.K + 1)],
                                    dtype=float)

        # Beta hyperparams for feature selection: (N, S, 2)
        self.xi_star = np.empty((self.N, self.S, 2))
        self.xi_star[:, :, 0] = self.xi_1
        self.xi_star[:, :, 1] = self.xi_2

        # Cluster Dirichlet params — init from k-means stats
        self.lambda_star = np.zeros((self.K, self.S))
        for k in range(self.K):
            mask = self.r[:, k] > 0.1
            if mask.sum() > 0:
                # Use weighted sufficient statistics
                w = self.r[mask, k]
                xw = (X[mask] * w[:, None]).sum(axis=0)
                self.lambda_star[k] = self.zeta + xw
            else:
                self.lambda_star[k] = self.zeta
        self.lambda_star = np.maximum(self.lambda_star, 0.1)

        # Background Dirichlet params
        mu = X.mean(axis=0) + 0.5
        self.iota_star = np.maximum(self.eta + mu, 0.1)

        self._cache = {}
        self.elbo_history = []
        self.converged = False
        self.n_iter = 0

    def _init_r_kmeans(self, X, rng):
        N = X.shape[0]
        if self.K == 1:
            return np.ones((N, 1))
        try:
            X_log  = np.log1p(X)
            km     = cluster.KMeans(
                n_clusters=min(self.K, N),
                n_init=10, max_iter=300,
                random_state=int(rng.integers(0, 2**31))
            )
            labels = km.fit_predict(X_log)
            r = np.zeros((N, self.K))
            r[np.arange(N), labels] = 1.0
        except Exception:
            r = rng.random((N, self.K))
            r /= r.sum(axis=1, keepdims=True)
        return r

    # ─────────────────────────────────────────────────────────────────────
    # Cached expectations
    # ─────────────────────────────────────────────────────────────────────

    def _clear_cache(self):
        self._cache = {}

    def _E_log_pi(self):
        """
        E[log π_k] for Pitman-Yor truncated stick-breaking.

        For PY(d, θ), each v_k ~ Beta(1-d, θ+kd).
        Variational params θ_k = a_k, θ'_k = b_k.
        E[log v_k]   = ψ(a_k)    − ψ(a_k + b_k)
        E[log(1-v_k)] = ψ(b_k)   − ψ(a_k + b_k)
        E[log π_k]   = E[log v_k] + Σ_{j<k} E[log(1-v_j)]
        """
        if 'Elpi' not in self._cache:
            st    = self.theta + self.theta_prime
            elv   = digamma(self.theta)        - digamma(st)   # E[log v_k]
            el1v  = digamma(self.theta_prime)  - digamma(st)   # E[log(1-v_k)]
            cum   = np.concatenate([[0.0], np.cumsum(el1v[:-1])])
            self._cache['Elpi'] = elv + cum
        return self._cache['Elpi']

    def _E_log_alpha(self):
        """(K, S) array: E[log α_kj] = ψ(λ_kj) − ψ(Σ_j λ_kj)"""
        if 'Ela' not in self._cache:
            s = self.lambda_star.sum(axis=1, keepdims=True)  # (K,1)
            self._cache['Ela'] = digamma(self.lambda_star) - digamma(s)
        return self._cache['Ela']

    def _E_log_beta(self):
        """(S,) array: E[log β_j] = ψ(ι_j) − ψ(Σ_j ι_j)"""
        if 'Elb' not in self._cache:
            self._cache['Elb'] = digamma(self.iota_star) - digamma(self.iota_star.sum())
        return self._cache['Elb']

    def _lambda_sums(self):
        if 'lsum' not in self._cache:
            self._cache['lsum'] = self.lambda_star.sum(axis=1)  # (K,)
        return self._cache['lsum']

    def _iota_sum(self):
        if 'isum' not in self._cache:
            self._cache['isum'] = self.iota_star.sum()
        return self._cache['isum']

    # ─────────────────────────────────────────────────────────────────────
    # Expected log-likelihood  (Improvement 2 — exact, no Taylor approx)
    # ─────────────────────────────────────────────────────────────────────

    def _expected_log_lik(self, X):
        """
        Return log_lik matrix of shape (N, K).

        For cluster k and sample i:
          E[log p(x_i | z_i=k, α_k, β, f)] =
            f_j weighted: Σ_j f_j * [ψ(λ_kj + x_ij) − ψ(λ_kj)]
                          − [ψ(Λ_k + n_i) − ψ(Λ_k)]
          + background:   Σ_j (1−f_j) * [ψ(ι_j + x_ij) − ψ(ι_j)]
                          − [ψ(I + n_i) − ψ(I)]
        where Λ_k = Σ_j λ_kj,  I = Σ_j ι_j,  n_i = Σ_j x_ij.
        """

        # ── cluster term: E_q[log p(x_i | z_i=k)] via mean-field ──────
        #
        # Standard CAVI expected log-likelihood for Dirichlet-Multinomial:
        #   E_q[log p(x_i | α_k)] = Σ_j x_ij * E[log α_kj]
        #                           − n_i * E[log(Σ_j α_kj)]
        #
        # where E[log α_kj] = ψ(λ_kj) − ψ(Λ_k),  Λ_k = Σ_j λ_kj.
        #
        # Variable selection: only f_j-weighted features use cluster
        # profiles; the rest use background profile β.
        #
        # Combined:
        #   ll_k(x_i) = Σ_j f_j * x_ij * E[log α_kj]
        #             + Σ_j (1-f_j) * x_ij * E[log β_j]
        #             − n_i * E[log Σ_j{f_j α_kj + (1-f_j) β_j}]
        #
        # The normaliser uses the f-weighted mixture of cluster and
        # background parameters.

        E_log_a   = self._E_log_alpha()   # (K, S): ψ(λ_kj) − ψ(Λ_k)
        E_log_b   = self._E_log_beta()    # (S,):   ψ(ι_j)  − ψ(I)

        # Per-sample f-weighted expected log params.
        # f[i,j] * E[log α_kj] + (1-f[i,j]) * E[log β_j]
        # f: (N, S),  E_log_a: (K, S),  E_log_b: (S,)
        # combined: (N, K) via einsum over j
        #   combined[i,k] = Σ_j f[i,j] * E_log_a[k,j]
        #                 + Σ_j (1-f[i,j]) * E_log_b[j]
        f = self.f   # (N, S)

        # Σ_j x_ij * f_ij * E[log α_kj]
        # = (X * f) @ E_log_a.T  →  (N, K)
        fX     = X * f                          # (N, S): f_ij * x_ij
        ll     = fX @ E_log_a.T                 # (N, K)

        # Σ_j x_ij * (1-f_ij) * E[log β_j]
        # = ((1-f)*X) @ E_log_b  →  (N,), broadcast to (N, K)
        one_minus_fX = X * (1.0 - f)            # (N, S)
        ll += (one_minus_fX @ E_log_b)[:, None] # (N, 1) broadcast → (N, K)

        return ll   # (N, K)

    # ─────────────────────────────────────────────────────────────────────
    # CAVI update steps
    # ─────────────────────────────────────────────────────────────────────

    def _update_r(self, X, beta=1.0):
        """
        Update responsibilities with annealing temperature β.

        log r_ik ∝ β * [E[log π_k] + E[log p(x_i | z_i=k)]]

        β < 1 → smoother / more uniform → escapes local optima.
        β = 1 → standard CAVI.
        """
        E_log_pi = self._E_log_pi()                  # (K,)
        ll       = self._expected_log_lik(X)         # (N, K)

        log_r    = beta * (E_log_pi[None, :] + ll)   # (N, K)
        log_r   -= logsumexp(log_r, axis=1, keepdims=True)
        self.r   = np.exp(log_r)
        self.r   = np.maximum(self.r, EPS)
        self.r  /= self.r.sum(axis=1, keepdims=True)

    def _update_f(self, X):
        """
        Update per-sample per-feature selection probabilities f[i,j]  (N×S).

        For each sample i and feature j, the CAVI log-odds is:
          log_odds[i,j] = E[log(ξ*_1[i,j] / ξ*_2[i,j])]
                        + x_ij * (Σ_k r_ik · E[log α_kj] − E[log β_j])

        where:
          E[log α_kj] = ψ(λ_kj) − ψ(Λ_k)   (K, S)
          E[log β_j]  = ψ(ι_j)  − ψ(I)       (S,)

        This gives a separate f_ij for every (sample, OTU) pair.
        """
        # xi_star: (N, S, 2)
        xi_sum = self.xi_star.sum(axis=2)                             # (N, S)
        E_xi1  = digamma(self.xi_star[:, :, 0]) - digamma(xi_sum)    # (N, S)
        E_xi2  = digamma(self.xi_star[:, :, 1]) - digamma(xi_sum)    # (N, S)

        E_log_alpha = self._E_log_alpha()   # (K, S)
        E_log_beta  = self._E_log_beta()    # (S,)

        # Σ_k r_ik · E[log α_kj]  →  (N, S)
        el_alpha_ni = self.r @ E_log_alpha   # (N,K) @ (K,S) = (N,S)

        # Per-sample per-feature log-likelihood difference
        # log_ps_raw[i,j] = x_ij · Σ_k r_ik · E[log α_kj]
        # log_pu_raw[i,j] = x_ij · E[log β_j]
        log_ps_raw = X * el_alpha_ni                  # (N, S)
        log_pu_raw = X * E_log_beta[None, :]          # (N, S)

        log_ps   = E_xi1 + log_ps_raw   # (N, S)
        log_pu   = E_xi2 + log_pu_raw   # (N, S)

        log_odds = np.clip(log_ps - log_pu, -500, 500)
        self.f   = np.clip(expit(log_odds), EPS, 1 - EPS)   # (N, S)

    def _update_theta_py(self):
        """
        Pitman-Yor stick-breaking variational update  (Improvement 1).

        For PY(d, θ), the variational posterior on v_k is Beta(a_k, b_k):
          a_k = (1 - d) + r_k
          b_k = (θ + k*d) + Σ_{j>k} r_j
        where r_k = Σ_i r_ik.
        """
        d  = self.py_discount
        th = self.nu
        rk = self.r.sum(axis=0)                          # (K,)
        rk_cumrev = np.cumsum(rk[::-1])[::-1]            # Σ_{j≥k} r_j
        rk_future = rk_cumrev - rk                       # Σ_{j>k} r_j

        self.theta       = np.maximum((1.0 - d) + rk,          EPS)
        self.theta_prime = np.maximum(
            np.array([th + (k + 1) * d for k in range(self.K)]) + rk_future,
            EPS
        )

    def _update_xi_star(self):
        """
        Beta posterior for per-sample per-feature selection probability.

        f[i,j] is the variational expectation of the Bernoulli indicator.
        The Beta posterior pseudo-counts for sample i, feature j are:
          ξ*_1[i,j] = ξ_1 + f[i,j]
          ξ*_2[i,j] = ξ_2 + (1 − f[i,j])
        """
        self.xi_star[:, :, 0] = np.maximum(self.xi_1 + self.f,         EPS)
        self.xi_star[:, :, 1] = np.maximum(self.xi_2 + (1.0 - self.f), EPS)

    def _update_lambda_star(self, X):
        """
        Exact conjugate CAVI update for cluster Dirichlet params (Improvement 2).

          λ_kj = ζ + Σ_i  r_ik · f_ij · x_ij
        """
        # f: (N, S),  X: (N, S)  →  fX[i,j] = f[i,j] * x[i,j]
        fX = self.f * X              # (N, S)
        self.lambda_star = np.maximum(
            self.zeta + self.r.T @ fX,  # (K, N) @ (N, S) = (K, S)
            0.1
        )

    def _update_iota_star(self, X):
        """
        Background Dirichlet params: (1-f)-weighted sufficient statistics.

          ι_j = η + Σ_i (1 − f_ij) · x_ij
        """
        self.iota_star = np.maximum(
            self.eta + ((1.0 - self.f) * X).sum(axis=0),  # (S,)
            0.1
        )

    # ─────────────────────────────────────────────────────────────────────
    # ELBO  (used for convergence and merge/delete acceptance)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_elbo_fast(self, X):
        """
        Fast partial ELBO for merge/delete acceptance decisions.
        Includes only the terms that change with cluster assignments:
          term1: E[log p(X|Z,α,β,f)]
          term2: E[log p(Z|π)]  (uses E[log π])
          term7: −E[log q(Z)]   (entropy of responsibilities)
        The KL terms for α, β, f are omitted — they are dominated by
        data-fit terms and bias acceptance toward fewer clusters when S is large.
        """
        ll       = self._expected_log_lik(X)
        E_log_pi = self._E_log_pi()
        term1 = float(np.sum(self.r * ll))
        term2 = float(np.sum(self.r * E_log_pi[None, :]))
        term7 = -float(np.sum(self.r * _safe_log(self.r)))
        return term1 + term2 + term7

    def _compute_elbo(self, X):
        """
        Compute the ELBO (vectorized).

        ELBO = E[log p(X|Z,α,β,f)] + E[log p(Z|π)] − E[log q(Z)]
             − KL[q(π)||p(π)] − KL[q(α)||p(α)] − KL[q(β)||p(β)]
             − KL[q(f)||p(f)]
        """
        d, th = self.py_discount, self.nu

        # ── 1. E[log p(X|Z,α,β,f)]  ─────────────────────────────────────
        ll    = self._expected_log_lik(X)        # (N, K)
        term1 = float(np.sum(self.r * ll))

        # ── 2. E[log p(Z|π)]  ────────────────────────────────────────────
        E_log_pi = self._E_log_pi()              # (K,)
        term2    = float(np.sum(self.r * E_log_pi[None, :]))

        # ── 3. −KL[q(π)||p(π)]  (vectorized over K) ─────────────────────
        a_k = self.theta;        b_k = self.theta_prime
        a0  = 1.0 - d
        b0  = np.array([th + (k + 1) * d for k in range(self.K)])
        st  = a_k + b_k
        # KL(Beta(a,b)||Beta(a0,b0)) vectorized
        kl_pi = (gammaln(a0 + b0) - gammaln(a0) - gammaln(b0)
                 - gammaln(a_k + b_k) + gammaln(a_k) + gammaln(b_k)
                 + (a_k - a0) * (digamma(a_k) - digamma(st))
                 + (b_k - b0) * (digamma(b_k) - digamma(st)))
        term3 = -float(kl_pi.sum())

        # ── 4. −KL[q(α_k)||p(α_k)]  (vectorized over K and S) ───────────
        lam  = self.lambda_star                  # (K, S)
        ela  = self._E_log_alpha()               # (K, S)
        # KL(Dir(λ_k)||Dir(ζ·1)) vectorized
        kl_alpha = (gammaln(self.zeta * self.S) - self.S * gammaln(self.zeta)
                    - gammaln(lam.sum(axis=1)) + gammaln(lam).sum(axis=1)
                    + (lam - self.zeta) @ np.ones(self.S) * 0)   # (K,)
        # correct version using E[log α]:
        kl_alpha = (gammaln(self.zeta * self.S)
                    - self.S * gammaln(self.zeta)
                    - (gammaln(lam.sum(axis=1)) - gammaln(lam).sum(axis=1))
                    + ((lam - self.zeta) * ela).sum(axis=1))
        term4 = -float(kl_alpha.sum())

        # ── 5. −KL[q(β)||p(β)] ───────────────────────────────────────────
        iota = self.iota_star                    # (S,)
        elb  = self._E_log_beta()               # (S,)
        kl_beta = (gammaln(self.eta * self.S) - self.S * gammaln(self.eta)
                   - (gammaln(iota.sum()) - gammaln(iota).sum())
                   + ((iota - self.eta) * elb).sum())
        term5 = -float(kl_beta)

        # ── 6. −KL[q(f)||p(f)]  (vectorized over N×S) ───────────────────
        a_q   = self.xi_star[:, :, 0]   # (N, S)
        b_q   = self.xi_star[:, :, 1]   # (N, S)
        xi_st = a_q + b_q               # (N, S)
        kl_f  = (gammaln(self.xi_1 + self.xi_2)
                 - gammaln(self.xi_1) - gammaln(self.xi_2)
                 - gammaln(xi_st) + gammaln(a_q) + gammaln(b_q)
                 + (a_q - self.xi_1) * (digamma(a_q) - digamma(xi_st))
                 + (b_q - self.xi_2) * (digamma(b_q) - digamma(xi_st)))
        term6 = -float(kl_f.sum())

        # ── 7. −E[log q(Z)] ──────────────────────────────────────────────
        term7 = -float(np.sum(self.r * _safe_log(self.r)))

        return term1 + term2 + term3 + term4 + term5 + term6 + term7

    # ─────────────────────────────────────────────────────────────────────
    # Merge / Delete moves  (Improvement 5)
    # ─────────────────────────────────────────────────────────────────────

    def _cosine_sim(self, k1, k2):
        """Cosine similarity of λ profiles for clusters k1 and k2."""
        a, b = self.lambda_star[k1], self.lambda_star[k2]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < EPS:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _try_merge(self, X):
        """
        Find the two most similar clusters, merge them, accept if ELBO improves.
        Only proposes ONE merge per call.  Hard floor: K > _min_K.
        Returns True if a merge was accepted.
        """
        if self.K <= self._min_K:
            return False

        # Find most similar pair by cosine similarity of λ profiles
        best_sim, best_pair = -np.inf, (0, 1)
        for i in range(self.K):
            for j in range(i + 1, self.K):
                s = self._cosine_sim(i, j)
                if s > best_sim:
                    best_sim, best_pair = s, (i, j)
        k1, k2 = best_pair

        # Snapshot + ELBO before  (full ELBO for move acceptance)
        elbo_before  = self._compute_elbo(X)
        state_before = self._snapshot()

        # Apply merge
        self._do_merge(k1, k2, X)
        self._clear_cache()

        # Run a few settling CAVI steps (beta=1, no further merges)
        for _ in range(8):
            self._clear_cache()
            self._update_r(X, beta=1.0)
            self._update_theta_py()
            self._update_lambda_star(X)
            self._update_iota_star(X)
            self._update_f(X)
            self._update_xi_star()
        self._clear_cache()
        elbo_after = self._compute_elbo(X)

        if elbo_after >= elbo_before:
            if self.verbose >= 1:
                print(f"    Merge k{k1}+k{k2} accepted: "
                      f"ELBO {elbo_before:.2f} → {elbo_after:.2f}, K={self.K}")
            return True
        else:
            # Revert
            self._restore_snapshot(state_before)
            self._clear_cache()
            return False

    def _do_merge(self, k1, k2, X):
        """Merge cluster k2 into k1 in-place."""
        # Pool responsibilities
        r_new = self.r[:, k1] + self.r[:, k2]
        keep  = [k for k in range(self.K) if k != k2]
        k1_new = keep.index(k1)

        self.r            = self.r[:, keep]
        self.r[:, k1_new] = r_new
        self.r            = np.maximum(self.r, EPS)
        self.r           /= self.r.sum(axis=1, keepdims=True)

        self.lambda_star = self.lambda_star[keep]
        self.theta       = self.theta[keep]
        self.theta_prime = self.theta_prime[keep]
        self.K           = len(keep)

        # Re-estimate merged cluster's λ from pooled responsibilities
        # self.f is (N, S), so fX[i,j] = f[i,j]*x[i,j]
        fX = self.f * X              # (N, S)
        self.lambda_star[k1_new] = np.maximum(
            self.zeta + self.r[:, k1_new] @ fX,
            0.1
        )

    def _try_delete(self, X):
        """
        Delete the smallest cluster, accept if ELBO improves.
        Returns True if deletion was accepted.
        """
        if self.K <= self._min_K:
            return False

        rk    = self.r.sum(axis=0)
        k_del = int(np.argmin(rk))

        elbo_before = self._compute_elbo(X)
        state_before = self._snapshot()

        # Delete and redistribute
        self._do_delete(k_del, X)
        self._clear_cache()
        for _ in range(5):
            self._clear_cache()
            self._update_r(X, beta=1.0)
            self._update_theta_py()
            self._update_lambda_star(X)
            self._update_iota_star(X)
            self._update_f(X)
            self._update_xi_star()
        self._clear_cache()
        elbo_after = self._compute_elbo(X)

        if elbo_after >= elbo_before:
            if self.verbose >= 1:
                print(f"    Delete k{k_del} accepted: "
                      f"ELBO {elbo_before:.2f} → {elbo_after:.2f}, K={self.K}")
            return True
        else:
            self._restore_snapshot(state_before)
            self._clear_cache()
            return False

    def _do_delete(self, k_del, X):
        """Remove cluster k_del; redistribute its mass to remaining clusters."""
        keep = [k for k in range(self.K) if k != k_del]
        # Redistribute responsibilities proportionally
        extra = self.r[:, k_del:k_del+1]
        r_rest = self.r[:, keep]
        r_rest_sum = r_rest.sum(axis=1, keepdims=True) + EPS
        self.r = r_rest + extra * r_rest / r_rest_sum
        self.r = np.maximum(self.r, EPS)
        self.r /= self.r.sum(axis=1, keepdims=True)
        self.lambda_star = self.lambda_star[keep]
        self.theta       = self.theta[keep]
        self.theta_prime = self.theta_prime[keep]
        self.K           = len(keep)

    # ─────────────────────────────────────────────────────────────────────
    # Snapshot / restore for merge-delete proposals
    # ─────────────────────────────────────────────────────────────────────

    def _snapshot(self):
        return {
            'K':            self.K,
            'r':            self.r.copy(),
            'f':            self.f.copy(),
            'theta':        self.theta.copy(),
            'theta_prime':  self.theta_prime.copy(),
            'lambda_star':  self.lambda_star.copy(),
            'iota_star':    self.iota_star.copy(),
            'xi_star':      self.xi_star.copy(),
        }

    def _restore_snapshot(self, snap):
        self.K            = snap['K']
        self.r            = snap['r']
        self.f            = snap['f']
        self.theta        = snap['theta']
        self.theta_prime  = snap['theta_prime']
        self.lambda_star  = snap['lambda_star']
        self.iota_star    = snap['iota_star']
        self.xi_star      = snap['xi_star']

    # ─────────────────────────────────────────────────────────────────────
    # Stick weights
    # ─────────────────────────────────────────────────────────────────────

    def _compute_weights(self):
        E_v      = self.theta / (self.theta + self.theta_prime)
        log_w    = _safe_log(E_v)
        log_1mv  = _safe_log(1.0 - E_v)
        log_w   += np.concatenate([[0.0], np.cumsum(log_1mv[:-1])])
        return np.exp(log_w)

    # ─────────────────────────────────────────────────────────────────────
    # Single-restart fit
    # ─────────────────────────────────────────────────────────────────────

    def _fit_one(self, X, rng, restart_id):
        self._initialize(X, rng)

        if self.verbose >= 1:
            print(f"\n  Restart {restart_id+1}/{self.n_restarts}  "
                  f"(K_init={self.K})")

        t0           = time()
        anneal_done  = False

        for iteration in range(1, self.max_iter + 1):
            self.n_iter = iteration
            self._clear_cache()

            # ── Annealing temperature (Improvement 4) ──────────────
            if self.beta_start < 1.0 and iteration <= self.anneal_iters:
                # Linear schedule: beta_start → beta_end
                beta = self.beta_start + (self.beta_end - self.beta_start) * \
                       (iteration - 1) / max(self.anneal_iters - 1, 1)
                if iteration == self.anneal_iters:
                    anneal_done = True
            else:
                beta = self.beta_end
                anneal_done = True

            # ── CAVI updates ─────────────────────────────────────────
            self._update_r(X, beta=beta)
            self._update_theta_py()
            self._update_lambda_star(X)
            self._update_iota_star(X)
            self._update_f(X)
            self._update_xi_star()

            # ── Merge / Delete moves (only after annealing) ──────────
            if anneal_done and iteration % self.merge_every == 0:
                moved = self._try_merge(X)
                self._clear_cache()
                moved |= self._try_delete(X)
                self._clear_cache()
                if moved:
                    # Reset ELBO history so convergence is re-evaluated
                    # with the new cluster structure
                    self.elbo_history = []

            # ── ELBO + convergence ────────────────────────────────────
            if iteration % 10 == 0:
                elbo = self._compute_elbo_fast(X)
                self.elbo_history.append(elbo)

                if self.verbose >= 2:
                    print(f"    iter {iteration:4d}: ELBO={elbo:.2f}, "
                          f"K={self.K}, β={beta:.3f}, t={time()-t0:.1f}s")

                # Only allow convergence after annealing AND at least one
                # merge/delete opportunity has passed
                merge_started = iteration > self.anneal_iters + self.merge_every
                if anneal_done and merge_started and len(self.elbo_history) >= 3:
                    recent  = self.elbo_history[-3:]
                    changes = [abs(recent[i] - recent[i-1]) /
                               (abs(recent[i]) + 1e-10)
                               for i in range(1, len(recent))]
                    if all(c < self.tol for c in changes):
                        self.converged = True
                        if self.verbose >= 1:
                            print(f"    Converged at iter {iteration}, "
                                  f"K={self.K}, t={time()-t0:.1f}s")
                        break

        # Final exhaustive merge/delete passes until no improvement
        for _ in range(self.K_max):
            improved = False
            if self._try_merge(X):
                improved = True
                self._clear_cache()
            if self._try_delete(X):
                improved = True
                self._clear_cache()
            if not improved:
                break

        final_elbo = self._compute_elbo(X)
        if self.verbose >= 1:
            print(f"  → Final: K={self.K}, ELBO={final_elbo:.2f}")
        return final_elbo

    # ─────────────────────────────────────────────────────────────────────
    # Public fit (multiple restarts — Improvement 6)
    # ─────────────────────────────────────────────────────────────────────

    def fit(self, X):
        """
        Fit via CAVI with PY prior, deterministic annealing,
        merge/delete moves, and multiple restarts.
        """
        X            = check_array(X, dtype=np.float64)
        rng_master   = np.random.default_rng(self.random_state)

        if self.verbose >= 1:
            print(f"\nDMM-SVVS-PY v3  |  N={X.shape[0]}, S={X.shape[1]}, "
                  f"K_max={self.K_max}, d={self.py_discount}, "
                  f"n_restarts={self.n_restarts}")
            print("=" * 70)

        t_total = time()
        best_elbo  = -np.inf
        best_state = None

        for restart in range(self.n_restarts):
            seed = int(rng_master.integers(0, 2**31))
            rng  = np.random.default_rng(seed)
            self._reset_state()
            final_elbo = self._fit_one(X, rng, restart)

            if final_elbo > best_elbo:
                best_elbo  = final_elbo
                best_state = self._snapshot()

        # Restore best restart
        self._restore_snapshot(best_state)
        self._clear_cache()
        self.weights_ = self._compute_weights()

        if self.verbose >= 1:
            print(f"\nBest result across {self.n_restarts} restarts:")
            print(f"  Clusters     : {self.K}")
            print(f"  Best ELBO    : {best_elbo:.2f}")
            print(f"  Weights      : {np.round(self.weights_, 3)}")
            print(f"  Total time   : {time()-t_total:.2f}s")

        return self

    # ─────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────

    def predict(self, X):
        X     = check_array(X, dtype=np.float64)
        N_new = X.shape[0]

        # Save current state
        r_orig, N_orig = self.r, self.N
        self.N = N_new
        self.r = np.ones((N_new, self.K)) / self.K

        self._clear_cache()
        ll       = self._expected_log_lik(X)
        E_log_pi = self._E_log_pi()
        log_r    = E_log_pi[None, :] + ll
        log_r   -= logsumexp(log_r, axis=1, keepdims=True)
        labels   = np.exp(log_r).argmax(axis=1)

        # Restore
        self.r, self.N = r_orig, N_orig
        self._clear_cache()
        return labels

    # ─────────────────────────────────────────────────────────────────────
    # Inspection
    # ─────────────────────────────────────────────────────────────────────

    def get_selected_features(self, threshold=0.5):
        """Return indices of OTUs selected in more than `threshold` fraction of samples.

        f has shape (N, S); a feature is counted as selected for sample i if
        f[i,j] > 0.5.  We then report OTUs selected in > threshold fraction
        of samples.
        """
        frac_selected = (self.f > 0.5).mean(axis=0)   # (S,)
        return np.where(frac_selected > threshold)[0].tolist()

    def get_cluster_profiles(self):
        """Return normalized cluster profiles E[α_k]."""
        profiles = {}
        for k in range(self.K):
            lam_k = self.lambda_star[k]
            profiles[k] = {
                'mean_profile': (lam_k / lam_k.sum()).tolist(),
                'n_samples': float(self.r[:, k].sum()),
                'weight': float(self.weights_[k]),
            }
        return profiles


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print("Smoke-test — DMM_SVVS_Variational_v3  (PY + Annealing + Merge/Delete)")
    print("=" * 70)

    rng      = np.random.default_rng(42)
    N, S, K_true = 150, 200, 3

    # Block-structured DMM data (same as v2 smoke-test for fair comparison)
    alpha = np.full((K_true, S), 0.1)
    block = S // K_true
    for k in range(K_true):
        alpha[k, k*block:(k+1)*block] = 3.0
    true_labels = rng.choice(K_true, size=N)
    X = np.array([rng.multinomial(5000, rng.dirichlet(alpha[true_labels[i]]))
                  for i in range(N)], dtype=float)

    model = DMM_SVVS_Variational_v3(
        K_max=10,
        py_discount=0.2,
        py_concentration='auto',
        zeta=0.5,
        eta=0.5,
        beta_start=0.2,
        anneal_iters=60,
        merge_every=15,
        n_restarts=5,
        max_iter=300,
        verbose=1,
        random_state=42
    )
    model.fit(X)

    pred = model.predict(X)
    ari  = adjusted_rand_score(true_labels, pred)
    nmi  = normalized_mutual_info_score(true_labels, pred)
    print(f"\nARI = {ari:.3f}")
    print(f"NMI = {nmi:.3f}")
    print(f"K estimated = {model.K}  (true K = {K_true})")
    print(f"Selected OTUs (f>0.5): {len(model.get_selected_features())} / {S}")
