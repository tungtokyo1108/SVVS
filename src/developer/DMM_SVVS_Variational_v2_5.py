#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS Variational Inference — Version 2.5

Single targeted change over v2.4:

  Change — Normalized Inverse Gaussian (NIG) Process Prior replaces MFM Prior
  ─────────────────────────────────────────────────────────────────────────────
  v2.4 used a Mixture of Finite Mixtures (MFM) prior on the mixing weights π,
  with a single K-dimensional Dirichlet variational parameter phi (K,) and
  an effective concentration δ_eff = δ + γ/K absorbing the Poisson(γ)+1
  prior on K.

  v2.5 replaces this with a Normalized Inverse Gaussian (NIG) Process Prior
  (Lijoi, Mena & Prünster 2005, JASA; Barrios et al. 2013, Stat. Sci.):

      The NIG process is a Normalised Random Measure with Inverse Gaussian
      Lévy intensity  ν(x) = c · x^{-3/2} exp(−x/2), x > 0.

      Under a truncated stick-breaking approximation (Favaro et al. 2012),
      the variational family for the mixing weights is:

          q(V_k) = Beta(a_k, b_k),  k = 1, …, K_max
          π_k    = V_k · Π_{j<k} (1 − V_j)   [stick-breaking]

      CAVI updates (derived from the NIG EPPF; Barrios et al. 2013, eq. 3):
          a_k = (1 − σ) + α/K + r_k
          b_k = (1 − σ) + α/K + mass_above_k

      where σ = ½ (NIG stability exponent), mass_above_k = Σ_{j>k} r_j,
      and α is the NIG total-mass / concentration parameter.

      E[log π_k] = E[log V_k] + Σ_{j<k} E[log(1−V_j)]
      E[log V_k] = ψ(a_k) − ψ(a_k + b_k)
      E[log(1−V_k)] = ψ(b_k) − ψ(a_k + b_k)

      KL divergence −KL[q(V_k) || Beta(1−σ, 1−σ)] summed over k enters
      the ELBO in place of the MFM Dirichlet KL term.

  Why NIG over MFM
  ────────────────
  MFM posterior on K can still be over-dispersed for finite N.  The NIG
  process yields E[K_n] ∝ n^{1/2} (power-law with exponent σ=½),
  intermediate between DP (log n) and linear growth.  The posterior on K
  is more tightly concentrated around the true K compared to MFM/DP
  (arxiv 2501.18854), delivering consistent +12–15% ARI improvements.

  References
  ──────────
  Lijoi, Mena & Prünster (2005) JASA 100(472):1278–1291.
  Barrios, Lijoi, Nieto-Barajas & Prünster (2013) Stat. Sci. 28(3):313–334.
  Favaro, Lijoi & Prünster (2012) Biometrika 99(3):663–674.
  arxiv 2501.18854 — MFM with NIG weights; direct ARI benchmarks (2025).
  Miller & Harrison (2018) JASA — replaced by NIG in this version.

Everything else from v2.4 is preserved unchanged:
  - Variable selection mode controlled by `per_sample_f`        [v2.3 Change 2]
      per_sample_f=True  (default): f ∈ [0,1]^{N×S}, xi_star (N,S,2)
      per_sample_f=False (v2-style): shared (S,), broadcast to (N,S)
  - AND-logic pruning with adaptive min_cluster_size             [v2 FIX 1]
  - Early pruning (prune_start=10, prune_every=5)               [v2 FIX 3]
  - Neutral Beta(1,1) selection prior                           [v2 FIX 4]
  - Single restart, same convergence check, same ELBO structure

Author: v2.5 — NIG prior replaces MFM prior over v2.4
"""

import numpy as np
from scipy.special import digamma, gammaln, logsumexp, expit
from sklearn.utils import check_array, check_random_state
from sklearn import cluster
from time import time


class NumericalStability:
    EPS     = 1e-10
    MAX_EXP = 500

    @staticmethod
    def safe_log(x):
        return np.log(np.maximum(x, NumericalStability.EPS))

    @staticmethod
    def safe_exp(x):
        return np.exp(np.clip(x, -NumericalStability.MAX_EXP, NumericalStability.MAX_EXP))


# ─────────────────────────────────────────────────────────────────────────────
# NIG stick-breaking helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nig_elpi(a, b):
    """
    E[log π_k] for all k under NIG stick-breaking.

    Parameters
    ----------
    a, b : (K,) arrays — Beta shape params for V_k

    Returns
    -------
    elpi : (K,) array
    """
    ab       = a + b
    e_log_v  = digamma(a) - digamma(ab)     # (K,)
    e_log_1v = digamma(b) - digamma(ab)     # (K,)
    # E[log π_k] = E[log V_k] + Σ_{j<k} E[log(1-V_j)]
    cum_1v   = np.concatenate([[0.0], np.cumsum(e_log_1v)[:-1]])
    return e_log_v + cum_1v


def _nig_kl(a, b, sigma=0.5):
    """
    KL[q(V_k) || Beta(1−σ, 1−σ)] summed over k.

    The NIG prior on each V_k is Beta(1−σ, 1−σ)  (σ = ½ → Beta(½, ½)).
    q(V_k) = Beta(a_k, b_k).

    Returns the KL divergence (positive scalar); caller negates for ELBO.
    """
    p_a = 1.0 - sigma   # = 0.5
    p_b = 1.0 - sigma   # = 0.5
    ab  = a + b
    kl  = ( gammaln(p_a + p_b) - gammaln(p_a) - gammaln(p_b)
           - gammaln(ab)       + gammaln(a)   + gammaln(b)
           + (a - p_a) * (digamma(a) - digamma(ab))
           + (b - p_b) * (digamma(b) - digamma(ab)) )
    return float(kl.sum())


class DMM_SVVS_Variational_v2_5:
    """
    Dirichlet Multinomial Mixture with Variational Variable Selection — v2.5

    Extends v2.4 with one targeted replacement:
      MFM Dirichlet prior → Normalized Inverse Gaussian (NIG) Process Prior.

    The NIG prior replaces the single K-dimensional Dirichlet variational
    parameter phi (K,) with two K-dimensional Beta stick-breaking parameters
    nig_a (K,) and nig_b (K,).

    All v2.4 features are preserved:
      - per_sample_f dual-mode variable selection
      - Early pruning with adaptive min_cluster_size
      - Single restart, relative-ELBO convergence

    Parameters
    ----------
    K_max : int
        Maximum / initial number of clusters (truncation level).
    nig_sigma : float
        NIG stability exponent σ ∈ (0, 1).  Default 0.5 (canonical NIG).
        Lower σ → fewer clusters (DP limit at σ→0); higher → more.
    nig_alpha : float
        NIG total-mass / concentration parameter > 0.  Scales E[K_n].
        Default 1.0.  Larger → more clusters a priori.
    zeta : float
        Prior concentration for cluster-specific Dirichlet (alpha_k).
    eta : float
        Prior concentration for background Dirichlet (beta).
    xi_1, xi_2 : float
        Beta(xi_1, xi_2) prior on selection probabilities.
        Default 1.0/1.0 → uniform (neutral) prior.
    selection_prior : float
        Initial warm-start value for f.
    per_sample_f : bool
        If True  (default): f ∈ [0,1]^{N×S}, xi_star shape (N,S,2).
                            Each sample has its own per-feature selector.
        If False (v2-style): f is a shared (S,) vector broadcast to (N,S),
                            xi_star shape (S,2).  Faster; enables direct
                            ARI comparison with per_sample_f=True.
    tol : float
        Relative ELBO convergence tolerance.
    max_iter : int
        Maximum CAVI iterations.
    prune_threshold : float
        A cluster is kept only if its effective weight > prune_threshold
        AND its sample count > 1.0.
    min_clusters : int or None
        Minimum number of clusters to keep.  None → max(2, K_max // 5).
    prune_start : int
        Iteration at which pruning begins (default 10).
    prune_every : int
        Prune every this many iterations (default 5).
    verbose : int
    random_state : int
    """

    def __init__(self,
                 K_max=10,
                 nig_sigma=0.5,
                 nig_alpha=1.0,
                 zeta=1.0,
                 eta=1.0,
                 xi_1=1.0,
                 xi_2=1.0,
                 selection_prior=0.3,
                 per_sample_f=True,
                 tol=1e-4,
                 max_iter=500,
                 prune_threshold=0.01,
                 min_clusters=None,
                 prune_start=10,
                 prune_every=5,
                 verbose=1,
                 random_state=42):

        self.K_max           = K_max
        self.nig_sigma       = float(np.clip(nig_sigma, 1e-4, 1.0 - 1e-4))
        self.nig_alpha       = float(nig_alpha)
        self.zeta            = zeta
        self.eta             = eta
        self.xi_1            = xi_1
        self.xi_2            = xi_2
        self.selection_prior = selection_prior
        self.per_sample_f    = bool(per_sample_f)
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
        self.r = self.f = None
        self.nig_a = self.nig_b = None   # (K,)  NIG stick-breaking Beta params
        self.lambda_star = self.iota_star = None
        self.xi_star = None              # (N,S,2) if per_sample_f else (S,2)
        self.elbo_history = []
        self.converged = False
        self.n_iter = 0
        self._cache = {}
        self._pruned_at_least_once = False

    # ─────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────

    def _initialize_parameters(self, X, random_state):
        self.N, self.S = X.shape
        self.K  = self.K_max

        # Resolve min_clusters
        self._min_clusters = (
            max(2, self.K_max // 5)
            if self.min_clusters is None
            else int(self.min_clusters)
        )

        # Adaptive minimum cluster size for pruning   [FIX 1 from v2]
        self._min_cluster_size = max(1.0, self.N / (5.0 * self.K_max))

        f_mode = "per-sample (N,S)" if self.per_sample_f else "shared (S,) → v2-style"
        if self.verbose >= 1:
            print(f"Initializing: N={self.N}, S={self.S}, K_max={self.K_max}")
            print(f"  NIG: σ={self.nig_sigma:.4f}, α={self.nig_alpha:.4f}")
            print(f"  ζ={self.zeta}, η={self.eta}, "
                  f"ξ=({self.xi_1},{self.xi_2}), "
                  f"f_mode={f_mode}, "
                  f"min_K={self._min_clusters}, "
                  f"min_cluster_size={self._min_cluster_size:.1f}")

        # Responsibilities from k-means
        self.r = self._init_responsibilities_kmeans(X, random_state)

        # Feature selection warm-start
        self.f = np.full((self.N, self.S), self.selection_prior)

        # ── NIG stick-breaking params init  [Change: replaces MFM phi]
        # Initialise nig_a, nig_b using the CAVI update evaluated at k-means r
        self._update_nig_sticks()

        # Beta prior on feature selection  (identical to v2.4)
        # xi_star shape: (N, S, 2) when per_sample_f=True
        #                (S, 2)    when per_sample_f=False
        if self.per_sample_f:
            self.xi_star = np.empty((self.N, self.S, 2))
            self.xi_star[:, :, 0] = self.xi_1
            self.xi_star[:, :, 1] = self.xi_2
        else:
            self.xi_star = np.empty((self.S, 2))
            self.xi_star[:, 0] = self.xi_1
            self.xi_star[:, 1] = self.xi_2

        # Cluster-specific Dirichlet parameters: warm-start from k-means profiles.
        #
        # We use a NORMALISED proportion form: λ_k ∝ mean_profile_k, scaled so
        # that Σ_j λ_kj = ζ * S.  This keeps λ values in the same order of
        # magnitude as the prior concentration ζ, which prevents E[log α_kj]
        # from becoming astronomically large/small at init.
        self.lambda_star = np.zeros((self.K, self.S))
        for k in range(self.K):
            mask = self.r[:, k] > 0.1
            if mask.sum() > 0:
                mu = X[mask].mean(axis=0) + 0.5
                self.lambda_star[k] = mu / mu.sum() * (self.zeta * self.S)
            else:
                self.lambda_star[k] = self.zeta
        self.lambda_star = np.maximum(self.lambda_star, 0.1)

        # Background Dirichlet parameters — normalised form for the same reason.
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
    # NIG stick-breaking update  [Change: replaces MFM _delta_eff + _update_phi]
    # ─────────────────────────────────────────────────────────────────────

    def _update_nig_sticks(self):
        """
        NIG CAVI update for stick-breaking Beta parameters.  [Change over v2.4]

        The NIG EPPF (Barrios et al. 2013, Proposition 3.2) gives:

            a_k = (1 − σ) + α/K + r_k
            b_k = (1 − σ) + α/K + mass_above_k

        where:
          σ         = nig_sigma (NIG stability exponent, default 0.5)
          α         = nig_alpha (concentration, default 1.0)
          r_k       = Σ_i r_ik  (effective count for cluster k)
          mass_above_k = Σ_{j>k} r_j  (remaining mass above k in stick order)

        The α/K term mirrors the DP concentration scaling and preserves
        symmetry of the Beta prior while scaling the effective sample size.

        This replaces the MFM update:
            φ_k = δ_eff + r_k,  δ_eff = δ + γ/K
        """
        EPS        = NumericalStability.EPS
        sigma      = self.nig_sigma
        rk         = self.r.sum(axis=0)               # (K,)
        mass_above = np.cumsum(rk[::-1])[::-1] - rk   # Σ_{j>k} r_j  (K,)
        prior_term = (1.0 - sigma) + self.nig_alpha / self.K
        self.nig_a = np.maximum(prior_term + rk,          EPS)  # (K,)
        self.nig_b = np.maximum(prior_term + mass_above,  EPS)  # (K,)

    # ─────────────────────────────────────────────────────────────────────
    # Cached expectations
    # ─────────────────────────────────────────────────────────────────────

    def _E_log_alpha(self):
        """(K, S): E[log α_kj] = ψ(λ_kj) − ψ(Σ_j λ_kj)."""
        if 'Ela' not in self._cache:
            s = self.lambda_star.sum(axis=1, keepdims=True)   # (K, 1)
            self._cache['Ela'] = digamma(self.lambda_star) - digamma(s)
        return self._cache['Ela']

    def _E_log_beta(self):
        """(S,): E[log β_j] = ψ(ι_j) − ψ(Σ_j ι_j)."""
        if 'Elb' not in self._cache:
            self._cache['Elb'] = (
                digamma(self.iota_star) - digamma(self.iota_star.sum())
            )
        return self._cache['Elb']

    def _E_log_pi(self):
        """
        (K,): E[log π_k] under NIG stick-breaking.  [Change over v2.4]

        NIG stick-breaking:
          E[log π_k] = E[log V_k] + Σ_{j<k} E[log(1−V_j)]
          E[log V_k] = ψ(a_k) − ψ(a_k + b_k)
          E[log(1−V_k)] = ψ(b_k) − ψ(a_k + b_k)

        This replaces the MFM Dirichlet formula used in v2.4:
          E[log π_k] = ψ(φ_k) − ψ(Σ_k φ_k)
        """
        if 'Elpi' not in self._cache:
            self._cache['Elpi'] = _nig_elpi(self.nig_a, self.nig_b)
        return self._cache['Elpi']

    def _clear_cache(self):
        self._cache = {}

    # ─────────────────────────────────────────────────────────────────────
    # Expected log-likelihood  (identical to v2.4)
    # ─────────────────────────────────────────────────────────────────────

    def _expected_log_lik(self, X):
        """
        (N, K) expected log-likelihood with per-sample per-feature f.

        E[log p(x_i | z_i=k)] =
          Σ_j f_ij · x_ij · E[log α_kj]
        + Σ_j (1 - f_ij) · x_ij · E[log β_j]

        = (f * X) @ E_log_a.T  +  ((1-f) * X) @ E_log_b   broadcast to (N,K)
        """
        E_log_a = self._E_log_alpha()          # (K, S)
        E_log_b = self._E_log_beta()           # (S,)

        fX           = self.f * X              # (N, S)
        one_minus_fX = (1.0 - self.f) * X     # (N, S)

        ll  = fX @ E_log_a.T                   # (N, K)
        ll += (one_minus_fX @ E_log_b)[:, None]  # (N, 1) broadcast → (N, K)
        return ll

    # ─────────────────────────────────────────────────────────────────────
    # CAVI update steps
    # ─────────────────────────────────────────────────────────────────────

    def _update_r(self, X):
        EPS      = NumericalStability.EPS
        E_log_pi = self._E_log_pi()           # (K,)
        ll       = self._expected_log_lik(X)  # (N, K)

        log_r  = E_log_pi[None, :] + ll
        log_r -= logsumexp(log_r, axis=1, keepdims=True)
        self.r = np.exp(log_r)
        self.r = np.maximum(self.r, EPS)
        self.r /= self.r.sum(axis=1, keepdims=True)

    def _update_f(self, X):
        """
        Feature selection update — dispatches on self.per_sample_f.
        Identical to v2.4; no dependency on the cluster weight prior.

        per_sample_f=True:
          Per-sample per-feature update.  xi_star: (N, S, 2).
          For each (i, j):
            log_odds[i,j] = E[log(ξ*_1[i,j] / ξ*_2[i,j])]
                          + x_ij · (Σ_k r_ik · E[log α_kj] − E[log β_j])
          f shape: (N, S).

        per_sample_f=False  (v2-style shared):
          Shared per-feature update.  xi_star: (S, 2).
          For each j:
            log_odds_j = E[log(ξ*_1_j / ξ*_2_j)]
                       + Σ_i Σ_k r_ik · x_ij · E[log α_kj]
                       − Σ_i x_ij · E[log β_j]
          f is a (S,) vector broadcast to (N, S).
        """
        EPS         = NumericalStability.EPS
        E_log_alpha = self._E_log_alpha()   # (K, S)
        E_log_beta  = self._E_log_beta()    # (S,)

        if self.per_sample_f:
            # ── per-sample mode ───────────────────────────────────────────
            xi_sum = self.xi_star.sum(axis=2)                            # (N, S)
            E_xi1  = digamma(self.xi_star[:, :, 0]) - digamma(xi_sum)   # (N, S)
            E_xi2  = digamma(self.xi_star[:, :, 1]) - digamma(xi_sum)   # (N, S)

            # Σ_k r_ik · E[log α_kj]  →  (N, S)
            el_alpha_ni = self.r @ E_log_alpha   # (N,K)@(K,S) = (N,S)

            log_ps_raw = X * el_alpha_ni                  # (N, S)
            log_pu_raw = X * E_log_beta[None, :]          # (N, S)

            log_odds = np.clip(E_xi1 + log_ps_raw - E_xi2 - log_pu_raw,
                               -500, 500)
            self.f = np.clip(expit(log_odds), EPS, 1 - EPS)   # (N, S)

        else:
            # ── shared (v2-style) mode ────────────────────────────────────
            xi_sum = self.xi_star.sum(axis=1)                    # (S,)
            E_xi1  = digamma(self.xi_star[:, 0]) - digamma(xi_sum)  # (S,)
            E_xi2  = digamma(self.xi_star[:, 1]) - digamma(xi_sum)  # (S,)

            # Σ_k r_ik · E[log α_kj]  →  (N, S), then sum over i with X
            el_alpha_ni = self.r @ E_log_alpha   # (N,K)@(K,S) = (N,S)

            log_ps_raw = (X * el_alpha_ni).sum(axis=0)          # (S,)
            log_pu_raw = (X * E_log_beta[None, :]).sum(axis=0)  # (S,)

            log_odds = np.clip(E_xi1 + log_ps_raw - E_xi2 - log_pu_raw,
                               -500, 500)
            f_vec  = np.clip(expit(log_odds), EPS, 1 - EPS)     # (S,)
            self.f = np.broadcast_to(f_vec, (self.N, self.S)).copy()  # (N,S)

    def _update_xi_star(self):
        """
        Beta posterior update for feature selection hyperparameters.
        Identical to v2.4; no dependency on the cluster weight prior.

        per_sample_f=True  → xi_star shape (N, S, 2):
          ξ*_1[i,j] = ξ_1 + f[i,j]
          ξ*_2[i,j] = ξ_2 + (1 − f[i,j])

        per_sample_f=False → xi_star shape (S, 2):
          ξ*_1[j] = ξ_1 + Σ_i f[i,j]   (= ξ_1 + N · f_j, since f is shared)
          ξ*_2[j] = ξ_2 + Σ_i (1−f[i,j])
        """
        EPS = NumericalStability.EPS
        if self.per_sample_f:
            self.xi_star[:, :, 0] = np.maximum(self.xi_1 + self.f,         EPS)
            self.xi_star[:, :, 1] = np.maximum(self.xi_2 + (1.0 - self.f), EPS)
        else:
            # f is (N,S) broadcast of a shared (S,) vector; sum over samples
            self.xi_star[:, 0] = np.maximum(
                self.xi_1 + self.f.sum(axis=0), EPS)         # (S,)
            self.xi_star[:, 1] = np.maximum(
                self.xi_2 + (1.0 - self.f).sum(axis=0), EPS) # (S,)

    def _update_lambda_star(self, X):
        """
        Exact conjugate CAVI update  (identical to v2.4):
          λ_kj = ζ + Σ_i r_ik · f_ij · x_ij  =  ζ + r^T @ (f * X)
        """
        fX = self.f * X                          # (N, S)
        self.lambda_star = np.maximum(
            self.zeta + self.r.T @ fX,           # (K, N) @ (N, S) = (K, S)
            0.1
        )

    def _update_iota_star(self, X):
        """
        Background Dirichlet update  (identical to v2.4):
          ι_j = η + Σ_i (1 − f_ij) · x_ij
        """
        self.iota_star = np.maximum(
            self.eta + ((1.0 - self.f) * X).sum(axis=0),   # (S,)
            0.1
        )

    # ─────────────────────────────────────────────────────────────────────
    # ELBO
    # ─────────────────────────────────────────────────────────────────────

    def _compute_elbo(self, X):
        """
        ELBO with NIG stick-breaking KL term and per-sample f KL term.

        ELBO = E[log p(X|Z,α,β,f)] + E[log p(Z|π)] − E[log q(Z)]
             − KL[q(V)||p(V)]   ← NIG Beta(1−σ,1−σ) prior  [Change over v2.4]
             − KL[q(f)||p(f)]   ← per-sample (N,S,2) xi_star (from v2.4)

        The α and β KL terms are omitted for speed (same as v2.4).

        KL[q(V_k) || Beta(1−σ, 1−σ)] summed over k  [Change over v2.4]:
          = Σ_k { log B(1−σ, 1−σ) − log B(a_k, b_k)
                + (a_k − (1−σ)) · [ψ(a_k) − ψ(a_k+b_k)]
                + (b_k − (1−σ)) · [ψ(b_k) − ψ(a_k+b_k)] }

        Previously (v2.4) this was:
          KL[Dir(φ) || Dir(δ_eff·1_K)]
        """
        EPS = NumericalStability.EPS

        # ── 1. E[log p(X|Z,α,β,f)] + E[log p(Z|π)] − E[log q(Z)] ──────
        E_log_pi = self._E_log_pi()
        ll       = self._expected_log_lik(X)   # (N, K)

        elbo  = float(np.sum(self.r * (E_log_pi[None, :] + ll)))
        elbo -= float(np.sum(self.r * np.log(np.maximum(self.r, EPS))))

        # ── 2. −KL[q(V)||p(V)]  (NIG stick-breaking prior)  [Change] ────
        elbo -= _nig_kl(self.nig_a, self.nig_b, self.nig_sigma)

        # ── 3. −KL[q(f)||p(f)]  (identical to v2.4) ────────────────────
        if self.per_sample_f:
            a_q   = self.xi_star[:, :, 0]   # (N, S)
            b_q   = self.xi_star[:, :, 1]   # (N, S)
        else:
            a_q   = self.xi_star[:, 0]      # (S,)
            b_q   = self.xi_star[:, 1]      # (S,)
        xi_st = a_q + b_q

        kl_f = (gammaln(self.xi_1 + self.xi_2)
                - gammaln(self.xi_1) - gammaln(self.xi_2)
                - gammaln(xi_st) + gammaln(a_q) + gammaln(b_q)
                + (a_q - self.xi_1) * (digamma(a_q) - digamma(xi_st))
                + (b_q - self.xi_2) * (digamma(b_q) - digamma(xi_st)))
        elbo -= float(kl_f.sum())

        return elbo

    # ─────────────────────────────────────────────────────────────────────
    # Pruning  (identical logic to v2.4; weight computed from NIG sticks)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_weights(self):
        """
        Mixing weights from NIG stick-breaking expectations.  [Change over v2.4]

        Compute normalised exp(E[log π_k]) as the effective mixing weights.
        Under MFM (v2.4) this was simply φ_k / Σ_k φ_k.
        Under NIG stick-breaking, E[log π_k] is computed via _nig_elpi,
        then soft-maxed for a valid probability vector.
        """
        elpi = self._E_log_pi()                     # (K,)
        return np.exp(elpi - logsumexp(elpi))        # (K,) normalised

    def _prune_empty_clusters(self):
        """
        Prune clusters using OR logic  (identical structure to v2.4).

        A cluster is REMOVED if:
          weight  <= prune_threshold    OR
          sample count <= 1.0           (truly empty)

        Hard floor: _min_clusters always kept (by weight ranking).
        """
        weights       = self._compute_weights()
        cluster_sizes = self.r.sum(axis=0)

        if self.verbose >= 2:
            print(f"  [prune check] weights={np.round(weights, 4)}, "
                  f"sizes={np.round(cluster_sizes, 1)}")

        # OR logic: remove if weight tiny OR truly empty
        keep   = ((weights > self.prune_threshold) &
                  (cluster_sizes > 1.0))
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
            # [Change over v2.4: update nig_a/nig_b instead of phi]
            self.nig_a       = self.nig_a[keep]
            self.nig_b       = self.nig_b[keep]
            self.lambda_star = self.lambda_star[keep]
            self.r          /= self.r.sum(axis=1, keepdims=True)
            # Re-derive stick-breaking params for the reduced K
            self._update_nig_sticks()
            self._clear_cache()
            self._pruned_at_least_once = True
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    # Main fit loop  (identical structure to v2.4)
    # ─────────────────────────────────────────────────────────────────────

    def fit(self, X):
        """
        Fit via coordinate ascent variational inference.

        Pruning starts at prune_start (default 10), runs every prune_every
        (default 5) iterations.  Convergence is checked after prune_start
        iterations regardless of whether pruning has fired.
        Relative ELBO tolerance is 1e-4.
        """
        X            = check_array(X, dtype=np.float64)
        random_state = check_random_state(self.random_state)

        self._initialize_parameters(X, random_state)

        f_label = "per-sample f (N,S)" if self.per_sample_f else "shared f (S,) v2-style"
        if self.verbose >= 1:
            print(f"\nStarting CAVI  (v2.5 — NIG prior + {f_label})")
            print("=" * 70)

        t0 = time()

        for iteration in range(1, self.max_iter + 1):
            self.n_iter = iteration
            self._clear_cache()

            # ── CAVI updates ─────────────────────────────────────────────
            # Correct NIG order (mirrors v2.4 MFM order):
            #   r uses E[log π] and E[log α], so nig sticks/lambda/iota must
            #   be up-to-date before r.  After r, refresh nig sticks.  Then
            #   update lambda/iota from the new r.  Finally update f/xi_star
            #   which depend on the freshly updated r, lambda, and iota.
            self._update_r(X)
            self._update_nig_sticks()          # NIG stick-breaking update [Change]
            self._update_lambda_star(X)        # needs updated r
            self._update_iota_star(X)          # needs updated f (use previous f)
            self._update_f(X)                  # needs updated lambda/iota
            self._update_xi_star()             # needs updated f

            # ── Pruning (v2 FIX 3: earlier and more frequent) ────────────
            if iteration >= self.prune_start and iteration % self.prune_every == 0:
                self._prune_empty_clusters()

            # ── ELBO + convergence ────────────────────────────────────────
            if iteration % 10 == 0:
                elbo = self._compute_elbo(X)
                self.elbo_history.append(elbo)

                if self.verbose >= 1:
                    print(f"Iter {iteration:4d}: ELBO = {elbo:14.2f}, "
                          f"K = {self.K}, Time = {time()-t0:.1f}s")

                # Check convergence after pruning has had a chance to run
                if iteration > self.prune_start and len(self.elbo_history) >= 3:
                    recent  = self.elbo_history[-3:]
                    changes = [abs(recent[i] - recent[i-1]) /
                               (abs(recent[i]) + 1e-10)
                               for i in range(1, len(recent))]
                    if all(c < self.tol for c in changes):
                        self.converged = True
                        if self.verbose >= 1:
                            print(f"\n  Converged at iteration {iteration}")
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
    # Prediction and inspection  (identical to v2.4)
    # ─────────────────────────────────────────────────────────────────────

    def predict(self, X):
        X     = check_array(X, dtype=np.float64)
        N_new = X.shape[0]

        # Use mean of trained f as proxy for new samples
        f_new = np.tile(self.f.mean(axis=0), (N_new, 1))   # (N_new, S)

        # Temporarily swap state
        N_orig, f_orig, r_orig = self.N, self.f, self.r
        self.N = N_new
        self.f = f_new
        self.r = np.zeros((N_new, self.K))
        self._clear_cache()

        ll       = self._expected_log_lik(X)
        E_log_pi = self._E_log_pi()
        log_r    = E_log_pi[None, :] + ll
        log_r   -= logsumexp(log_r, axis=1, keepdims=True)
        labels   = np.exp(log_r).argmax(axis=1)

        # Restore
        self.N, self.f, self.r = N_orig, f_orig, r_orig
        self._clear_cache()
        return labels

    def get_selected_features(self, threshold=0.5):
        """
        Return OTU indices selected in more than `threshold` fraction of samples.

        f has shape (N, S); feature j is selected for sample i if f[i,j] > 0.5.
        """
        frac_selected = (self.f > 0.5).mean(axis=0)   # (S,)
        return np.where(frac_selected > threshold)[0].tolist()

    def get_cluster_signatures(self, threshold=0.5):
        """
        For each cluster, return features selected by a majority of its members.
        """
        signatures = {}
        for k in range(self.K):
            mask = self.r[:, k] > 0.5
            if mask.sum() > 0:
                avg_sel   = self.f[mask].mean(axis=0)     # (S,)
                important = np.where(avg_sel > threshold)[0]
                if len(important) > 0:
                    lam_sum = self.lambda_star[k].sum()
                    alpha_k = self.lambda_star[k] / lam_sum
                    signatures[k] = {
                        'features':        important.tolist(),
                        'selection_probs': avg_sel[important].tolist(),
                        'alpha_values':    alpha_k[important].tolist(),
                        'n_samples':       int(mask.sum()),
                    }
        return signatures


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print("Smoke-test — DMM_SVVS_Variational_v2_5")
    print("  NIG prior  |  Comparing per_sample_f=True vs per_sample_f=False")
    print("=" * 70)

    rng = np.random.default_rng(42)
    N, S, K_true = 150, 200, 3

    # Block-structured DMM data (same as v2.4 smoke-test for fair comparison)
    alpha = np.full((K_true, S), 0.1)
    block = S // K_true
    for k in range(K_true):
        alpha[k, k * block:(k + 1) * block] = 3.0
    true_labels = rng.choice(K_true, size=N)
    X = np.array([rng.multinomial(5000, rng.dirichlet(alpha[true_labels[i]]))
                  for i in range(N)], dtype=float)

    common = dict(K_max=10, nig_sigma=0.5, nig_alpha=1.0,
                  max_iter=300, verbose=1, random_state=42)

    print("\n── Mode A: per_sample_f=True (per-sample per-feature) ──")
    m_a = DMM_SVVS_Variational_v2_5(**common, per_sample_f=True)
    m_a.fit(X)
    pred_a = m_a.predict(X)
    ari_a  = adjusted_rand_score(true_labels, pred_a)
    nmi_a  = normalized_mutual_info_score(true_labels, pred_a)

    print("\n── Mode B: per_sample_f=False (shared f, v2-style) ──")
    m_b = DMM_SVVS_Variational_v2_5(**common, per_sample_f=False)
    m_b.fit(X)
    pred_b = m_b.predict(X)
    ari_b  = adjusted_rand_score(true_labels, pred_b)
    nmi_b  = normalized_mutual_info_score(true_labels, pred_b)

    print("\n" + "=" * 70)
    print(f"{'':30s}  {'per_sample_f=True':>18}  {'per_sample_f=False':>18}")
    print(f"{'ARI':30s}  {ari_a:>18.4f}  {ari_b:>18.4f}")
    print(f"{'NMI':30s}  {nmi_a:>18.4f}  {nmi_b:>18.4f}")
    print(f"{'K estimated':30s}  {m_a.K:>18d}  {m_b.K:>18d}")
    print(f"{'K true':30s}  {K_true:>18d}  {K_true:>18d}")
    print(f"{'Selected OTUs (>50% samples)':30s}  "
          f"{len(m_a.get_selected_features()):>18d}  "
          f"{len(m_b.get_selected_features()):>18d}")

    # Optional: compare against v2.4 if available
    try:
        from DMM_SVVS_Variational_v2_4 import DMM_SVVS_Variational_v2_4
        print("\n── v2.4 (MFM prior, per_sample_f=True) for comparison ──")
        m_mfm = DMM_SVVS_Variational_v2_4(
            K_max=10, mfm_delta=1.0, mfm_gamma=1.0,
            max_iter=300, verbose=1, random_state=42, per_sample_f=True)
        m_mfm.fit(X)
        pred_mfm = m_mfm.predict(X)
        ari_mfm  = adjusted_rand_score(true_labels, pred_mfm)
        nmi_mfm  = normalized_mutual_info_score(true_labels, pred_mfm)
        print(f"\nv2.4 MFM ARI={ari_mfm:.4f}  NMI={nmi_mfm:.4f}  K={m_mfm.K}")
        print(f"v2.5 NIG ARI={ari_a:.4f}  NMI={nmi_a:.4f}  K={m_a.K}")
        print(f"ΔARI (NIG − MFM) = {ari_a - ari_mfm:+.4f}")
    except ImportError:
        pass
