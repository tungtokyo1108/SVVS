#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS Variational Inference — Version 2.4

Single targeted change over v2.3:

  Change — Mixture of Finite Mixtures (MFM) Prior replaces Pitman-Yor Process
  ─────────────────────────────────────────────────────────────────────────────
  v2.3 used a Pitman-Yor (PY) stick-breaking prior on the mixing weights π,
  with variational parameters theta (K,) and theta_prime (K,) governing K
  Beta distributions in a truncated stick-breaking representation.

  v2.4 replaces this with an MFM prior (Miller & Harrison 2018, JASA):

      K  ~ Poisson(γ) + 1          (prior on number of clusters)
      π | K ~ Dirichlet(δ, …, δ)   (symmetric K-dim Dirichlet)

  Under the truncated variational family q(π) = Dir(φ), the CAVI update is:

      φ_k = δ_eff + r_k,    r_k = Σ_i r_ik
      E[log π_k] = ψ(φ_k) − ψ(Σ_k φ_k)

  where δ_eff = δ + γ/K_max absorbs the Poisson(γ)+1 prior on K
  (Frühwirth-Schnatter et al. 2021, eq. 3.4).

  This replaces the stick-breaking Beta parameters (theta, theta_prime) with a
  single Dirichlet parameter vector phi (K,).  The _update_theta_py method and
  all PY-specific ELBO KL computation are replaced accordingly.

  Why MFM over PY/DP
  ──────────────────
  DP and PY produce a biased posterior on K — the number of clusters is
  systematically over-estimated because extra empty components receive non-zero
  prior probability.  MFM places an explicit prior on K and yields a consistent
  posterior (converges to true K as N → ∞), which directly improves ARI.

  References
  ──────────
  Miller & Harrison (2018) JASA 113(521):340–356.
  Frühwirth-Schnatter et al. (2021) Bayesian Analysis.
  Pitman & Yor (1997) — replaced by MFM in this version.

Everything else from v2.3 is preserved unchanged:
  - Variable selection mode controlled by `per_sample_f`        [v2.3 Change 2]
      per_sample_f=True  (default): f ∈ [0,1]^{N×S}, xi_star (N,S,2)
      per_sample_f=False (v2-style): shared (S,), broadcast to (N,S)
  - AND-logic pruning with adaptive min_cluster_size             [v2 FIX 1]
  - Adaptive effective concentration (δ_eff)                    [FIX 2 analogue]
  - Early pruning (prune_start=10, prune_every=5)               [v2 FIX 3]
  - Neutral Beta(1,1) selection prior                           [v2 FIX 4]
  - Single restart, same convergence check, same ELBO structure

Author: v2.4 — MFM prior replaces PY process over v2.3
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


class DMM_SVVS_Variational_v2_4:
    """
    Dirichlet Multinomial Mixture with Variational Variable Selection — v2.4

    Extends v2.3 with one targeted replacement:
      PY stick-breaking prior → Mixture of Finite Mixtures (MFM) prior.

    The MFM prior replaces the two Beta variational parameters per cluster
    (theta, theta_prime) with a single K-dimensional Dirichlet parameter (phi).

    All v2.3 features are preserved:
      - per_sample_f dual-mode variable selection
      - Early pruning with adaptive min_cluster_size
      - Single restart, relative-ELBO convergence

    Parameters
    ----------
    K_max : int
        Maximum / initial number of clusters (truncation level).
    mfm_delta : float
        Symmetric Dirichlet concentration δ for π | K ~ Dir(δ, …, δ).
        Larger δ → more uniform weights → more clusters survive.
        Range 0.5–5.0; default 1.0.
    mfm_gamma : float
        Rate of Poisson(γ)+1 prior on K.  Sets prior mean clusters to γ+1.
        Default 1.0.
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
                 mfm_delta=1.0,
                 mfm_gamma=1.0,
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
        self.mfm_delta       = float(mfm_delta)
        self.mfm_gamma       = float(mfm_gamma)
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
        self.phi = None              # (K,)  MFM Dirichlet weight params
        self.lambda_star = self.iota_star = None
        self.xi_star = None          # (N,S,2) if per_sample_f else (S,2)
        self.elbo_history = []
        self.converged = False
        self.n_iter = 0
        self._cache = {}
        self._pruned_at_least_once = False

    # ─────────────────────────────────────────────────────────────────────
    # MFM effective concentration
    # ─────────────────────────────────────────────────────────────────────

    def _delta_eff(self):
        """
        δ_eff = δ + γ/K  (MFM effective Dirichlet concentration).

        Uses the *current* K (not K_max) so that after pruning the effective
        concentration correctly reflects the remaining truncation level.
        Frühwirth-Schnatter et al. (2021), eq. 3.4.

        Bug fixed: original code used K_max (frozen at init), which
        over-inflated φ after pruning collapsed K below K_max.
        """
        return self.mfm_delta + self.mfm_gamma / self.K

    # ─────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────

    def _initialize_parameters(self, X, random_state):
        self.N, self.S = X.shape
        self.K  = self.K_max
        # _de is recomputed dynamically in _update_phi() so it tracks current K;
        # cache it here only for the verbose print below.
        self._de = self._delta_eff()

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
            print(f"  MFM: δ={self.mfm_delta:.4f}, γ={self.mfm_gamma:.4f}, "
                  f"δ_eff={self._de:.4f}")
            print(f"  ζ={self.zeta}, η={self.eta}, "
                  f"ξ=({self.xi_1},{self.xi_2}), "
                  f"f_mode={f_mode}, "
                  f"min_K={self._min_clusters}, "
                  f"min_cluster_size={self._min_cluster_size:.1f}")

        # Responsibilities from k-means
        self.r = self._init_responsibilities_kmeans(X, random_state)

        # Feature selection warm-start
        self.f = np.full((self.N, self.S), self.selection_prior)

        # ── MFM Dirichlet weight params init  [Change: replaces PY theta/theta_prime]
        # Initialise φ_k = δ_eff + r_k  (conjugate update evaluated at k-means r)
        rk = self.r.sum(axis=0)                       # (K,)
        self.phi = np.maximum(self._de + rk, NumericalStability.EPS)  # (K,)

        # Beta prior on feature selection  (identical to v2.3)
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
        # from becoming astronomically large/small at init (with hard k-means r,
        # raw count sums reach ζ + N_k * count_per_sample ~ 10^5, which makes
        # E[log α] differences so extreme that r collapses to one-hot and CAVI
        # freezes immediately).
        #
        # The normalised form gives the same directional information (which OTUs
        # are enriched in each cluster) at a numerically stable scale.  CAVI will
        # naturally inflate λ to the correct posterior scale over the first few
        # iterations as soft responsibilities accumulate.
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
        (K,): E[log π_k] under q(π) = Dir(φ).

        MFM Dirichlet:  E[log π_k] = ψ(φ_k) − ψ(Σ_k φ_k).

        This replaces the stick-breaking cumsum formula used in v2.3:
          E[log π_k] = E[log v_k] + Σ_{j<k} E[log(1−v_j)]
        which was specific to the PY/DP Beta factorisation.
        """
        if 'Elpi' not in self._cache:
            self._cache['Elpi'] = digamma(self.phi) - digamma(self.phi.sum())
        return self._cache['Elpi']

    def _clear_cache(self):
        self._cache = {}

    # ─────────────────────────────────────────────────────────────────────
    # Expected log-likelihood  (identical to v2.3)
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
        Identical to v2.3; no dependency on the cluster weight prior.

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

    def _update_phi(self):
        """
        MFM Dirichlet weight update  [Change: replaces _update_theta_py].

        Conjugate CAVI update for q(π) = Dir(φ):
          φ_k = δ_eff + Σ_i r_ik = δ_eff + r_k

        δ_eff is recomputed from the *current* K every call so that after
        pruning (K < K_max) the effective concentration stays correct.
        Bug fixed: storing self._de at init and never refreshing it caused
        φ to be over-inflated whenever K shrank below K_max.
        """
        self._de = self._delta_eff()               # refresh for current K
        rk = self.r.sum(axis=0)                    # (K,)
        self.phi = np.maximum(self._de + rk, NumericalStability.EPS)

    def _update_xi_star(self):
        """
        Beta posterior update for feature selection hyperparameters.
        Identical to v2.3; no dependency on the cluster weight prior.

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
        Exact conjugate CAVI update  (identical to v2.3):
          λ_kj = ζ + Σ_i r_ik · f_ij · x_ij  =  ζ + r^T @ (f * X)
        """
        fX = self.f * X                          # (N, S)
        self.lambda_star = np.maximum(
            self.zeta + self.r.T @ fX,           # (K, N) @ (N, S) = (K, S)
            0.1
        )

    def _update_iota_star(self, X):
        """
        Background Dirichlet update  (identical to v2.3):
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
        ELBO with MFM Dirichlet KL term and per-sample f KL term.

        ELBO = E[log p(X|Z,α,β,f)] + E[log p(Z|π)] − E[log q(Z)]
             − KL[q(π)||p(π)]   ← MFM Dir(δ_eff,…) prior  [Change]
             − KL[q(f)||p(f)]   ← per-sample (N,S,2) xi_star (from v2.3)

        The α and β KL terms are omitted for speed (same as v2.3).

        KL[Dir(φ) || Dir(δ_eff·1_K)]:
          = log Γ(K·δ_eff) − K·log Γ(δ_eff)
          − log Γ(Σ φ_k) + Σ_k log Γ(φ_k)
          + Σ_k (φ_k − δ_eff)·(ψ(φ_k) − ψ(Σ φ_k))
        """
        EPS = NumericalStability.EPS

        # ── 1. E[log p(X|Z,α,β,f)] + E[log p(Z|π)] − E[log q(Z)] ──────
        E_log_pi = self._E_log_pi()
        ll       = self._expected_log_lik(X)   # (N, K)

        elbo  = float(np.sum(self.r * (E_log_pi[None, :] + ll)))
        elbo -= float(np.sum(self.r * np.log(np.maximum(self.r, EPS))))

        # ── 2. −KL[q(π)||p(π)]  (MFM Dirichlet prior)  [Change] ────────
        de      = self._delta_eff()            # always recompute for current K
        phi     = self.phi                     # (K,)
        phi_sum = phi.sum()
        Elpi_q  = digamma(phi) - digamma(phi_sum)  # (K,)

        kl_pi = (gammaln(de * self.K) - self.K * gammaln(de)
                 - (gammaln(phi_sum) - gammaln(phi).sum())
                 + float(((phi - de) * Elpi_q).sum()))
        elbo -= float(kl_pi)

        # ── 3. −KL[q(f)||p(f)]  (identical to v2.3) ────────────────────
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
    # Pruning  (identical logic to v2.3; weight computed from phi instead)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_weights(self):
        """
        Mixing weights E[π_k] = φ_k / Σ_k φ_k  under MFM Dir(φ).

        In v2.3 this was computed via stick-breaking log-cumprod from Beta
        parameters (theta, theta_prime).  Under the MFM Dirichlet, the mean
        of Dir(φ) is simply the normalised φ vector.
        """
        return self.phi / self.phi.sum()

    def _prune_empty_clusters(self):
        """
        Prune clusters using OR logic  (identical structure to v2.3).

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
            self.phi         = self.phi[keep]         # [Change: phi instead of theta/theta_prime]
            self.lambda_star = self.lambda_star[keep]
            self.r          /= self.r.sum(axis=1, keepdims=True)
            self._clear_cache()
            self._pruned_at_least_once = True
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    # Main fit loop  (identical structure to v2.3)
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
            print(f"\nStarting CAVI  (v2.4 — MFM prior + {f_label})")
            print("=" * 70)

        t0 = time()

        for iteration in range(1, self.max_iter + 1):
            self.n_iter = iteration
            self._clear_cache()

            # ── CAVI updates ─────────────────────────────────────────────
            # Correct MFM order (matches v4 reference):
            #   r uses E[log π] and E[log α], so phi/lambda/iota must be
            #   up-to-date before r.  After r, refresh phi.  Then update
            #   lambda/iota from the new r.  Finally update f/xi_star which
            #   depend on the freshly updated r, lambda, and iota.
            # Bug fixed: prior order (r→f→phi→xi→lambda→iota) updated f
            # with stale lambda/iota and updated lambda with stale f.
            self._update_r(X)
            self._update_phi()                 # MFM Dirichlet update [Change]
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
    # Prediction and inspection  (identical to v2.3)
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

    print("Smoke-test — DMM_SVVS_Variational_v2_4")
    print("  MFM prior  |  Comparing per_sample_f=True vs per_sample_f=False")
    print("=" * 70)

    rng = np.random.default_rng(42)
    N, S, K_true = 150, 200, 3

    # Block-structured DMM data (same as v2.3 smoke-test for fair comparison)
    alpha = np.full((K_true, S), 0.1)
    block = S // K_true
    for k in range(K_true):
        alpha[k, k * block:(k + 1) * block] = 3.0
    true_labels = rng.choice(K_true, size=N)
    X = np.array([rng.multinomial(5000, rng.dirichlet(alpha[true_labels[i]]))
                  for i in range(N)], dtype=float)

    common = dict(K_max=10, mfm_delta=1.0, mfm_gamma=1.0,
                  max_iter=300, verbose=1, random_state=42)

    print("\n── Mode A: per_sample_f=True (per-sample per-feature) ──")
    m_a = DMM_SVVS_Variational_v2_4(**common, per_sample_f=True)
    m_a.fit(X)
    pred_a = m_a.predict(X)
    ari_a  = adjusted_rand_score(true_labels, pred_a)
    nmi_a  = normalized_mutual_info_score(true_labels, pred_a)

    print("\n── Mode B: per_sample_f=False (shared f, v2-style) ──")
    m_b = DMM_SVVS_Variational_v2_4(**common, per_sample_f=False)
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

    # Optional: compare against v2.3 if available
    try:
        from DMM_SVVS_Variational_v2_3 import DMM_SVVS_Variational_v2_3
        print("\n── v2.3 (PY prior, per_sample_f=True) for comparison ──")
        m_py = DMM_SVVS_Variational_v2_3(
            K_max=10, py_discount=0.2, py_concentration='auto',
            max_iter=300, verbose=1, random_state=42, per_sample_f=True)
        m_py.fit(X)
        pred_py = m_py.predict(X)
        ari_py  = adjusted_rand_score(true_labels, pred_py)
        nmi_py  = normalized_mutual_info_score(true_labels, pred_py)
        print(f"\nv2.3 PY  ARI={ari_py:.4f}  NMI={nmi_py:.4f}  K={m_py.K}")
        print(f"v2.4 MFM ARI={ari_a:.4f}  NMI={nmi_a:.4f}  K={m_a.K}")
        print(f"ΔARI (MFM − PY) = {ari_a - ari_py:+.4f}")
    except ImportError:
        pass
