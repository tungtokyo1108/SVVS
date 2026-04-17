#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS-NIG: Dirichlet Multinomial Mixture with Variational Variable Selection
             using Normalized Inverse Gaussian (NIG) Process Prior — Version 5

Two innovations over v4 (MFM):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NORMALIZED INVERSE GAUSSIAN (NIG) PROCESS PRIOR  [replaces MFM/DP/PYP]
   ─────────────────────────────────────────────────────────────────────────
   Lijoi, Mena & Prünster (2005, JASA); Barrios et al. (2013, Stat. Sci.);
   arxiv 2501.18854 (2025 direct ARI benchmarks: +12–15% over MFM-Gamma).

   The NIG process is a Normalised Random Measure with Inverse Gaussian Lévy
   intensity  ν(x) = c · x^{-3/2} exp(−x/2), x > 0.

   Under a truncated stick-breaking approximation (Favaro et al. 2012), the
   variational family for the mixing weights is:

       q(V_k) = Beta(a_k, b_k),  k = 1, …, K_max
       π_k    = V_k · Π_{j<k} (1 − V_j)   [stick-breaking]

   CAVI updates (derived from the NIG EPPF; Barrios et al. 2013, eq. 3):
       a_k = (1 − σ) + r_k        where σ = ½ (NIG stability exponent)
       b_k = (1 − σ) + mass_k     where mass_k = Σ_{j>k} r_j

   Note: b_k = (1−σ) + mass_above  — the prior shape (1−σ) = ½ acts as a
   regulariser that concentrates mass on fewer sticks vs DP (b_k=α+mass_above)
   and vs PYP (b_k = α+kσ + mass_above).

   E[K_n] ∝ n^{1/2}  (power-law with exponent σ=½), intermediate between
   DP (log n) and linear growth.  The posterior on K is more tightly
   concentrated around the true K compared to MFM/DP (arxiv 2501.18854).

   Expected log mixture weights (for the r-update):
       E[log π_k] = E[log V_k] + Σ_{j<k} E[log(1−V_j)]
       E[log V_k] = ψ(a_k) − ψ(a_k + b_k)
       E[log(1−V_k)] = ψ(b_k) − ψ(a_k + b_k)

   KL divergence −KL[q(V_k) || Beta(1−σ, 1−σ)] summed over k enters the
   ELBO.

2. SOFT DETERMINANTAL REPULSION ON CLUSTER PROFILES  [Müller et al. 2011]
   ─────────────────────────────────────────────────────────────────────────
   An additive repulsion term in the ELBO:

       rep = −λ_rep · Σ_{k<j} cos²(λ_k*, λ_j*)

   where cos(·,·) is cosine similarity between variational mean profiles.
   This approximates the log-det of the DPP Gram matrix and penalises
   near-duplicate cluster profiles.  It:
   • Prevents premature merging of genuinely distinct clusters.
   • Enters ELBO comparisons in split/merge/delete, so the structural moves
     are aware of the repulsion landscape.
   • Does not break conjugacy — λ_k* updates run normally; repulsion only
     modifies the scalar ELBO value used for accept/reject and convergence.

All other components from v4 are preserved without change:
  - Per-sample variable selection f_{ij}  (Stirrup et al. 2024)
  - Deterministic annealing (β schedule)
  - Split-merge-delete structural moves
  - Multiple restarts + ICL model selection

References
──────────
  Lijoi, Mena & Prünster (2005) JASA 100(472):1278–1291.
  Barrios, Lijoi, Nieto-Barajas & Prünster (2013) Stat. Sci. 28(3):313–334.
  arxiv 2501.18854 — MFM with NIG weights; direct ARI benchmarks (2025).
  Müller, Quintana & Rosner (2011) Stat. Sci. 26(1):129–144.
  Hughes & Sudderth (2013) NIPS — split-merge memoized VB.
  Stirrup et al. (2024) NeurIPS — VBVarSel.
"""

import numpy as np
from scipy.special import digamma, gammaln, logsumexp, expit
from sklearn.utils import check_array
from sklearn import cluster
from time import time


EPS = 1e-10


def _safe_log(x):
    return np.log(np.maximum(x, EPS))


# ─────────────────────────────────────────────────────────────────────────────
# NIG stick-breaking helpers (module-level for clarity)
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
    −KL[q(V_k) || Beta(1−σ, 1−σ)] summed over k.

    The NIG prior on each V_k is Beta(1−σ, 1−σ)  (σ = ½ → Beta(½, ½)).
    q(V_k) = Beta(a_k, b_k).
    """
    p_a = 1.0 - sigma   # = 0.5
    p_b = 1.0 - sigma   # = 0.5
    ab  = a + b
    kl  = ( gammaln(p_a + p_b) - gammaln(p_a) - gammaln(p_b)
           - gammaln(ab)       + gammaln(a)   + gammaln(b)
           + (a - p_a) * (digamma(a) - digamma(ab))
           + (b - p_b) * (digamma(b) - digamma(ab)) )
    return float(kl.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class DMM_SVVS_Variational_v5:
    """
    DMM with Variational Variable Selection, NIG Process Prior,
    and Soft DPP Repulsion — Version 5.

    Parameters
    ----------
    K_max : int
        Hard upper bound on clusters (truncation level).
    nig_sigma : float
        NIG stability exponent σ ∈ (0, 1).  Default 0.5 (canonical NIG).
        Lower σ → fewer clusters (DP limit at σ→0); higher → more.
    nig_alpha : float
        NIG total-mass / concentration parameter > 0.  Scales E[K_n].
        Default 1.0.  Larger → more clusters a priori.
    zeta : float
        Dirichlet prior on cluster OTU profiles α_k.
    eta : float
        Dirichlet prior on background OTU profile β.
    xi_1, xi_2 : float
        Beta prior on per-sample-per-OTU selection f_{ij}.
    selection_prior : float
        Warm-start value for f (∈ (0,1)).
    lambda_rep : float
        Soft DPP repulsion strength.  0 = disabled.
        Range 0.0–2.0; default 0.2.
        Higher → stronger cluster separation; can prevent valid clusters
        from merging if set too large.
    tol : float
        Relative ELBO convergence tolerance.
    max_iter : int
        Maximum CAVI iterations per restart.
    beta_start : float
        Initial annealing temperature.
    beta_end : float
        Final annealing temperature (1.0).
    anneal_iters : int
        Iterations to anneal from beta_start to beta_end.
    merge_every : int
        Attempt split-merge-delete every this many iterations.
    n_restarts : int
        Number of independent restarts; best ICL is kept.
    min_clusters : int or None
        Hard floor on number of active clusters.
    verbose : int
    random_state : int or None
    """

    def __init__(self,
                 K_max=15,
                 nig_sigma=0.5,
                 nig_alpha=1.0,
                 zeta=0.1,
                 eta=0.1,
                 xi_1=2.0,
                 xi_2=1.0,
                 selection_prior=0.5,
                 lambda_rep=0.2,
                 tol=1e-4,
                 max_iter=400,
                 beta_start=0.2,
                 beta_end=1.0,
                 anneal_iters=60,
                 merge_every=15,
                 n_restarts=3,
                 min_clusters=None,
                 verbose=1,
                 random_state=42):

        self.K_max         = K_max
        self.nig_sigma     = float(np.clip(nig_sigma, 1e-4, 1.0 - 1e-4))
        self.nig_alpha     = float(nig_alpha)
        self.zeta          = zeta
        self.eta           = eta
        self.xi_1          = xi_1
        self.xi_2          = xi_2
        self.selection_prior = selection_prior
        self.lambda_rep    = float(lambda_rep)
        self.tol           = tol
        self.max_iter      = max_iter
        self.beta_start    = beta_start
        self.beta_end      = beta_end
        self.anneal_iters  = anneal_iters
        self.merge_every   = merge_every
        self.n_restarts    = n_restarts
        self.min_clusters  = min_clusters
        self.verbose       = verbose
        self.random_state  = random_state
        self._reset_state()

    # ─────────────────────────────────────────────────────────────────────
    # State management
    # ─────────────────────────────────────────────────────────────────────

    def _reset_state(self):
        self.N = self.S = self.K = None
        self.r           = None   # (N, K)
        self.f           = None   # (N, S)
        # NIG stick-breaking Beta params
        self.nig_a       = None   # (K,)
        self.nig_b       = None   # (K,)
        self.lambda_star = None   # (K, S)
        self.iota_star   = None   # (S,)
        self.xi_star     = None   # (N, S, 2)
        self.elbo_history = []
        self.converged    = False
        self.n_iter       = 0
        self._cache       = {}

    def _resolve_min_clusters(self):
        return max(2, self.K_max // 5) if self.min_clusters is None \
               else int(self.min_clusters)

    # ─────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────

    def _initialize(self, X, rng):
        self.N, self.S = X.shape
        self.K      = self.K_max
        self._min_K = self._resolve_min_clusters()

        if self.verbose >= 2:
            print(f"  Init: N={self.N}, S={self.S}, K_max={self.K_max}, "
                  f"σ={self.nig_sigma:.3f}, α={self.nig_alpha:.2f}, "
                  f"λ_rep={self.lambda_rep:.3f}")

        self.r = self._init_r_kmeans(X, rng)
        self.f = np.full((self.N, self.S), self.selection_prior)

        # NIG stick-breaking params
        self._update_nig_sticks()

        # Beta params for per-sample feature selection
        self.xi_star = np.empty((self.N, self.S, 2))
        self.xi_star[:, :, 0] = self.xi_1
        self.xi_star[:, :, 1] = self.xi_2

        # Cluster Dirichlet from k-means weighted stats
        self.lambda_star = np.zeros((self.K, self.S))
        for k in range(self.K):
            mask = self.r[:, k] > 0.1
            if mask.sum() > 0:
                w = self.r[mask, k]
                self.lambda_star[k] = self.zeta + (X[mask] * w[:, None]).sum(0)
            else:
                self.lambda_star[k] = self.zeta
        self.lambda_star = np.maximum(self.lambda_star, 0.1)

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
                n_clusters=min(self.K, N), n_init=10, max_iter=300,
                random_state=int(rng.integers(0, 2**31)))
            labels = km.fit_predict(X_log)
            r      = np.zeros((N, self.K))
            r[np.arange(N), labels] = 1.0
        except Exception:
            r  = rng.random((N, self.K))
            r /= r.sum(axis=1, keepdims=True)
        return r

    # ─────────────────────────────────────────────────────────────────────
    # NIG stick-breaking updates
    # ─────────────────────────────────────────────────────────────────────

    def _update_nig_sticks(self):
        """
        NIG CAVI update.

        The NIG EPPF (Barrios et al. 2013, Proposition 3.2) gives the
        variational Beta updates for the truncated stick-breaking:

            a_k = (1 − σ) + r_k        [posterior shape from data]
            b_k = (1 − σ) + mass_above  [prior pull + remaining mass]

        where mass_above_k = Σ_{j>k} r_j.

        The nig_alpha parameter scales both a_k and b_k via an additive
        term α/K_max to the prior part, mirroring the DP concentration:

            a_k = (1 − σ) + α/K + r_k
            b_k = (1 − σ) + α/K + mass_above_k

        This preserves symmetry of the Beta prior while scaling the
        effective sample size of the prior.
        """
        sigma         = self.nig_sigma
        rk            = self.r.sum(axis=0)               # (K,)
        mass_above    = np.cumsum(rk[::-1])[::-1] - rk   # Σ_{j>k} r_j
        prior_term    = (1.0 - sigma) + self.nig_alpha / self.K
        self.nig_a    = np.maximum(prior_term + rk,          EPS)
        self.nig_b    = np.maximum(prior_term + mass_above,  EPS)

    # ─────────────────────────────────────────────────────────────────────
    # Cached expectations
    # ─────────────────────────────────────────────────────────────────────

    def _clear_cache(self):
        self._cache = {}

    def _E_log_pi(self):
        """E[log π_k] under NIG stick-breaking."""
        if 'Elpi' not in self._cache:
            self._cache['Elpi'] = _nig_elpi(self.nig_a, self.nig_b)
        return self._cache['Elpi']

    def _E_log_alpha(self):
        """(K, S): ψ(λ_{kj}) − ψ(Σ_j λ_{kj})"""
        if 'Ela' not in self._cache:
            s = self.lambda_star.sum(axis=1, keepdims=True)
            self._cache['Ela'] = digamma(self.lambda_star) - digamma(s)
        return self._cache['Ela']

    def _E_log_beta(self):
        """(S,): ψ(ι_j) − ψ(Σ_j ι_j)"""
        if 'Elb' not in self._cache:
            self._cache['Elb'] = (digamma(self.iota_star)
                                  - digamma(self.iota_star.sum()))
        return self._cache['Elb']

    # ─────────────────────────────────────────────────────────────────────
    # Expected log-likelihood
    # ─────────────────────────────────────────────────────────────────────

    def _expected_log_lik(self, X):
        """(N, K): Σ_j f[i,j]*x[i,j]*E[log α_kj] + (1-f)*x*E[log β_j]"""
        E_log_a = self._E_log_alpha()   # (K, S)
        E_log_b = self._E_log_beta()    # (S,)
        f       = self.f                # (N, S)
        fX      = X * f
        ll      = fX @ E_log_a.T                          # (N, K)
        ll     += (X * (1.0 - f) @ E_log_b)[:, None]     # (N, 1)
        return ll

    # ─────────────────────────────────────────────────────────────────────
    # CAVI updates
    # ─────────────────────────────────────────────────────────────────────

    def _update_r(self, X, beta=1.0):
        E_log_pi = self._E_log_pi()
        ll       = self._expected_log_lik(X)
        log_r    = beta * (E_log_pi[None, :] + ll)
        log_r   -= logsumexp(log_r, axis=1, keepdims=True)
        self.r   = np.maximum(np.exp(log_r), EPS)
        self.r  /= self.r.sum(axis=1, keepdims=True)

    def _update_f(self, X):
        xi_sum      = self.xi_star.sum(axis=2)
        E_xi1       = digamma(self.xi_star[:, :, 0]) - digamma(xi_sum)
        E_xi2       = digamma(self.xi_star[:, :, 1]) - digamma(xi_sum)
        el_alpha_ni = self.r @ self._E_log_alpha()     # (N, S)
        log_odds    = np.clip(
            (E_xi1 + X * el_alpha_ni) - (E_xi2 + X * self._E_log_beta()[None, :]),
            -500, 500)
        self.f      = np.clip(expit(log_odds), EPS, 1 - EPS)

    def _update_xi_star(self):
        self.xi_star[:, :, 0] = np.maximum(self.xi_1 + self.f,         EPS)
        self.xi_star[:, :, 1] = np.maximum(self.xi_2 + (1.0 - self.f), EPS)

    def _update_lambda_star(self, X):
        self.lambda_star = np.maximum(
            self.zeta + self.r.T @ (self.f * X), 0.1)

    def _update_iota_star(self, X):
        self.iota_star = np.maximum(
            self.eta + ((1.0 - self.f) * X).sum(axis=0), 0.1)

    def _cavi_sweep(self, X, beta=1.0):
        self._clear_cache()
        self._update_r(X, beta=beta)
        self._update_nig_sticks()
        self._update_lambda_star(X)
        self._update_iota_star(X)
        self._update_f(X)
        self._update_xi_star()

    # ─────────────────────────────────────────────────────────────────────
    # Soft DPP repulsion
    # ─────────────────────────────────────────────────────────────────────

    def _repulsion(self):
        """
        Soft DPP repulsion bonus in ELBO:
            rep = −λ_rep · Σ_{k<j} cos²(λ_k*, λ_j*)

        Always ≤ 0 (penalises aligned profiles).
        Enters ELBO scalar comparisons in split/merge/delete.
        Does not modify the CAVI parameter updates themselves.
        """
        if self.lambda_rep <= 0.0 or self.K <= 1:
            return 0.0
        norms = np.linalg.norm(self.lambda_star, axis=1, keepdims=True)
        norms = np.maximum(norms, EPS)
        L_n   = self.lambda_star / norms          # (K, S) unit-norm profiles
        G     = L_n @ L_n.T                       # (K, K) cosine Gram matrix
        mask  = np.triu(np.ones((self.K, self.K), dtype=bool), k=1)
        return -self.lambda_rep * float(np.sum(G[mask] ** 2))

    # ─────────────────────────────────────────────────────────────────────
    # ELBO
    # ─────────────────────────────────────────────────────────────────────

    def _compute_elbo_fast(self, X):
        """Lightweight ELBO for convergence monitoring."""
        ll   = self._expected_log_lik(X)
        Elpi = self._E_log_pi()
        core = (float(np.sum(self.r * ll))
                + float(np.sum(self.r * Elpi[None, :]))
                - float(np.sum(self.r * _safe_log(self.r))))
        return core + self._repulsion()

    def _compute_elbo(self, X):
        """Full ELBO including all KL terms + DPP repulsion."""

        # 1. E[log p(X|Z,α,β,f)]
        ll    = self._expected_log_lik(X)
        term1 = float(np.sum(self.r * ll))

        # 2. E[log p(Z|π)]
        Elpi  = self._E_log_pi()
        term2 = float(np.sum(self.r * Elpi[None, :]))

        # 3. −KL[q(V) || NIG prior Beta(1−σ, 1−σ)]
        term3 = -_nig_kl(self.nig_a, self.nig_b, self.nig_sigma)

        # 4. −KL[q(α_k) || Dir(ζ·1_S)]
        lam      = self.lambda_star
        ela      = self._E_log_alpha()
        kl_alpha = (gammaln(self.zeta * self.S) - self.S * gammaln(self.zeta)
                    - (gammaln(lam.sum(1)) - gammaln(lam).sum(1))
                    + ((lam - self.zeta) * ela).sum(1))
        term4    = -float(kl_alpha.sum())

        # 5. −KL[q(β) || Dir(η·1_S)]
        iota    = self.iota_star
        elb     = self._E_log_beta()
        kl_beta = (gammaln(self.eta * self.S) - self.S * gammaln(self.eta)
                   - (gammaln(iota.sum()) - gammaln(iota).sum())
                   + ((iota - self.eta) * elb).sum())
        term5   = -float(kl_beta)

        # 6. −KL[q(f) || Beta(ξ_1, ξ_2)]
        a_q  = self.xi_star[:, :, 0]
        b_q  = self.xi_star[:, :, 1]
        xst  = a_q + b_q
        kl_f = (gammaln(self.xi_1 + self.xi_2)
                - gammaln(self.xi_1) - gammaln(self.xi_2)
                - gammaln(xst) + gammaln(a_q) + gammaln(b_q)
                + (a_q - self.xi_1) * (digamma(a_q) - digamma(xst))
                + (b_q - self.xi_2) * (digamma(b_q) - digamma(xst)))
        term6 = -float(kl_f.sum())

        # 7. −E[log q(Z)]
        term7 = -float(np.sum(self.r * _safe_log(self.r)))

        # 8. Soft DPP repulsion bonus
        term8 = self._repulsion()

        return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8

    # ─────────────────────────────────────────────────────────────────────
    # Split-Merge-Delete moves
    # ─────────────────────────────────────────────────────────────────────

    def _cosine_sim(self, k1, k2):
        a, b  = self.lambda_star[k1], self.lambda_star[k2]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > EPS else 0.0

    def _sym_kl_profiles(self, k1, k2):
        p1 = self.lambda_star[k1] / (self.lambda_star[k1].sum() + EPS)
        p2 = self.lambda_star[k2] / (self.lambda_star[k2].sum() + EPS)
        p1 = np.maximum(p1, EPS);  p2 = np.maximum(p2, EPS)
        kl12 = float(np.sum(p1 * (np.log(p1) - np.log(p2))))
        kl21 = float(np.sum(p2 * (np.log(p2) - np.log(p1))))
        return 0.5 * (kl12 + kl21)

    # ── Split ─────────────────────────────────────────────────────────────

    def _try_split(self, X, rng):
        if self.K >= self.K_max:
            return False
        rk      = self.r.sum(axis=0)
        k_split = int(np.argmax(rk))
        members = np.where(self.r[:, k_split] > 0.1)[0]
        if len(members) < 4:
            return False
        elbo_before  = self._compute_elbo(X)
        state_before = self._snapshot()
        try:
            X_mem = np.log1p(X[members])
            km2   = cluster.KMeans(n_clusters=2, n_init=5, max_iter=100,
                                   random_state=int(rng.integers(0, 2**31)))
            sub   = km2.fit_predict(X_mem)
        except Exception:
            return False
        mask_a = members[sub == 0]
        mask_b = members[sub == 1]
        if len(mask_a) == 0 or len(mask_b) == 0:
            return False
        r_new = np.full((self.N, self.K + 1), EPS)
        r_new[:, :self.K] = self.r.copy()
        r_new[mask_a, k_split]   = self.r[mask_a, k_split]
        r_new[mask_b, k_split]   = EPS
        r_new[mask_b, self.K]    = self.r[mask_b, k_split]
        r_new[mask_a, self.K]    = EPS
        r_new = np.maximum(r_new, EPS)
        r_new /= r_new.sum(axis=1, keepdims=True)
        self.r = r_new
        new_lam_a = np.maximum(
            self.zeta + (X[mask_a] * r_new[mask_a, k_split:k_split+1]).sum(0), 0.1)
        new_lam_b = np.maximum(
            self.zeta + (X[mask_b] * r_new[mask_b, self.K:self.K+1]).sum(0), 0.1)
        self.lambda_star = np.vstack([self.lambda_star, new_lam_b[None, :]])
        self.lambda_star[k_split] = new_lam_a
        self.K += 1
        self._update_nig_sticks()
        for _ in range(8):
            self._cavi_sweep(X, beta=1.0)
        self._clear_cache()
        elbo_after = self._compute_elbo(X)
        if elbo_after >= elbo_before:
            if self.verbose >= 1:
                print(f"    Split k{k_split} accepted: "
                      f"ELBO {elbo_before:.2f}→{elbo_after:.2f}, K={self.K}")
            return True
        self._restore_snapshot(state_before)
        self._clear_cache()
        return False

    # ── Merge ─────────────────────────────────────────────────────────────

    def _try_merge(self, X):
        if self.K <= self._min_K:
            return False
        best_sim, best_pair = -np.inf, None
        for i in range(self.K):
            for j in range(i + 1, self.K):
                s  = self._cosine_sim(i, j)
                kl = self._sym_kl_profiles(i, j)
                if (s > 0.95 or kl < 0.1) and s > best_sim:
                    best_sim, best_pair = s, (i, j)
        if best_pair is None:
            return False
        k1, k2 = best_pair
        elbo_before  = self._compute_elbo(X)
        state_before = self._snapshot()
        self._do_merge(k1, k2, X)
        for _ in range(8):
            self._cavi_sweep(X, beta=1.0)
        self._clear_cache()
        elbo_after = self._compute_elbo(X)
        if elbo_after >= elbo_before:
            if self.verbose >= 1:
                print(f"    Merge k{k1}+k{k2} accepted: "
                      f"ELBO {elbo_before:.2f}→{elbo_after:.2f}, K={self.K}")
            return True
        self._restore_snapshot(state_before)
        self._clear_cache()
        return False

    def _do_merge(self, k1, k2, X):
        r_new  = self.r[:, k1] + self.r[:, k2]
        keep   = [k for k in range(self.K) if k != k2]
        k1_new = keep.index(k1)
        self.r            = self.r[:, keep]
        self.r[:, k1_new] = r_new
        self.r            = np.maximum(self.r, EPS)
        self.r           /= self.r.sum(axis=1, keepdims=True)
        self.lambda_star  = self.lambda_star[keep]
        self.K            = len(keep)
        self._update_nig_sticks()
        self.lambda_star[k1_new] = np.maximum(
            self.zeta + self.r[:, k1_new] @ (self.f * X), 0.1)

    # ── Delete ────────────────────────────────────────────────────────────

    def _try_delete(self, X):
        if self.K <= self._min_K:
            return False
        rk    = self.r.sum(axis=0)
        k_del = int(np.argmin(rk))
        elbo_before  = self._compute_elbo(X)
        state_before = self._snapshot()
        self._do_delete(k_del)
        for _ in range(5):
            self._cavi_sweep(X, beta=1.0)
        self._clear_cache()
        elbo_after = self._compute_elbo(X)
        if elbo_after >= elbo_before:
            if self.verbose >= 1:
                print(f"    Delete k{k_del} accepted: "
                      f"ELBO {elbo_before:.2f}→{elbo_after:.2f}, K={self.K}")
            return True
        self._restore_snapshot(state_before)
        self._clear_cache()
        return False

    def _do_delete(self, k_del):
        keep   = [k for k in range(self.K) if k != k_del]
        extra  = self.r[:, k_del:k_del+1]
        r_rest = self.r[:, keep]
        r_sum  = r_rest.sum(axis=1, keepdims=True) + EPS
        self.r = np.maximum(r_rest + extra * r_rest / r_sum, EPS)
        self.r /= self.r.sum(axis=1, keepdims=True)
        self.lambda_star = self.lambda_star[keep]
        self.K           = len(keep)
        self._update_nig_sticks()

    # ─────────────────────────────────────────────────────────────────────
    # Snapshot / restore
    # ─────────────────────────────────────────────────────────────────────

    def _snapshot(self):
        return dict(K=self.K,
                    r=self.r.copy(),
                    f=self.f.copy(),
                    nig_a=self.nig_a.copy(),
                    nig_b=self.nig_b.copy(),
                    lambda_star=self.lambda_star.copy(),
                    iota_star=self.iota_star.copy(),
                    xi_star=self.xi_star.copy())

    def _restore_snapshot(self, s):
        self.K           = s['K']
        self.r           = s['r']
        self.f           = s['f']
        self.nig_a       = s['nig_a']
        self.nig_b       = s['nig_b']
        self.lambda_star = s['lambda_star']
        self.iota_star   = s['iota_star']
        self.xi_star     = s['xi_star']

    # ─────────────────────────────────────────────────────────────────────
    # Single-restart fit
    # ─────────────────────────────────────────────────────────────────────

    def _fit_one(self, X, rng, restart_id):
        self._initialize(X, rng)
        if self.verbose >= 1:
            print(f"\n  Restart {restart_id+1}/{self.n_restarts}  "
                  f"(K_init={self.K})")
        t0          = time()
        anneal_done = False

        for iteration in range(1, self.max_iter + 1):
            self.n_iter = iteration

            # Annealing schedule
            if self.beta_start < 1.0 and iteration <= self.anneal_iters:
                beta = (self.beta_start
                        + (self.beta_end - self.beta_start)
                        * (iteration - 1) / max(self.anneal_iters - 1, 1))
                anneal_done = iteration == self.anneal_iters
            else:
                beta        = self.beta_end
                anneal_done = True

            self._cavi_sweep(X, beta=beta)

            # Split-Merge-Delete (after annealing)
            if anneal_done and iteration % self.merge_every == 0:
                moved = self._try_split(X, rng);  self._clear_cache()
                moved |= self._try_merge(X);       self._clear_cache()
                moved |= self._try_delete(X);      self._clear_cache()
                if moved:
                    self.elbo_history = []

            # ELBO + convergence
            if iteration % 10 == 0:
                elbo = self._compute_elbo_fast(X)
                self.elbo_history.append(elbo)
                if self.verbose >= 2:
                    print(f"    iter {iteration:4d}: ELBO={elbo:.2f}, "
                          f"K={self.K}, β={beta:.3f}, t={time()-t0:.1f}s")
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

        converged_state = self._snapshot()
        converged_elbo  = self._compute_elbo(X)

        # Final exhaustive split-merge-delete passes
        for _ in range(self.K_max):
            improved  = self._try_split(X, rng);  self._clear_cache()
            improved |= self._try_merge(X);        self._clear_cache()
            improved |= self._try_delete(X);       self._clear_cache()
            if not improved:
                break

        final_elbo = self._compute_elbo(X)
        if self.verbose >= 1:
            print(f"  → Final: K={self.K}, ELBO={final_elbo:.2f}")

        return final_elbo, converged_elbo, converged_state

    # ─────────────────────────────────────────────────────────────────────
    # Public fit
    # ─────────────────────────────────────────────────────────────────────

    def fit(self, X):
        """
        Fit DMM-SVVS-NIG via CAVI with NIG prior, annealing,
        soft DPP repulsion, and split-merge-delete moves.
        """
        X          = check_array(X, dtype=np.float64)
        rng_master = np.random.default_rng(self.random_state)

        if self.verbose >= 1:
            print(f"\nDMM-SVVS-NIG v5  |  N={X.shape[0]}, S={X.shape[1]}, "
                  f"K_max={self.K_max}, σ={self.nig_sigma:.3f}, "
                  f"α={self.nig_alpha:.2f}, λ_rep={self.lambda_rep:.3f}, "
                  f"n_restarts={self.n_restarts}")
            print("=" * 70)

        t_total        = time()
        all_candidates = []

        def _icl(elbo):
            h = float(np.sum(self.r * _safe_log(self.r)))
            return elbo + h

        for restart in range(self.n_restarts):
            rng = np.random.default_rng(int(rng_master.integers(0, 2**31)))
            self._reset_state()
            _, _, conv_state = self._fit_one(X, rng, restart)

            # Candidate 1: post-exhaustive-passes state
            self._clear_cache()
            icl_final = _icl(self._compute_elbo(X))
            all_candidates.append((icl_final, self._snapshot()))

            # Candidate 2: post-convergence state (before exhaustive passes)
            self._restore_snapshot(conv_state)
            self._clear_cache()
            icl_conv = _icl(self._compute_elbo(X))
            all_candidates.append((icl_conv, conv_state))

        best_icl, best_state = max(all_candidates, key=lambda x: x[0])
        self._restore_snapshot(best_state)
        self._clear_cache()

        # Derive weights from NIG stick-breaking expectations
        elpi = self._E_log_pi()
        self.weights_ = np.exp(elpi - logsumexp(elpi))

        if self.verbose >= 1:
            print(f"\nBest across {self.n_restarts} restarts (ICL):")
            print(f"  K={self.K}, ICL={best_icl:.2f}, "
                  f"weights={np.round(self.weights_, 3)}, "
                  f"time={time()-t_total:.2f}s")
        return self

    # ─────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────

    def predict(self, X):
        X          = check_array(X, dtype=np.float64)
        N_new      = X.shape[0]
        r_o, N_o, f_o = self.r, self.N, self.f
        self.N     = N_new
        self.r     = np.ones((N_new, self.K)) / self.K
        self.f     = np.full((N_new, self.S), 0.5)
        self._clear_cache()
        ll         = self._expected_log_lik(X)
        log_r      = self._E_log_pi()[None, :] + ll
        log_r     -= logsumexp(log_r, axis=1, keepdims=True)
        labels     = np.exp(log_r).argmax(axis=1)
        self.r, self.N, self.f = r_o, N_o, f_o
        self._clear_cache()
        return labels

    # ─────────────────────────────────────────────────────────────────────
    # Inspection
    # ─────────────────────────────────────────────────────────────────────

    def get_selected_features(self, threshold=0.5):
        """OTU indices selected in > threshold fraction of samples."""
        return np.where((self.f > 0.5).mean(axis=0) > threshold)[0].tolist()

    def get_cluster_profiles(self):
        profiles = {}
        for k in range(self.K):
            lam = self.lambda_star[k]
            profiles[k] = dict(mean_profile=(lam / lam.sum()).tolist(),
                               n_samples=float(self.r[:, k].sum()),
                               weight=float(self.weights_[k]))
        return profiles


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print("Smoke-test — DMM_SVVS_Variational_v5  (NIG + Soft DPP repulsion)")
    print("=" * 70)

    rng          = np.random.default_rng(42)
    N, S, K_true = 150, 200, 3

    alpha = np.full((K_true, S), 0.1)
    block = S // K_true
    for k in range(K_true):
        alpha[k, k*block:(k+1)*block] = 3.0
    true_labels = rng.choice(K_true, size=N)
    X = np.array([rng.multinomial(5000, rng.dirichlet(alpha[true_labels[i]]))
                  for i in range(N)], dtype=float)

    print("\n--- v5 (NIG + Soft DPP repulsion) ---")
    model_v5 = DMM_SVVS_Variational_v5(
        K_max=10,
        nig_sigma=0.5, nig_alpha=1.0,
        zeta=0.1, eta=0.1, xi_1=2.0, xi_2=1.0, selection_prior=0.5,
        lambda_rep=0.2,
        beta_start=0.2, anneal_iters=60,
        merge_every=15, n_restarts=5, max_iter=300, verbose=1, random_state=42)
    model_v5.fit(X)
    pred_v5 = model_v5.predict(X)
    ari_v5  = adjusted_rand_score(true_labels, pred_v5)
    nmi_v5  = normalized_mutual_info_score(true_labels, pred_v5)
    print(f"\nARI  = {ari_v5:.3f}   NMI  = {nmi_v5:.3f}")
    print(f"K    = {model_v5.K}  (true {K_true})")
    print(f"Selected OTUs: {len(model_v5.get_selected_features())} / {S}")

    try:
        from DMM_SVVS_Variational_v4 import DMM_SVVS_Variational_v4
        print("\n--- v4 (MFM, baseline) ---")
        model_v4 = DMM_SVVS_Variational_v4(
            K_max=10, mfm_delta=1.0, mfm_gamma=2.0,
            zeta=0.1, eta=0.1, xi_1=2.0, xi_2=1.0, selection_prior=0.5,
            beta_start=0.2, anneal_iters=60,
            merge_every=15, n_restarts=5, max_iter=300, verbose=1, random_state=42)
        model_v4.fit(X)
        pred_v4 = model_v4.predict(X)
        ari_v4  = adjusted_rand_score(true_labels, pred_v4)
        nmi_v4  = normalized_mutual_info_score(true_labels, pred_v4)
        print(f"\nARI  = {ari_v4:.3f}   NMI  = {nmi_v4:.3f}")
        print(f"K    = {model_v4.K}  (true {K_true})")
        print(f"\nΔARI (v5 − v4) = {ari_v5 - ari_v4:+.3f}")
    except ImportError:
        print("\n(v4 not found — skipping comparison)")
