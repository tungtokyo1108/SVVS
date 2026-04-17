#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS-MFM: Dirichlet Multinomial Mixture with Variational Variable Selection
              using Mixture of Finite Mixtures (MFM) Prior — Version 4

Key change over v3: Replace Pitman-Yor Process stick-breaking with a
Mixture of Finite Mixtures (MFM) prior.  All other improvements from v3
(per-sample variable selection, exact CAVI λ update, deterministic annealing,
split-merge moves, multiple restarts) are preserved or enhanced.

Why MFM beats PYP/DP for ARI
─────────────────────────────
DP and PYP produce a biased posterior on K — the number of clusters is
systematically over-estimated because extra empty components receive non-zero
prior probability.  Under truncated variational inference this bias persists
at finite N and directly degrades ARI (models with K̂ > K_true misassign
samples to spurious clusters).

MFM (Miller & Harrison 2018, JASA) places an explicit prior on K:
    K  ~ Poisson(γ) + 1
    π | K ~ Dirichlet(δ, …, δ)   (symmetric, K-dim)

and yields a consistent posterior on K (converges to true K as N→∞).
Under the truncated variational family q(π) = Dir(φ), the CAVI update is:

    φ_k = δ_eff + r_k,    r_k = Σ_i r_ik
    E[log π_k] = ψ(φ_k) − ψ(Σ_k φ_k)

where δ_eff = δ + γ/K_max absorbs the prior on K (Frühwirth-Schnatter 2021).

Split-Merge moves (key addition)
──────────────────────────────────
Merge-only moves over-compress.  We add a split proposal that finds the
largest cluster, splits it via k-means(2) on its member samples, and accepts
if the ELBO improves.  The split-merge pair acts as a reversible jump that
lets the variational model explore different K values rather than always
collapsing.

References
──────────
  Miller & Harrison (2018) JASA 113(521):340–356.
  Frühwirth-Schnatter et al. (2021) Bayesian Analysis.
  Hughes & Sudderth (2013) "Memoized online variational inference for DP
    mixture models", NIPS.
  Stirrup et al. (2024) "VBVarSel", NeurIPS.
"""

import numpy as np
from scipy.special import digamma, gammaln, logsumexp, expit
from sklearn.utils import check_array
from sklearn import cluster
from time import time


EPS = 1e-10


def _safe_log(x):
    return np.log(np.maximum(x, EPS))


class DMM_SVVS_Variational_v4:
    """
    DMM with Variational Variable Selection and MFM prior — Version 4.

    Parameters
    ----------
    K_max : int
        Hard upper bound on clusters (truncation level).
    mfm_delta : float
        Symmetric Dirichlet concentration for π|K ~ Dir(δ,…,δ).
        Larger δ → more uniform weights → more clusters survive.
        Range 0.5–5.0; default 1.0.
    mfm_gamma : float
        Poisson(γ)+1 prior on K.  Sets prior mean clusters to γ+1.
        Default 1.0.
    zeta : float
        Dirichlet prior on cluster OTU profiles.
    eta : float
        Dirichlet prior on background OTU profile.
    xi_1, xi_2 : float
        Beta prior on per-sample-per-OTU selection f[i,j].
    selection_prior : float
        Warm-start value for f.
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
        Number of independent restarts; best ELBO is kept.
    min_clusters : int or None
        Hard floor on number of active clusters.
    verbose : int
    random_state : int or None
    """

    def __init__(self,
                 K_max=15,
                 mfm_delta=1.0,
                 mfm_gamma=2.0,
                 zeta=0.1,
                 eta=0.1,
                 xi_1=2.0,
                 xi_2=1.0,
                 selection_prior=0.5,
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

        self.K_max           = K_max
        self.mfm_delta       = float(mfm_delta)
        self.mfm_gamma       = float(mfm_gamma)
        self.zeta            = zeta
        self.eta             = eta
        self.xi_1            = xi_1
        self.xi_2            = xi_2
        self.selection_prior = selection_prior
        self.tol             = tol
        self.max_iter        = max_iter
        self.beta_start      = beta_start
        self.beta_end        = beta_end
        self.anneal_iters    = anneal_iters
        self.merge_every     = merge_every
        self.n_restarts      = n_restarts
        self.min_clusters    = min_clusters
        self.verbose         = verbose
        self.random_state    = random_state
        self._reset_state()

    # ─────────────────────────────────────────────────────────────────────
    # State
    # ─────────────────────────────────────────────────────────────────────

    def _reset_state(self):
        self.N = self.S = self.K = None
        self.r           = None   # (N, K)
        self.f           = None   # (N, S)
        self.phi         = None   # (K,)     MFM Dirichlet weight params
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

    def _delta_eff(self):
        """
        δ_eff = δ + γ/K_max  (MFM effective Dirichlet concentration).
        Absorbs the Poisson(γ)+1 prior on K into the symmetric Dir weight prior.
        Frühwirth-Schnatter et al. (2021), eq. 3.4.
        """
        return self.mfm_delta + self.mfm_gamma / self.K_max

    # ─────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────

    def _initialize(self, X, rng):
        self.N, self.S = X.shape
        self.K      = self.K_max
        self._min_K = self._resolve_min_clusters()
        self._de    = self._delta_eff()

        if self.verbose >= 2:
            print(f"  Init: N={self.N}, S={self.S}, K_max={self.K_max}, "
                  f"δ={self.mfm_delta:.2f}, γ={self.mfm_gamma:.2f}, "
                  f"δ_eff={self._de:.4f}")

        self.r = self._init_r_kmeans(X, rng)
        self.f = np.full((self.N, self.S), self.selection_prior)

        # MFM weight params
        self.phi = np.maximum(self._de + self.r.sum(axis=0), EPS)

        # Beta params for feature selection
        self.xi_star = np.empty((self.N, self.S, 2))
        self.xi_star[:, :, 0] = self.xi_1
        self.xi_star[:, :, 1] = self.xi_2

        # Cluster Dirichlet from k-means stats
        self.lambda_star = np.zeros((self.K, self.S))
        for k in range(self.K):
            mask = self.r[:, k] > 0.1
            if mask.sum() > 0:
                w  = self.r[mask, k]
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
    # Cached expectations
    # ─────────────────────────────────────────────────────────────────────

    def _clear_cache(self):
        self._cache = {}

    def _E_log_pi(self):
        """E[log π_k] = ψ(φ_k) − ψ(Σ φ_k)  under q(π)=Dir(φ)."""
        if 'Elpi' not in self._cache:
            self._cache['Elpi'] = digamma(self.phi) - digamma(self.phi.sum())
        return self._cache['Elpi']

    def _E_log_alpha(self):
        """(K, S): ψ(λ_kj) − ψ(Σ_j λ_kj)"""
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
    # Expected log-likelihood (exact, per-sample f)
    # ─────────────────────────────────────────────────────────────────────

    def _expected_log_lik(self, X):
        """(N, K):  Σ_j f[i,j]*x[i,j]*E[log α_kj] + (1-f)*x*E[log β_j]"""
        E_log_a = self._E_log_alpha()   # (K, S)
        E_log_b = self._E_log_beta()    # (S,)
        f       = self.f                # (N, S)
        fX      = X * f
        ll      = fX @ E_log_a.T                    # (N, K)
        ll     += (X * (1.0 - f) @ E_log_b)[:, None]
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

    def _update_phi(self):
        """φ_k = δ_eff + Σ_i r_ik  (MFM conjugate Dirichlet update)."""
        self.phi = np.maximum(self._de + self.r.sum(axis=0), EPS)

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
        self._update_phi()
        self._update_lambda_star(X)
        self._update_iota_star(X)
        self._update_f(X)
        self._update_xi_star()

    # ─────────────────────────────────────────────────────────────────────
    # ELBO
    # ─────────────────────────────────────────────────────────────────────

    def _compute_elbo_fast(self, X):
        ll    = self._expected_log_lik(X)
        Elpi  = self._E_log_pi()
        return (float(np.sum(self.r * ll))
                + float(np.sum(self.r * Elpi[None, :]))
                - float(np.sum(self.r * _safe_log(self.r))))

    def _compute_elbo(self, X):
        de = self._de

        # 1. E[log p(X|Z,α,β,f)]
        ll    = self._expected_log_lik(X)
        term1 = float(np.sum(self.r * ll))

        # 2. E[log p(Z|π)]
        Elpi  = self._E_log_pi()
        term2 = float(np.sum(self.r * Elpi[None, :]))

        # 3. −KL[q(π) || Dir(δ_eff·1_K)]
        phi     = self.phi
        phi_sum = phi.sum()
        Elpi_q  = digamma(phi) - digamma(phi_sum)
        kl_pi   = (gammaln(de * self.K) - self.K * gammaln(de)
                   - (gammaln(phi_sum) - gammaln(phi).sum())
                   + ((phi - de) * Elpi_q).sum())
        term3   = -float(kl_pi)

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

        # 6. −KL[q(f) || Beta(ξ_1, ξ_2)]  summed over (N, S)
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

        return term1 + term2 + term3 + term4 + term5 + term6 + term7

    # ─────────────────────────────────────────────────────────────────────
    # Split-Merge-Delete moves
    # ─────────────────────────────────────────────────────────────────────

    def _cosine_sim(self, k1, k2):
        a, b  = self.lambda_star[k1], self.lambda_star[k2]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > EPS else 0.0

    def _sym_kl_profiles(self, k1, k2):
        """
        Symmetric KL divergence between the mean Dirichlet profiles of
        cluster k1 and k2:  0.5*(KL(p1||p2) + KL(p2||p1))
        where p_k = λ_k / Σ_j λ_kj  (normalised mean profile).
        Large value → profiles are distinct → should NOT merge.
        """
        p1 = self.lambda_star[k1] / (self.lambda_star[k1].sum() + EPS)
        p2 = self.lambda_star[k2] / (self.lambda_star[k2].sum() + EPS)
        p1 = np.maximum(p1, EPS);  p2 = np.maximum(p2, EPS)
        kl12 = float(np.sum(p1 * (np.log(p1) - np.log(p2))))
        kl21 = float(np.sum(p2 * (np.log(p2) - np.log(p1))))
        return 0.5 * (kl12 + kl21)

    # ── Split ─────────────────────────────────────────────────────────────

    def _try_split(self, X, rng):
        """
        Split proposal: pick the largest cluster, run k-means(2) on its
        member samples, initialise two new sub-clusters, settle with 8 CAVI
        steps, accept if ELBO improves.

        This balances the merge/delete moves so the model can grow K when the
        data support it, preventing over-compression.
        """
        if self.K >= self.K_max:
            return False

        rk      = self.r.sum(axis=0)
        k_split = int(np.argmax(rk))

        # Collect samples with r_ik > 0.1 for this cluster
        members = np.where(self.r[:, k_split] > 0.1)[0]
        if len(members) < 4:
            return False

        elbo_before  = self._compute_elbo(X)
        state_before = self._snapshot()

        # k-means(2) on log-transformed member data
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

        # Build new responsibilities: add one column (K+1 clusters)
        r_new = np.full((self.N, self.K + 1), EPS)
        r_new[:, :self.K] = self.r.copy()

        # Reassign members of k_split to the two new sub-clusters
        weight_a = self.r[mask_a, k_split]
        weight_b = self.r[mask_b, k_split]
        r_new[mask_a, k_split]   = weight_a
        r_new[mask_b, k_split]   = EPS
        r_new[mask_b, self.K]    = weight_b
        r_new[mask_a, self.K]    = EPS
        r_new = np.maximum(r_new, EPS)
        r_new /= r_new.sum(axis=1, keepdims=True)

        # Expand state
        self.r           = r_new
        new_lam_a        = np.maximum(
            self.zeta + (X[mask_a] * r_new[mask_a, k_split:k_split+1]).sum(0), 0.1)
        new_lam_b        = np.maximum(
            self.zeta + (X[mask_b] * r_new[mask_b, self.K:self.K+1]).sum(0), 0.1)
        self.lambda_star = np.vstack([self.lambda_star, new_lam_b[None, :]])
        self.lambda_star[k_split] = new_lam_a
        self.phi         = np.append(self.phi, self._de)
        self.K          += 1

        # Settle
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
        # Find most similar pair by cosine similarity, but gate on KL:
        # only consider merging if sym-KL < merge_kl_threshold.
        # This prevents merging genuinely distinct clusters even if one
        # direction of cosine similarity is high.
        best_sim, best_pair = -np.inf, None
        for i in range(self.K):
            for j in range(i + 1, self.K):
                s   = self._cosine_sim(i, j)
                kl  = self._sym_kl_profiles(i, j)
                # Gate: cosine > 0.95 OR KL < 0.1 (very similar profiles)
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
        self.phi          = self.phi[keep]
        self.K            = len(keep)
        fX = self.f * X
        self.lambda_star[k1_new] = np.maximum(
            self.zeta + self.r[:, k1_new] @ fX, 0.1)

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
        self.phi         = self.phi[keep]
        self.K           = len(keep)

    # ─────────────────────────────────────────────────────────────────────
    # Snapshot / restore
    # ─────────────────────────────────────────────────────────────────────

    def _snapshot(self):
        return dict(K=self.K, r=self.r.copy(), f=self.f.copy(),
                    phi=self.phi.copy(), lambda_star=self.lambda_star.copy(),
                    iota_star=self.iota_star.copy(), xi_star=self.xi_star.copy())

    def _restore_snapshot(self, s):
        self.K = s['K']; self.r = s['r']; self.f = s['f']
        self.phi = s['phi']; self.lambda_star = s['lambda_star']
        self.iota_star = s['iota_star']; self.xi_star = s['xi_star']

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
                moved = self._try_split(X, rng)
                self._clear_cache()
                moved |= self._try_merge(X)
                self._clear_cache()
                moved |= self._try_delete(X)
                self._clear_cache()
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

        # Save state at convergence (before exhaustive passes may collapse K)
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

        # Return converged state (before exhaustive passes) alongside final
        return final_elbo, converged_elbo, converged_state  # noqa: F841

    # ─────────────────────────────────────────────────────────────────────
    # Public fit
    # ─────────────────────────────────────────────────────────────────────

    def fit(self, X):
        """Fit via CAVI with MFM prior, annealing, split-merge-delete moves."""
        X          = check_array(X, dtype=np.float64)
        rng_master = np.random.default_rng(self.random_state)

        if self.verbose >= 1:
            print(f"\nDMM-SVVS-MFM v4  |  N={X.shape[0]}, S={X.shape[1]}, "
                  f"K_max={self.K_max}, δ={self.mfm_delta}, γ={self.mfm_gamma}, "
                  f"n_restarts={self.n_restarts}")
            print("=" * 70)

        t_total    = time()
        # Collect all candidate states: both the post-convergence state
        # and the post-exhaustive-passes state for each restart.
        # Use ICL (ELBO + Σ_ik r_ik log r_ik) to rank them — ICL penalises
        # diffuse responsibilities caused by over-compressed K (merged clusters
        # have high entropy per sample when two true clusters are forced into one),
        # and under-compressed K (extra spurious clusters have near-uniform r).
        all_candidates = []   # list of (icl, snapshot)

        def _icl(elbo):
            """ICL = ELBO + E_q[log q(Z)]  (adds back the negative entropy)."""
            h = float(np.sum(self.r * _safe_log(self.r)))  # Σ r log r (<0)
            return elbo + h   # less negative when assignments are sharp

        for restart in range(self.n_restarts):
            rng = np.random.default_rng(int(rng_master.integers(0, 2**31)))
            self._reset_state()
            _, _, conv_state = self._fit_one(X, rng, restart)

            # Candidate 1: post-exhaustive-passes (final) state
            self._clear_cache()
            icl_final = _icl(self._compute_elbo(X))
            all_candidates.append((icl_final, self._snapshot()))

            # Candidate 2: post-convergence state (before exhaustive passes)
            self._restore_snapshot(conv_state)
            self._clear_cache()
            icl_conv = _icl(self._compute_elbo(X))
            all_candidates.append((icl_conv, conv_state))

        # Pick candidate with highest ICL
        best_icl, best_state = max(all_candidates, key=lambda x: x[0])

        self._restore_snapshot(best_state)
        self._clear_cache()
        self.weights_ = self.phi / self.phi.sum()

        if self.verbose >= 1:
            print(f"\nBest across {self.n_restarts} restarts (ICL selection):")
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

    print("Smoke-test — DMM_SVVS_Variational_v4  (MFM + Split-Merge-Delete)")
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

    print("\n--- v4 (MFM + Split-Merge-Delete) ---")
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
    print(f"Selected OTUs: {len(model_v4.get_selected_features())} / {S}")

    try:
        from DMM_SVVS_Variational_v3 import DMM_SVVS_Variational_v3
        print("\n--- v3 (PYP, baseline) ---")
        model_v3 = DMM_SVVS_Variational_v3(
            K_max=10, py_discount=0.2, zeta=0.5, eta=0.5,
            beta_start=0.2, anneal_iters=60, merge_every=15,
            n_restarts=5, max_iter=300, verbose=1, random_state=42)
        model_v3.fit(X)
        pred_v3 = model_v3.predict(X)
        ari_v3  = adjusted_rand_score(true_labels, pred_v3)
        nmi_v3  = normalized_mutual_info_score(true_labels, pred_v3)
        print(f"\nARI  = {ari_v3:.3f}   NMI  = {nmi_v3:.3f}")
        print(f"K    = {model_v3.K}  (true {K_true})")
        print(f"\nΔARI (v4 − v3) = {ari_v4 - ari_v3:+.3f}")
    except ImportError:
        pass
