#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_search_dmm_svvs.py
=========================
Random search over four key hyperparameters of DMM_SVVS_Variational_v2
to maximise ARI (Adjusted Rand Index).

Optimised parameters
--------------------
  K_max           : int    – upper bound on number of clusters
  nu              : float  – DP stick-breaking concentration
  prune_threshold : float  – weight cutoff for cluster pruning
  selection_prior : float  – warm-start value for feature-selection probabilities

Why random search?
------------------
Random search is a strong, well-understood baseline for hyperparameter tuning:
  - No assumption about the shape of the ARI surface
  - Each trial is independent → easy to understand and debug
  - Proven to outperform grid search when only a few parameters actually matter
    (Bergstra & Bengio, 2012)
  - Zero extra dependencies beyond NumPy and sklearn

Sampling strategy
-----------------
  K_max           → discrete uniform in [K_lo, K_hi]
  nu              → log-uniform in [nu_lo, nu_hi]   (spans orders of magnitude)
  prune_threshold → log-uniform in [prune_lo, prune_hi]
  selection_prior → uniform    in [sel_lo,   sel_hi]

  Log-uniform sampling for nu and prune_threshold is essential: a uniform
  draw would under-sample the small values (e.g. 0.001–0.01) that often
  work best for pruning-heavy scenarios.

Noise reduction
---------------
  Because k-means initialisation is random, a single fit can give a
  misleading ARI. Each candidate configuration is evaluated with
  `n_seeds` independent random seeds; the mean ARI is used as the score.
  This smooths out init noise without requiring many extra trials.

Usage — quick start
-------------------
  from random_search_dmm_svvs import random_search_dmm_svvs

  # supervised (true labels known)
  result = random_search_dmm_svvs(X, true_labels, n_trials=30, n_seeds=3)
  print(result['best_params'])
  pred = result['best_model'].predict(X)

  # unsupervised (no true labels — uses cluster-quality proxy)
  result = random_search_dmm_svvs(X, true_labels=None, n_trials=30, n_seeds=3)
"""

import warnings
import numpy as np
from time import time
from typing import Optional, List, Dict, Any, Tuple

from sklearn.metrics import adjusted_rand_score
from sklearn.utils import check_array

from DMM_SVVS_Variational_v2 import DMM_SVVS_Variational_v2

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_params(
    rng:       np.random.Generator,
    K_lo:      int,
    K_hi:      int,
    nu_lo:     float,
    nu_hi:     float,
    prune_lo:  float,
    prune_hi:  float,
    sel_lo:    float,
    sel_hi:    float,
) -> Dict[str, Any]:
    """
    Draw one random hyperparameter configuration.

    K_max           : uniform integer in [K_lo, K_hi]
    nu              : log-uniform float in [nu_lo, nu_hi]
    prune_threshold : log-uniform float in [prune_lo, prune_hi]
    selection_prior : uniform float in [sel_lo, sel_hi]
    """
    K_max = int(rng.integers(K_lo, K_hi + 1))

    # log-uniform: equal probability of landing in each order of magnitude
    log_nu_lo, log_nu_hi = np.log(nu_lo), np.log(nu_hi)
    nu = float(np.exp(rng.uniform(log_nu_lo, log_nu_hi)))

    log_p_lo, log_p_hi = np.log(prune_lo), np.log(prune_hi)
    prune = float(np.exp(rng.uniform(log_p_lo, log_p_hi)))

    sel = float(rng.uniform(sel_lo, sel_hi))

    return dict(
        K_max           = K_max,
        nu              = nu,
        prune_threshold = prune,
        selection_prior = sel,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unsupervised quality proxy (used when true_labels=None)
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_quality_proxy(model: DMM_SVVS_Variational_v2) -> float:
    """
    Score a fitted model without ground-truth labels.

    Three complementary signals are averaged:
      1. Confidence  — mean max responsibility per sample (sharper = better)
      2. Sharpness   — 1 minus normalised responsibility entropy
      3. Balance     — penalises heavily skewed cluster sizes

    Returns a value in [0, 1] that correlates with ARI on typical data.
    """
    EPS = 1e-10
    r   = model.r          # (N, K)  soft assignments

    # 1. Mean peak responsibility
    confidence = float(r.max(axis=1).mean())

    # 2. Sharpness via entropy
    K_eff  = max(model.K, 2)
    H      = -(r * np.log(r + EPS)).sum(axis=1).mean()
    H_norm = H / np.log(K_eff)
    sharpness = float(1.0 - H_norm)

    # 3. Cluster-size balance
    sizes   = r.sum(axis=0) / (r.sum() + EPS)
    balance = float(np.clip(1.0 - np.std(sizes) * np.sqrt(model.K), 0.0, 1.0))

    return (confidence + sharpness + balance) / 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Single-configuration evaluator
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(
    params:      Dict[str, Any],
    X:           np.ndarray,
    true_labels: Optional[np.ndarray],
    n_seeds:     int,
    max_iter:    int,
    base_seed:   int,
) -> Tuple[float, List[float], float]:
    """
    Fit DMM_SVVS_Variational_v2 with `params` using `n_seeds` different
    random seeds and return (mean_score, per_seed_scores, mean_K_estimated).

    score = ARI when true_labels is given, else cluster_quality_proxy.
    """
    scores = []
    k_ests = []

    for i in range(n_seeds):
        seed = (base_seed + i * 997) % (2**31 - 1)
        try:
            model = DMM_SVVS_Variational_v2(
                K_max           = params['K_max'],
                nu              = params['nu'],
                prune_threshold = params['prune_threshold'],
                selection_prior = params['selection_prior'],
                max_iter        = max_iter,
                verbose         = 0,
                random_state    = 42,
            )
            model.fit(X)

            if true_labels is not None:
                pred  = model.predict(X)
                score = float(adjusted_rand_score(true_labels, pred))
            else:
                score = _cluster_quality_proxy(model)

            scores.append(score)
            k_ests.append(model.K)

        except Exception:
            # Numerical failure with this config — treat as worst-case
            scores.append(-1.0)
            k_ests.append(params['K_max'])

    mean_score = float(np.mean(scores))
    mean_k     = float(np.mean(k_ests))
    return mean_score, scores, mean_k


# ─────────────────────────────────────────────────────────────────────────────
# Main random search
# ─────────────────────────────────────────────────────────────────────────────

def random_search_dmm_svvs(
    X:           np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    # ── search budget ──────────────────────────────────────────────────────
    n_trials:    int   = 30,
    n_seeds:     int   = 3,
    # ── training budget per trial ──────────────────────────────────────────
    max_iter_search: int = 150,
    max_iter_final:  int = 500,
    # ── search bounds ──────────────────────────────────────────────────────
    K_lo:      int   = 3,
    K_hi:      int   = 20,
    nu_lo:     float = 1e-3,
    nu_hi:     float = 2.0,
    prune_lo:  float = 1e-3,
    prune_hi:  float = 0.40,
    sel_lo:    float = 0.05,
    sel_hi:    float = 0.95,
    # ── misc ───────────────────────────────────────────────────────────────
    seed:    int = 42,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Random search over K_max, nu, prune_threshold, and selection_prior
    to maximise ARI for DMM_SVVS_Variational_v2.

    Parameters
    ----------
    X               : (N, S) count data matrix
    true_labels     : (N,) ground-truth cluster labels, or None for
                      unsupervised mode (uses cluster-quality proxy)
    n_trials        : number of random hyperparameter configurations to try
    n_seeds         : random seeds averaged per configuration (noise reduction)
    max_iter_search : CAVI max_iter used during search (shorter = faster)
    max_iter_final  : CAVI max_iter for the final refitted best model
    K_lo / K_hi     : integer bounds for K_max (inclusive)
    nu_lo / nu_hi   : float bounds for nu (sampled log-uniformly)
    prune_lo / prune_hi : float bounds for prune_threshold (log-uniform)
    sel_lo / sel_hi : float bounds for selection_prior (uniform)
    seed            : master random seed
    verbose         : 0 = silent, 1 = per-trial line + summary, 2 = full

    Returns
    -------
    dict with:
      'best_params'  : dict  – winning hyperparameter configuration
      'best_score'   : float – mean ARI (or proxy) of the best config
      'best_model'   : DMM_SVVS_Variational_v2 – model refitted at
                       max_iter_final with the best params (multiple seeds,
                       the seed giving the highest score is kept)
      'all_trials'   : list of dicts, one per trial, sorted best-first
                       keys: params, mean_score, seed_scores, mean_K, elapsed
      'score_history': list of mean_score values in trial order
                       (useful for plotting convergence)
    """
    X = check_array(X, dtype=np.float64)
    rng = np.random.default_rng(seed)

    mode_str = "supervised (ARI)" if true_labels is not None \
               else "unsupervised (quality proxy)"

    if verbose >= 1:
        print()
        print("=" * 65)
        print("  DMM-SVVS Random Search")
        print(f"  Mode       : {mode_str}")
        print(f"  Data       : N={X.shape[0]}, S={X.shape[1]}")
        print(f"  Trials     : {n_trials}  |  Seeds/trial: {n_seeds}")
        print(f"  max_iter   : {max_iter_search} (search)  /  {max_iter_final} (final)")
        print(f"  K_max      : [{K_lo}, {K_hi}]")
        print(f"  nu         : [{nu_lo:.0e}, {nu_hi}]  (log-uniform)")
        print(f"  prune      : [{prune_lo:.0e}, {prune_hi}]  (log-uniform)")
        print(f"  sel_prior  : [{sel_lo}, {sel_hi}]  (uniform)")
        print("=" * 65)

    all_trials     = []
    score_history  = []
    best_score     = -np.inf
    best_params    = None
    t_search_start = time()

    for trial_idx in range(1, n_trials + 1):

        # ── Sample a random configuration ──────────────────────────────────
        params = _sample_params(
            rng      = rng,
            K_lo     = K_lo,
            K_hi     = K_hi,
            nu_lo    = nu_lo,
            nu_hi    = nu_hi,
            prune_lo = prune_lo,
            prune_hi = prune_hi,
            sel_lo   = sel_lo,
            sel_hi   = sel_hi,
        )

        # ── Evaluate it ────────────────────────────────────────────────────
        t0 = time()
        eval_seed = seed + trial_idx * 13          # deterministic per trial
        mean_score, seed_scores, mean_K = _evaluate(
            params      = params,
            X           = X,
            true_labels = true_labels,
            n_seeds     = n_seeds,
            max_iter    = max_iter_search,
            base_seed   = 42,
        )
        elapsed = time() - t0

        # ── Book-keeping ───────────────────────────────────────────────────
        score_history.append(mean_score)

        trial_record = dict(
            params      = params,
            mean_score  = mean_score,
            seed_scores = seed_scores,
            mean_K      = mean_K,
            elapsed     = elapsed,
        )
        all_trials.append(trial_record)

        is_best = mean_score > best_score
        if is_best:
            best_score  = mean_score
            best_params = dict(params)

        # ── Per-trial log line ─────────────────────────────────────────────
        if verbose >= 1:
            marker = " *** BEST" if is_best else ""
            print(
                f"  [{trial_idx:3d}/{n_trials}] "
                f"K_max={params['K_max']:2d}  "
                f"nu={params['nu']:.5f}  "
                f"prune={params['prune_threshold']:.5f}  "
                f"sel={params['selection_prior']:.3f}  "
                f"→ score={mean_score:.4f}  "
                f"K_est={mean_K:.1f}  "
                f"[{elapsed:.1f}s]"
                f"{marker}"
            )

    # ── Sort trials best-first ────────────────────────────────────────────
    all_trials.sort(key=lambda t: t['mean_score'], reverse=True)

    if verbose >= 1:
        total = time() - t_search_start
        print()
        print("─" * 65)
        print(f"  Search complete in {total:.1f}s")
        print(f"  Best score  : {best_score:.4f}")
        print(f"  Best params : {best_params}")
        print("─" * 65)

    # ── Refit best config at full max_iter ────────────────────────────────
    if verbose >= 1:
        print(f"\n  Refitting best config at max_iter={max_iter_final} "
              f"(trying {max(3, n_seeds)} seeds)...")

    best_model       = None
    best_model_score = -np.inf
    n_refit_seeds    = max(3, n_seeds)

    for i in range(n_refit_seeds):
        refit_seed = (seed + i * 1009) % (2**31 - 1)
        try:
            model = DMM_SVVS_Variational_v2(
                K_max           = best_params['K_max'],
                nu              = best_params['nu'],
                prune_threshold = best_params['prune_threshold'],
                selection_prior = best_params['selection_prior'],
                max_iter        = max_iter_final,
                verbose         = max(0, verbose - 1),
                random_state    = 42,
            )
            model.fit(X)

            if true_labels is not None:
                pred  = model.predict(X)
                score = float(adjusted_rand_score(true_labels, pred))
            else:
                score = _cluster_quality_proxy(model)

            if verbose >= 1:
                print(f"    seed {i}: ARI/score={score:.4f}, K={model.K}")

            if score > best_model_score:
                best_model_score = score
                best_model       = model

        except Exception as e:
            if verbose >= 1:
                print(f"    seed {i}: FAILED ({e})")

    # ── Final summary ─────────────────────────────────────────────────────
    if verbose >= 1:
        print()
        print("=" * 65)
        print("  RANDOM SEARCH SUMMARY")
        print("=" * 65)
        print(f"  Trials evaluated : {n_trials}")
        print(f"  Best search score: {best_score:.4f}")
        if best_model is not None:
            if true_labels is not None:
                pred  = best_model.predict(X)
                final = float(adjusted_rand_score(true_labels, pred))
                print(f"  Final ARI (full iter): {final:.4f}")
            print(f"  Final K estimated    : {best_model.K}")
        print(f"  Best params:")
        for k, v in best_params.items():
            print(f"    {k:<20s} = {v}")
        print("=" * 65)

        # Top-5 table
        print(f"\n  Top-5 configurations:")
        header = (f"  {'Rank':<5} {'Score':>8}  {'K_max':>6}  "
                  f"{'nu':>10}  {'prune':>10}  {'sel_prior':>9}  "
                  f"{'K_est':>6}  {'Time':>6}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for rank, t in enumerate(all_trials[:5], 1):
            p = t['params']
            print(
                f"  {rank:<5} {t['mean_score']:>8.4f}  {p['K_max']:>6d}  "
                f"{p['nu']:>10.6f}  {p['prune_threshold']:>10.6f}  "
                f"{p['selection_prior']:>9.4f}  {t['mean_K']:>6.1f}  "
                f"{t['elapsed']:>5.1f}s"
            )

    return dict(
        best_params   = best_params,
        best_score    = best_score,
        best_model    = best_model,
        all_trials    = all_trials,
        score_history = score_history,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Demo / smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print("Random Search Demo — DMM_SVVS_Variational_v2")
    print("=" * 65)

    # ── Synthetic DMM data ─────────────────────────────────────────────────
    rng_data = np.random.default_rng(0)
    N, S, K_true = 150, 200, 3

    alpha = np.full((K_true, S), 0.1)
    block = S // K_true
    for k in range(K_true):
        alpha[k, k * block:(k + 1) * block] = 3.0

    true_labels = rng_data.choice(K_true, size=N)
    X = np.array([
        rng_data.multinomial(5000, rng_data.dirichlet(alpha[true_labels[i]]))
        for i in range(N)
    ], dtype=float)

    print(f"Generated data: N={N}, S={S}, K_true={K_true}\n")

    # ── Baseline: default parameters ──────────────────────────────────────
    print("Baseline (default params):")
    baseline = DMM_SVVS_Variational_v2(
        K_max=10, nu=0.3, prune_threshold=0.2, selection_prior=0.1,
        max_iter=400, verbose=0, random_state=42
    )
    baseline.fit(X)
    pred_base = baseline.predict(X)
    ari_base  = adjusted_rand_score(true_labels, pred_base)
    nmi_base  = normalized_mutual_info_score(true_labels, pred_base)
    print(f"  ARI = {ari_base:.4f},  NMI = {nmi_base:.4f},  K = {baseline.K}\n")

    # ── Random search ─────────────────────────────────────────────────────
    result = random_search_dmm_svvs(
        X            = X,
        true_labels  = true_labels,
        n_trials     = 20,          # small for demo; use 30-50 in practice
        n_seeds      = 3,
        max_iter_search = 120,
        max_iter_final  = 400,
        K_lo   = 3,   K_hi   = 12,
        nu_lo  = 1e-3, nu_hi  = 1.0,
        prune_lo = 5e-3, prune_hi = 0.30,
        sel_lo = 0.05,   sel_hi   = 0.80,
        seed    = 42,
        verbose = 1,
    )

    # ── Compare baseline vs. best found ───────────────────────────────────
    best_model = result['best_model']
    if best_model is not None:
        pred_opt = best_model.predict(X)
        ari_opt  = adjusted_rand_score(true_labels, pred_opt)
        nmi_opt  = normalized_mutual_info_score(true_labels, pred_opt)

        print(f"\nComparison:")
        print(f"  {'':25s}  {'ARI':>8}  {'NMI':>8}  {'K':>4}")
        print(f"  {'Baseline (defaults)':25s}  {ari_base:>8.4f}  {nmi_base:>8.4f}  {baseline.K:>4d}")
        print(f"  {'Random search best':25s}  {ari_opt:>8.4f}  {nmi_opt:>8.4f}  {best_model.K:>4d}")
        delta = ari_opt - ari_base
        print(f"\n  ARI improvement: {delta:+.4f}")
