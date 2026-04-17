#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Search Hyperparameter Optimisation for DMM_SVVS_Variational_v2_5
========================================================================

Optimises five hyperparameters of DMM_SVVS_Variational_v2_5 by maximising
the Adjusted Rand Index (ARI) against ground-truth labels.

The only structural difference from random_search_dmm_svvs_v2_4.py is that
the two MFM-specific parameters (mfm_delta, mfm_gamma) are replaced by the
two NIG-specific parameters (nig_sigma, nig_alpha).  All sampling strategies,
output formatting, refit logic, and utility functions are preserved unchanged.

Hyperparameters searched
------------------------
  K_max            int      [low, high]         uniform integer
  nig_sigma        float    (0, 1)              uniform float
                             NIG stability exponent σ.
                             Uniform: σ lives on a bounded unit interval
                             so equal-width buckets are appropriate.
                             σ = 0.5 is the canonical NIG; values near 0
                             approach DP behaviour; values near 1 allow
                             more clusters.
  nig_alpha        float    (0, ∞)              log-uniform float
                             NIG total-mass / concentration parameter.
                             Log-uniform: equal probability mass per decade,
                             appropriate for a scale parameter > 0.
  selection_prior  float    (0, 1)              uniform float
  prune_threshold  float    (0, 1)              log-uniform float

All other model parameters (zeta, eta, xi_1, xi_2, tol, max_iter,
prune_start, prune_every, min_clusters, per_sample_f) are fixed during
the search.

Usage
-----
    from random_search_dmm_svvs_v2_5 import random_search, refit_best, print_top_k

    results = random_search(
        X           = X,           # (N, S) count matrix
        true_labels = true_labels, # (N,)   ground-truth cluster labels
        n_trials    = 60,

        # Override any range you like; defaults are used for the rest
        K_max_range           = (3, 15),
        nig_sigma_range       = (0.1, 0.9),
        nig_alpha_range       = (0.1, 5.0),
        selection_prior_range = (0.05, 0.95),
        prune_threshold_range = (1e-4, 0.1),

        master_seed = 42,
        verbose     = True,
    )

    print_top_k(results, top_k=10)

    best_model = refit_best(
        X           = X,
        true_labels = true_labels,
        best_result = results["best_result"],
        n_restarts  = 5,
        verbose     = True,
    )

Returns
-------
random_search() returns a dict with:
    best_config  : dict  — hyperparameters of the best trial
    best_result  : dict  — full result record of the best trial
    all_results  : list  — all trial records sorted by ARI descending

Each result record contains:
    ari, nmi, K_estimated, elapsed, config, trial_seed, error (or None)
"""

import json
import time as _time
import warnings

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from DMM_SVVS_Variational_v2_5 import DMM_SVVS_Variational_v2_5

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Default search ranges
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_K_MAX_RANGE           = (3, 15)       # int,   uniform
DEFAULT_NIG_SIGMA_RANGE       = (0.1, 0.9)   # float, uniform;     σ ∈ (0, 1)
DEFAULT_NIG_ALPHA_RANGE       = (0.1, 5.0)   # float, log-uniform; α > 0
DEFAULT_SELECTION_PRIOR_RANGE = (0.05, 0.95) # float, uniform
DEFAULT_PRUNE_THRESHOLD_RANGE = (1e-4, 0.1)  # float, log-uniform

# Parameters fixed for every trial during the search.
# max_iter is kept moderate to make each trial fast; refit_best uses a higher
# value for the final solution.
FIXED_PARAMS = dict(
    zeta         = 1.0,
    eta          = 1.0,
    xi_1         = 1.0,
    xi_2         = 1.0,
    tol          = 1e-4,
    max_iter     = 300,   # reduced for search speed; refit uses full budget
    prune_start  = 10,
    prune_every  = 5,
    per_sample_f = False,
    min_clusters = None,
    verbose      = 0,     # suppress per-iteration output during search
)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_int(rng, low, high):
    """Sample an integer uniformly from [low, high] (both ends inclusive)."""
    return int(rng.integers(int(low), int(high) + 1))


def _sample_uniform(rng, low, high):
    """Sample a float uniformly from [low, high)."""
    return float(rng.uniform(float(low), float(high)))


def _sample_loguniform(rng, low, high):
    """
    Sample a float log-uniformly from [low, high).

    Log-uniform sampling ensures the search spends equal probability mass
    on each decade, which is appropriate for scale parameters like
    nig_alpha and prune_threshold.
    """
    log_low  = np.log(float(low))
    log_high = np.log(float(high))
    return float(np.exp(rng.uniform(log_low, log_high)))


def _sample_config(rng,
                   K_max_range,
                   nig_sigma_range,
                   nig_alpha_range,
                   selection_prior_range,
                   prune_threshold_range):
    """
    Draw one random configuration from the given search ranges.

    nig_sigma is sampled uniformly on its bounded (0, 1) domain.
    nig_alpha must be strictly positive; log-uniform sampling from a range
    with a positive lower bound guarantees this.

    Returns
    -------
    dict with keys: K_max, nig_sigma, nig_alpha,
                    selection_prior, prune_threshold
    """
    return {
        "K_max":           _sample_int(rng, *K_max_range),
        "nig_sigma":       _sample_uniform(rng, *nig_sigma_range),
        "nig_alpha":       _sample_loguniform(rng, *nig_alpha_range),
        "selection_prior": _sample_uniform(rng, *selection_prior_range),
        "prune_threshold": _sample_loguniform(rng, *prune_threshold_range),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────

def _run_trial(X, true_labels, config, trial_seed):
    """
    Fit one model configuration and return ARI, NMI, K, elapsed time.

    The trial_seed is passed as random_state so each trial is independently
    reproducible.  On any exception (numerical failure, invalid config) the
    trial returns ARI = NMI = -1 and records the error message.

    Parameters
    ----------
    X            : (N, S) float array
    true_labels  : (N,)   int array
    config       : dict   — sampled hyperparameters
    trial_seed   : int    — random_state for this trial

    Returns
    -------
    dict with keys: ari, nmi, K_estimated, elapsed, config, trial_seed, error
    """
    params = {**FIXED_PARAMS, **config, "random_state": trial_seed}
    t0     = _time.time()
    error  = None

    try:
        model  = DMM_SVVS_Variational_v2_5(**params)
        model.fit(X)
        pred   = model.predict(X)
        ari    = float(adjusted_rand_score(true_labels, pred))
        nmi    = float(normalized_mutual_info_score(true_labels, pred))
        K_est  = int(model.K)
    except Exception as exc:
        ari, nmi, K_est = -1.0, -1.0, -1
        error = str(exc)

    elapsed = round(_time.time() - t0, 2)

    return {
        "ari":         ari,
        "nmi":         nmi,
        "K_estimated": K_est,
        "elapsed":     elapsed,
        "config":      config,
        "trial_seed":  trial_seed,
        "error":       error,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main random-search function
# ─────────────────────────────────────────────────────────────────────────────

def random_search(X,
                  true_labels,
                  n_trials              = 50,
                  K_max_range           = DEFAULT_K_MAX_RANGE,
                  nig_sigma_range       = DEFAULT_NIG_SIGMA_RANGE,
                  nig_alpha_range       = DEFAULT_NIG_ALPHA_RANGE,
                  selection_prior_range = DEFAULT_SELECTION_PRIOR_RANGE,
                  prune_threshold_range = DEFAULT_PRUNE_THRESHOLD_RANGE,
                  master_seed           = 42,
                  verbose               = True):
    """
    Random search over DMM_SVVS_Variational_v2_5 hyperparameters,
    maximising Adjusted Rand Index (ARI) against ground-truth labels.

    Parameters
    ----------
    X : np.ndarray, shape (N, S)
        Count data matrix (e.g. OTU counts).
    true_labels : np.ndarray, shape (N,)
        Ground-truth cluster labels for ARI evaluation.
    n_trials : int
        Number of random configurations to evaluate.
    K_max_range : tuple (int_low, int_high)
        Search range for the truncation level.  Both ends are inclusive.
        Example: (3, 15)
    nig_sigma_range : tuple (float_low, float_high)
        Search range for the NIG stability exponent σ ∈ (0, 1).
        Sampled uniformly; σ = 0.5 is the canonical NIG.
        Lower σ → fewer clusters (DP-like); higher σ → more clusters.
        Example: (0.1, 0.9)
    nig_alpha_range : tuple (float_low, float_high)
        Search range for the NIG concentration parameter α > 0.
        Sampled log-uniformly so that small values are well-explored.
        Larger α → more clusters a priori.
        Example: (0.1, 5.0)
    selection_prior_range : tuple (float_low, float_high)
        Search range for the f warm-start value in (0, 1).
        Sampled uniformly.  Example: (0.05, 0.95)
    prune_threshold_range : tuple (float_low, float_high)
        Search range for the weight deletion threshold.
        Sampled log-uniformly.  Example: (1e-4, 0.1)
    master_seed : int
        Seed for the search RNG — makes the entire run reproducible.
    verbose : bool
        Print a live per-trial progress table if True.

    Returns
    -------
    dict with keys:
        best_config  : dict  — hyperparameters of the best trial
        best_result  : dict  — full result record of the best trial
        all_results  : list  — all trial records sorted by ARI descending
    """
    X           = np.asarray(X, dtype=float)
    true_labels = np.asarray(true_labels)
    master_rng  = np.random.default_rng(int(master_seed))

    # ── Header ────────────────────────────────────────────────────────────
    if verbose:
        print("\nDMM-SVVS v2.5 — Random Hyperparameter Search  (scored by ARI)")
        print("=" * 90)
        print(f"  n_trials      = {n_trials},   master_seed = {master_seed}")
        print(f"  K_max            : {K_max_range}   [uniform int]")
        print(f"  nig_sigma        : {nig_sigma_range}   [uniform float, σ ∈ (0,1)]")
        print(f"  nig_alpha        : {nig_alpha_range}   [log-uniform, α > 0]")
        print(f"  selection_prior  : {selection_prior_range}  [uniform float]")
        print(f"  prune_threshold  : {prune_threshold_range}  [log-uniform]")
        print(f"  Fixed: {json.dumps(FIXED_PARAMS)}")
        print("=" * 90)
        _print_row_header()

    all_results = []
    best_ari    = -np.inf
    best_result = None
    best_config = None

    for trial in range(1, n_trials + 1):
        # Each trial gets its own deterministic seed derived from master_rng,
        # so results are fully reproducible regardless of trial order.
        config     = _sample_config(
                         master_rng,
                         K_max_range,
                         nig_sigma_range,
                         nig_alpha_range,
                         selection_prior_range,
                         prune_threshold_range)
        trial_seed = int(master_rng.integers(0, 2**31))

        result = _run_trial(X, true_labels, config, trial_seed)
        all_results.append(result)

        is_best = result["ari"] > best_ari
        if is_best:
            best_ari    = result["ari"]
            best_result = result
            best_config = config

        if verbose:
            _print_trial_row(trial, result, is_best)

    # Sort by ARI descending; break ties by NMI
    all_results.sort(key=lambda r: (r["ari"], r["nmi"]), reverse=True)

    if verbose:
        print("=" * 90)
        print(f"\n  Best ARI    : {best_ari:.4f}")
        print(f"  Best NMI    : {best_result['nmi']:.4f}")
        print(f"  Best K_est  : {best_result['K_estimated']}")
        print(f"  Best seed   : {best_result['trial_seed']}")
        print(f"  Best config :")
        for k, v in best_config.items():
            if isinstance(v, float):
                print(f"      {k:<22s} = {v:.6f}")
            else:
                print(f"      {k:<22s} = {v}")
        print(f"      {'random_state':<22s} = {best_result['trial_seed']}")

    return {
        "best_config": best_config,
        "best_result": best_result,
        "all_results": all_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Final refit with best config
# ─────────────────────────────────────────────────────────────────────────────

def refit_best(X,
               true_labels,
               best_result,
               n_restarts = 5,
               max_iter   = 500,
               verbose    = True):
    """
    Re-fit the model using the best configuration found by random_search,
    running multiple independent restarts and keeping the one with the
    highest ARI.

    The original trial_seed that produced the best ARI during the search is
    always used as one of the restart seeds, guaranteeing the search result
    is reproduced at minimum.

    Parameters
    ----------
    X            : np.ndarray, shape (N, S)
    true_labels  : np.ndarray, shape (N,)
    best_result  : dict
        The dict under results["best_result"] returned by random_search().
        Must contain keys "config" and "trial_seed".
    n_restarts   : int
        Total number of independent random restarts.
        The original trial seed counts as the first restart.
    max_iter     : int
        Maximum CAVI iterations for each restart.
        Higher than the search budget (default 500) for a more refined fit.
    verbose      : bool

    Returns
    -------
    Fitted DMM_SVVS_Variational_v2_5 model with the highest ARI.
    """
    X           = np.asarray(X, dtype=float)
    true_labels = np.asarray(true_labels)

    best_config = best_result["config"]
    trial_seed  = best_result["trial_seed"]

    # Safety clamp: a very large prune_threshold from a wide search range
    # can remove all clusters on the refit.  Cap at 0.1.
    safe_config = dict(best_config)
    if safe_config.get("prune_threshold", 0.0) > 0.1:
        if verbose:
            old_val = safe_config["prune_threshold"]
            print(f"  [refit_best] Clamping prune_threshold {old_val:.6f} → 0.1")
        safe_config["prune_threshold"] = 0.1

    # Build restart seeds: original trial seed first, then derived extras.
    rng_extra   = np.random.default_rng(trial_seed + 1)
    n_extra     = max(n_restarts - 1, 0)
    extra_seeds = rng_extra.integers(0, 2**31, size=n_extra).tolist()
    all_seeds   = [trial_seed] + extra_seeds   # length == n_restarts

    if verbose:
        print(f"\nFinal refit — n_restarts={n_restarts},  max_iter={max_iter}")
        print("=" * 76)
        print(f"  Config:")
        for k, v in safe_config.items():
            if isinstance(v, float):
                print(f"      {k:<22s} = {v:.6f}")
            else:
                print(f"      {k:<22s} = {v}")
        print()
        print(f"  {'Restart':>8}  {'Seed':>12}  {'ARI':>7}  "
              f"{'NMI':>7}  {'K':>4}  Note")
        print("  " + "-" * 60)

    best_model = None
    best_ari   = -np.inf
    best_nmi   = -np.inf

    for idx, seed in enumerate(all_seeds):
        params = {
            **FIXED_PARAMS,
            **safe_config,
            "max_iter":     max_iter,
            "random_state": seed,
            "verbose":      0,
        }
        try:
            m     = DMM_SVVS_Variational_v2_5(**params)
            m.fit(X)
            pred  = m.predict(X)
            ari_r = float(adjusted_rand_score(true_labels, pred))
            nmi_r = float(normalized_mutual_info_score(true_labels, pred))

            note = "← original trial seed" if idx == 0 else ""
            if verbose:
                marker = " *" if ari_r > best_ari else "  "
                print(f"  {idx+1:>8d}  {seed:>12d}  {ari_r:>7.4f}  "
                      f"{nmi_r:>7.4f}  {m.K:>4d}  {note}{marker}")

            # Keep the restart with the highest ARI; break ties with NMI
            if ari_r > best_ari or (ari_r == best_ari and nmi_r > best_nmi):
                best_ari   = ari_r
                best_nmi   = nmi_r
                best_model = m

        except Exception as exc:
            if verbose:
                print(f"  {idx+1:>8d}  {seed:>12d}  FAILED: {exc}")

    if best_model is None:
        raise RuntimeError("All restarts failed — check config and data.")

    if verbose:
        pred  = best_model.predict(X)
        ari   = adjusted_rand_score(true_labels, pred)
        nmi   = normalized_mutual_info_score(true_labels, pred)
        n_sel = len(best_model.get_selected_features())
        print(f"\n  Best refit ARI  = {ari:.4f}")
        print(f"  Best refit NMI  = {nmi:.4f}")
        print(f"  K estimated     = {best_model.K}")
        print(f"  Selected OTUs   = {n_sel} / {X.shape[1]}  (f>0.5 in >50% of samples)")

    return best_model


# ─────────────────────────────────────────────────────────────────────────────
# Summary table helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_row_header():
    print(f"  {'#':>5}  {'ARI':>7}  {'NMI':>7}  {'K_est':>5}  {'sec':>5}  "
          f"{'K_max':>5}  {'σ':>7}  {'α':>8}  {'sel':>6}  {'prune':>8}  {'seed':>12}")
    print("  " + "-" * 90)


def _print_trial_row(trial, result, is_best):
    c      = result["config"]
    marker = " ◀ best" if is_best else ""
    err    = f"  ERROR: {result['error']}" if result["error"] else ""
    print(f"  {trial:5d}  {result['ari']:7.4f}  {result['nmi']:7.4f}  "
          f"{result['K_estimated']:5d}  {result['elapsed']:5.1f}  "
          f"{c['K_max']:5d}  {c['nig_sigma']:7.4f}  "
          f"{c['nig_alpha']:8.4f}  "
          f"{c['selection_prior']:6.3f}  "
          f"{c['prune_threshold']:8.5f}  "
          f"{result['trial_seed']:>12d}"
          f"{marker}{err}")


def print_top_k(search_result, top_k=10):
    """
    Print a ranked summary table of the top-k configurations by ARI.

    Parameters
    ----------
    search_result : dict  — return value of random_search()
    top_k         : int   — number of configurations to show
    """
    results = search_result["all_results"][:top_k]
    n_shown = len(results)
    print(f"\nTop-{n_shown} configurations (by ARI):")
    print(f"  {'Rank':>4}  {'ARI':>7}  {'NMI':>7}  {'K_est':>5}  "
          f"{'K_max':>5}  {'σ':>7}  {'α':>8}  {'sel':>6}  {'prune':>8}  {'seed':>12}")
    print("  " + "-" * 90)
    for rank, r in enumerate(results, 1):
        c = r["config"]
        print(f"  {rank:4d}  {r['ari']:7.4f}  {r['nmi']:7.4f}  "
              f"{r['K_estimated']:5d}  "
              f"{c['K_max']:5d}  {c['nig_sigma']:7.4f}  "
              f"{c['nig_alpha']:8.4f}  "
              f"{c['selection_prior']:6.3f}  "
              f"{c['prune_threshold']:8.5f}  "
              f"{r['trial_seed']:>12d}")


def save_results(search_result, path="random_search_v2_5_results.json"):
    """
    Save all trial results to a JSON file for later analysis.

    Parameters
    ----------
    search_result : dict  — return value of random_search()
    path          : str   — output file path
    """
    import json as _json
    with open(path, "w") as fh:
        _json.dump(search_result["all_results"], fh, indent=2)
    print(f"  Results saved to {path}  ({len(search_result['all_results'])} trials)")


# ─────────────────────────────────────────────────────────────────────────────
# Demo — run when executed directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Generate synthetic block-structured DMM data ──────────────────────
    rng_data = np.random.default_rng(0)
    N, S, K_true = 150, 200, 3

    alpha_true = np.full((K_true, S), 0.1)
    block = S // K_true
    for k in range(K_true):
        alpha_true[k, k * block:(k + 1) * block] = 3.0
    true_labels = rng_data.choice(K_true, size=N)
    X = np.array(
        [rng_data.multinomial(5000, rng_data.dirichlet(alpha_true[true_labels[i]]))
         for i in range(N)],
        dtype=float
    )
    print(f"Synthetic data: N={N}, S={S}, K_true={K_true}, "
          f"class sizes={np.bincount(true_labels).tolist()}")

    # ── Run random search ─────────────────────────────────────────────────
    results = random_search(
        X           = X,
        true_labels = true_labels,
        n_trials    = 40,

        # Search ranges (override defaults as needed for your data)
        K_max_range           = (3, 12),
        nig_sigma_range       = (0.1, 0.9),
        nig_alpha_range       = (0.1, 5.0),
        selection_prior_range = (0.05, 0.95),
        prune_threshold_range = (1e-4, 0.1),

        master_seed = 42,
        verbose     = True,
    )

    # ── Summary table ─────────────────────────────────────────────────────
    print_top_k(results, top_k=10)

    # ── Save all results ──────────────────────────────────────────────────
    save_results(results, path="random_search_v2_5_results.json")

    # ── Final refit with best config ──────────────────────────────────────
    best_model = refit_best(
        X           = X,
        true_labels = true_labels,
        best_result = results["best_result"],
        n_restarts  = 5,
        max_iter    = 500,
        verbose     = True,
    )
