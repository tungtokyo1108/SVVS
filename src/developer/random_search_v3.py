#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Search Hyperparameter Optimisation for DMM_SVVS_Variational_v3
======================================================================

Usage
-----
Call `random_search(...)` with your data and custom search ranges:

    results = random_search(
        X            = X,
        true_labels  = true_labels,
        n_trials     = 60,

        # ── Specify ranges for each hyperparameter ──────────────────
        K_max_range            = (5, 20),        # int range [low, high]
        py_discount_range      = (0.0, 0.9),     # float range [low, high)
        py_concentration_range = (0.1, 10.0),    # float range, log-uniform
        selection_prior_range  = (0.05, 0.95),   # float range [low, high)
        prune_threshold_range  = (1e-4, 0.1),    # float range, log-uniform

        master_seed  = 42,
        verbose      = True,
    )

Then refit with the best result found:

    best_model = refit_best(X, true_labels, results["best_result"])

Each range is a 2-tuple (low, high).  Defaults are provided for every
parameter so you only need to pass the ranges you want to override.
"""

import numpy as np
import json
import time as _time
import warnings
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from DMM_SVVS_Variational_v3 import DMM_SVVS_Variational_v3

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Default search ranges (overridable by the caller)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_K_MAX_RANGE            = (5, 20)       # integer, sampled uniformly
DEFAULT_PY_DISCOUNT_RANGE      = (0.0, 0.9)   # float, sampled uniformly
DEFAULT_PY_CONCENTRATION_RANGE = (0.1, 10.0)  # float, sampled log-uniformly
DEFAULT_SELECTION_PRIOR_RANGE  = (0.05, 0.95) # float, sampled uniformly
DEFAULT_PRUNE_THRESHOLD_RANGE  = (1e-4, 0.1)  # float, sampled log-uniformly

# Fixed parameters not included in the search
FIXED_PARAMS = dict(
    zeta         = 0.5,
    eta          = 0.5,
    xi_1         = 1.0,
    xi_2         = 1.0,
    tol          = 1e-4,
    max_iter     = 300,
    beta_start   = 0.2,
    beta_end     = 1.0,
    anneal_iters = 60,
    merge_every  = 15,
    n_restarts   = 3,
    min_clusters = None,
    verbose      = 0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_int(rng, low, high):
    """Sample an integer uniformly from [low, high] (both inclusive)."""
    return int(rng.integers(low, high + 1))


def _sample_uniform(rng, low, high):
    """Sample a float uniformly from [low, high)."""
    return float(rng.uniform(low, high))


def _sample_loguniform(rng, low, high):
    """Sample a float log-uniformly from [low, high)."""
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _sample_config(rng,
                   K_max_range,
                   py_discount_range,
                   py_concentration_range,
                   selection_prior_range,
                   prune_threshold_range):
    """
    Draw one random configuration from the given ranges.

    Parameters
    ----------
    rng                      : numpy.random.Generator
    K_max_range              : (int_low, int_high)    — inclusive
    py_discount_range        : (float_low, float_high)
    py_concentration_range   : (float_low, float_high)  — log-uniform
    selection_prior_range    : (float_low, float_high)
    prune_threshold_range    : (float_low, float_high)  — log-uniform

    Returns
    -------
    dict of sampled hyperparameter values
    """
    return {
        "K_max":            _sample_int(
                                rng,
                                int(K_max_range[0]),
                                int(K_max_range[1])),
        "py_discount":      _sample_uniform(
                                rng,
                                float(py_discount_range[0]),
                                float(py_discount_range[1])),
        "py_concentration": _sample_loguniform(
                                rng,
                                float(py_concentration_range[0]),
                                float(py_concentration_range[1])),
        "selection_prior":  _sample_uniform(
                                rng,
                                float(selection_prior_range[0]),
                                float(selection_prior_range[1])),
        "prune_threshold":  _sample_loguniform(
                                rng,
                                float(prune_threshold_range[0]),
                                float(prune_threshold_range[1])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────

def _run_trial(X, true_labels, config, trial_seed):
    """Fit one model configuration and return ARI, NMI, K, time."""
    params = {**FIXED_PARAMS, **config, "random_state": trial_seed}
    t0 = _time.time()
    try:
        model = DMM_SVVS_Variational_v3(**params)
        model.fit(X)
        pred  = model.predict(X)
        ari   = float(adjusted_rand_score(true_labels, pred))
        nmi   = float(normalized_mutual_info_score(true_labels, pred))
        K_est = int(model.K)
    except Exception as exc:
        ari, nmi, K_est = -1.0, -1.0, -1
    elapsed = _time.time() - t0
    return {"ari": ari, "nmi": nmi, "K_estimated": K_est,
            "elapsed": round(elapsed, 2), "config": config,
            "trial_seed": trial_seed}


# ─────────────────────────────────────────────────────────────────────────────
# Main random-search function
# ─────────────────────────────────────────────────────────────────────────────

def random_search(X,
                  true_labels,
                  n_trials                 = 50,
                  K_max_range              = DEFAULT_K_MAX_RANGE,
                  py_discount_range        = DEFAULT_PY_DISCOUNT_RANGE,
                  py_concentration_range   = DEFAULT_PY_CONCENTRATION_RANGE,
                  selection_prior_range    = DEFAULT_SELECTION_PRIOR_RANGE,
                  prune_threshold_range    = DEFAULT_PRUNE_THRESHOLD_RANGE,
                  master_seed              = 42,
                  verbose                  = True):
    """
    Random search over DMM_SVVS_Variational_v3 hyperparameters.

    Parameters
    ----------
    X                       : np.ndarray, shape (N, S)
        Count data matrix.
    true_labels             : np.ndarray, shape (N,)
        Ground-truth cluster labels for ARI evaluation.
    n_trials                : int
        Number of random configurations to evaluate.
    K_max_range             : tuple (int_low, int_high)
        Range for the maximum number of clusters (both ends inclusive).
        Example: (5, 20)
    py_discount_range       : tuple (float_low, float_high)
        Range for the Pitman-Yor discount d ∈ [0, 1).
        Sampled uniformly.  Example: (0.0, 0.8)
    py_concentration_range  : tuple (float_low, float_high)
        Range for the PY concentration θ.
        Sampled log-uniformly (so small values are well-explored).
        Example: (0.1, 10.0)
    selection_prior_range   : tuple (float_low, float_high)
        Range for the f warm-start value.
        Sampled uniformly.  Example: (0.1, 0.9)
    prune_threshold_range   : tuple (float_low, float_high)
        Range for the stick-weight deletion floor.
        Sampled log-uniformly.  Example: (1e-4, 0.05)
    master_seed             : int
        Seed for the search RNG — makes the run fully reproducible.
    verbose                 : bool
        Print a per-trial progress table if True.

    Returns
    -------
    dict with keys:
        best_config   : dict  — hyperparameters of the best trial
        best_result   : dict  — full result of the best trial
        all_results   : list  — all results sorted by ARI descending
    """
    X           = np.asarray(X, dtype=float)
    true_labels = np.asarray(true_labels)
    master_rng  = np.random.default_rng(master_seed)

    if verbose:
        print("\nDMM-SVVS-v3  Random Hyperparameter Search")
        print("=" * 72)
        print(f"  n_trials={n_trials},  master_seed={master_seed}")
        print(f"  K_max            : {K_max_range}")
        print(f"  py_discount      : {py_discount_range}")
        print(f"  py_concentration : {py_concentration_range}  [log-uniform]")
        print(f"  selection_prior  : {selection_prior_range}")
        print(f"  prune_threshold  : {prune_threshold_range}  [log-uniform]")
        print("=" * 72)
        print(f"  {'#':>4}  {'ARI':>6}  {'NMI':>6}  {'K_est':>5}  "
              f"{'sec':>5}  K_max  d      θ       sel    prune")
        print("  " + "-" * 70)

    all_results = []
    best_ari    = -np.inf
    best_result = None
    best_config = None

    for trial in range(1, n_trials + 1):
        config     = _sample_config(
                         master_rng,
                         K_max_range,
                         py_discount_range,
                         py_concentration_range,
                         selection_prior_range,
                         prune_threshold_range)
        trial_seed = int(master_rng.integers(0, 2**31))

        result = _run_trial(X, true_labels, config, 42)
        all_results.append(result)

        is_best = result["ari"] > best_ari
        if is_best:
            best_ari    = result["ari"]
            best_result = result
            best_config = config

        if verbose:
            c      = config
            marker = " ◀" if is_best else "  "
            print(f"  {trial:4d}  {result['ari']:6.3f}  {result['nmi']:6.3f}  "
                  f"{result['K_estimated']:5d}  {result['elapsed']:5.1f}  "
                  f"{c['K_max']:5d}  {c['py_discount']:.3f}  "
                  f"{c['py_concentration']:6.3f}  "
                  f"{c['selection_prior']:.3f}  "
                  f"{c['prune_threshold']:.4f}{marker}")

    all_results.sort(key=lambda r: r["ari"], reverse=True)

    if verbose:
        print("\n" + "=" * 72)
        print(f"  Best ARI   : {best_ari:.4f}")
        print(f"  Best config: K_max={best_config['K_max']}  "
              f"py_discount={best_config['py_discount']:.4f}  "
              f"py_concentration={best_config['py_concentration']:.4f}  "
              f"selection_prior={best_config['selection_prior']:.4f}  "
              f"prune_threshold={best_config['prune_threshold']:.6f}")

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
               n_restarts_final = 10,
               extra_seed       = 1,
               verbose          = True):
    """
    Re-fit the model using the best result from random_search,
    with more restarts for a robust final solution.

    The original trial seed that produced the best ARI is always included
    as one of the restart seeds, guaranteeing the search result is reproduced
    at minimum.  Additional restarts explore nearby initialisations.

    Parameters
    ----------
    X                : np.ndarray, shape (N, S)
    true_labels      : np.ndarray, shape (N,)
    best_result      : dict  — output of random_search(...)["best_result"]
                       Must contain keys "config" and "trial_seed".
    n_restarts_final : int   — total number of restarts for the final fit
                       (the original trial seed counts as one of these)
    extra_seed       : int   — seed used to generate the remaining restart seeds
    verbose          : bool

    Returns
    -------
    Fitted DMM_SVVS_Variational_v3 model
    """
    best_config  = best_result["config"]
    trial_seed   = best_result["trial_seed"]

    # Safety: clamp prune_threshold to a sensible ceiling so that
    # high values sampled during wide-range searches do not aggressively
    # delete all clusters during the refit.
    safe_config = dict(best_config)
    if safe_config.get("prune_threshold", 0.0) > 0.1:
        if verbose:
            print(f"  [refit_best] Clamping prune_threshold "
                  f"{safe_config['prune_threshold']:.4f} → 0.1")
        safe_config["prune_threshold"] = 0.1

    # Build the list of restart seeds: original trial seed first, then extras.
    rng_extra  = np.random.default_rng(extra_seed)
    n_extra    = max(n_restarts_final - 1, 0)
    extra_seeds = rng_extra.integers(0, 2**31, size=n_extra).tolist()
    all_seeds  = [trial_seed] + extra_seeds   # length == n_restarts_final

    if verbose:
        print(f"\nFinal refit  (n_restarts={n_restarts_final},  "
              f"trial_seed={trial_seed})")
        print("=" * 72)
        print(f"  Config: {json.dumps(safe_config, indent=4)}")
        print()

    # Run each seed as an independent single-restart fit; keep the best.
    best_model  = None
    best_ari    = -np.inf
    best_elbo   = -np.inf

    for restart_idx, seed in enumerate(all_seeds):
        params = {
            **FIXED_PARAMS,
            **safe_config,
            "n_restarts":   1,
            "verbose":      0,
            "random_state": 42,
        }
        try:
            m = DMM_SVVS_Variational_v3(**params)
            m.fit(X)
            pred_r = m.predict(X)
            ari_r  = float(adjusted_rand_score(true_labels, pred_r))
            nmi_r  = float(normalized_mutual_info_score(true_labels, pred_r))

            if verbose:
                tag = " ← original trial" if restart_idx == 0 else ""
                print(f"  restart {restart_idx+1:2d}/{n_restarts_final}"
                      f"  seed={seed:10d}  ARI={ari_r:.4f}"
                      f"  NMI={nmi_r:.4f}  K={m.K}{tag}")

            # Keep the restart with the highest ARI (tie-break: ELBO)
            if ari_r > best_ari or (ari_r == best_ari and m.best_elbo > best_elbo):
                best_ari   = ari_r
                best_elbo  = m.best_elbo
                best_model = m
        except Exception as exc:
            if verbose:
                print(f"  restart {restart_idx+1:2d}/{n_restarts_final}"
                      f"  seed={seed:10d}  FAILED: {exc}")

    pred = best_model.predict(X)
    ari  = adjusted_rand_score(true_labels, pred)
    nmi  = normalized_mutual_info_score(true_labels, pred)

    if verbose:
        print(f"\n  Best refit ARI    = {ari:.4f}")
        print(f"  Best refit NMI    = {nmi:.4f}")
        print(f"  K estimated       = {best_model.K}")
        sel = best_model.get_selected_features()
        print(f"  Selected features = {len(sel)} / {X.shape[1]}")

    return best_model


# ─────────────────────────────────────────────────────────────────────────────
# Print top-k summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_top_k(search_result, top_k=10):
    """Print a ranked table of the top-k configurations by ARI."""
    results = search_result["all_results"][:top_k]
    print(f"\nTop-{top_k} configurations:")
    print(f"  {'Rank':>4}  {'ARI':>6}  {'NMI':>6}  {'K_est':>5}  "
          f"K_max  d      θ       sel    prune")
    print("  " + "-" * 68)
    for rank, r in enumerate(results, 1):
        c = r["config"]
        print(f"  {rank:4d}  {r['ari']:6.3f}  {r['nmi']:6.3f}  "
              f"{r['K_estimated']:5d}  "
              f"{c['K_max']:5d}  {c['py_discount']:.3f}  "
              f"{c['py_concentration']:6.3f}  "
              f"{c['selection_prior']:.3f}  "
              f"{c['prune_threshold']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo — run when executed directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Generate synthetic block-structured DMM data ──────────────────────
    rng_data = np.random.default_rng(42)
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
    print(f"Data: N={N}, S={S}, K_true={K_true}  "
          f"class sizes={np.bincount(true_labels).tolist()}")

    # ── Run random search with custom ranges ─────────────────────────────
    results = random_search(
        X            = X,
        true_labels  = true_labels,
        n_trials     = 60,

        # ── User-specified search ranges ──────────────────────────────────
        K_max_range            = (5, 20),
        py_discount_range      = (0.0, 0.8),
        py_concentration_range = (0.1, 8.0),
        selection_prior_range  = (0.1, 0.9),
        prune_threshold_range  = (1e-4, 0.05),

        master_seed = 42,
        verbose     = True,
    )

    # ── Print top-10 summary ──────────────────────────────────────────────
    print_top_k(results, top_k=10)

    # ── Final refit with best result ──────────────────────────────────────
    best_model = refit_best(
        X                = X,
        true_labels      = true_labels,
        best_result      = results["best_result"],
        n_restarts_final = 10,
        extra_seed       = 42,
        verbose          = True,
    )
