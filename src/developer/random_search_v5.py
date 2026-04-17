#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Search Hyperparameter Optimisation for DMM_SVVS_Variational_v5
======================================================================

Searches over the five NIG-specific hyperparameters:

    K_max           — truncation level (int, uniform)
    nig_sigma       — NIG stability exponent σ ∈ (0,1) (float, uniform)
    nig_alpha       — NIG total-mass / concentration α > 0 (float, log-uniform)
    selection_prior — per-sample variable-selection warm-start f (float, uniform)
    lambda_rep      — soft DPP repulsion strength (float, log-uniform)

Usage
-----
    results = random_search(
        X            = X,
        true_labels  = true_labels,
        n_trials     = 60,

        K_max_range           = (5, 20),
        nig_sigma_range       = (0.1, 0.9),
        nig_alpha_range       = (0.1, 10.0),
        selection_prior_range = (0.05, 0.95),
        lambda_rep_range      = (0.01, 2.0),

        master_seed  = 42,
        verbose      = True,
    )

    best_model = refit_best(X, true_labels, results["best_result"])
    print_top_k(results, top_k=10)

Each range is a 2-tuple (low, high).  Defaults are provided for every
parameter so you only need to pass the ranges you want to override.
"""

import numpy as np
import json
import time as _time
import warnings
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from DMM_SVVS_Variational_v5 import DMM_SVVS_Variational_v5

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Default search ranges (overridable by the caller)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_K_MAX_RANGE           = (5, 20)      # integer, uniform
DEFAULT_NIG_SIGMA_RANGE       = (0.1, 0.9)   # float, uniform  (σ ∈ (0,1))
DEFAULT_NIG_ALPHA_RANGE       = (0.1, 10.0)  # float, log-uniform
DEFAULT_SELECTION_PRIOR_RANGE = (0.05, 0.95) # float, uniform
DEFAULT_LAMBDA_REP_RANGE      = (0.01, 2.0)  # float, log-uniform

# Fixed parameters not included in the search
FIXED_PARAMS = dict(
    zeta         = 0.1,
    eta          = 0.1,
    xi_1         = 2.0,
    xi_2         = 1.0,
    tol          = 1e-4,
    max_iter     = 300,
    beta_start   = 0.2,
    beta_end     = 1.0,
    anneal_iters = 60,
    merge_every  = 15,
    n_restarts   = 1,
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
                   nig_sigma_range,
                   nig_alpha_range,
                   selection_prior_range,
                   lambda_rep_range):
    """
    Draw one random configuration from the given ranges.

    Parameters
    ----------
    rng                   : numpy.random.Generator
    K_max_range           : (int_low, int_high)     — uniform integer
    nig_sigma_range       : (float_low, float_high) — uniform float in (0,1)
    nig_alpha_range       : (float_low, float_high) — log-uniform float > 0
    selection_prior_range : (float_low, float_high) — uniform float
    lambda_rep_range      : (float_low, float_high) — log-uniform float ≥ 0

    Returns
    -------
    dict of sampled hyperparameter values
    """
    return {
        "K_max":           _sample_int(
                               rng,
                               int(K_max_range[0]),
                               int(K_max_range[1])),
        "nig_sigma":       _sample_uniform(
                               rng,
                               float(nig_sigma_range[0]),
                               float(nig_sigma_range[1])),
        "nig_alpha":       _sample_loguniform(
                               rng,
                               float(nig_alpha_range[0]),
                               float(nig_alpha_range[1])),
        "selection_prior": _sample_uniform(
                               rng,
                               float(selection_prior_range[0]),
                               float(selection_prior_range[1])),
        "lambda_rep":      _sample_loguniform(
                               rng,
                               float(lambda_rep_range[0]),
                               float(lambda_rep_range[1])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────

def _run_trial(X, true_labels, config, trial_seed):
    """
    Fit one model configuration and return ARI, NMI, K, ICL proxy, time.

    Returns a result dict with keys:
        ari, nmi, K_estimated, icl, elapsed, config, trial_seed
    """
    params = {**FIXED_PARAMS, **config, "random_state": 42}
    t0 = _time.time()
    try:
        model = DMM_SVVS_Variational_v5(**params)
        model.fit(X)
        pred  = model.predict(X)
        ari   = float(adjusted_rand_score(true_labels, pred))
        nmi   = float(normalized_mutual_info_score(true_labels, pred))
        K_est = int(model.K)
        # ICL proxy: ELBO + entropy(r) — sharper assignments give higher ICL
        icl   = float(np.sum(model.r * np.log(np.maximum(model.r, 1e-10))))
    except Exception as exc:
        ari, nmi, K_est, icl = -1.0, -1.0, -1, -np.inf
    elapsed = _time.time() - t0
    return {
        "ari":         ari,
        "nmi":         nmi,
        "K_estimated": K_est,
        "icl":         icl,
        "elapsed":     round(elapsed, 2),
        "config":      config,
        "trial_seed":  trial_seed,
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
                  lambda_rep_range      = DEFAULT_LAMBDA_REP_RANGE,
                  master_seed           = 42,
                  verbose               = True):
    """
    Random search over DMM_SVVS_Variational_v5 hyperparameters.

    Parameters
    ----------
    X                     : np.ndarray, shape (N, S)
        Count data matrix.
    true_labels           : np.ndarray, shape (N,)
        Ground-truth cluster labels for ARI evaluation.
    n_trials              : int
        Number of random configurations to evaluate.
    K_max_range           : tuple (int_low, int_high)
        Range for the maximum number of clusters (both ends inclusive).
        Example: (5, 20)
    nig_sigma_range       : tuple (float_low, float_high)
        Range for the NIG stability exponent σ ∈ (0, 1).
        Sampled uniformly.  Example: (0.1, 0.9)
        • σ = 0.5 is the canonical NIG.
        • Smaller σ → fewer clusters (DP-like); larger → more clusters.
    nig_alpha_range       : tuple (float_low, float_high)
        Range for the NIG total-mass / concentration α > 0.
        Sampled log-uniformly (small values are well-explored).
        Example: (0.1, 10.0)
    selection_prior_range : tuple (float_low, float_high)
        Range for the per-sample variable-selection warm-start f.
        Sampled uniformly.  Example: (0.05, 0.95)
    lambda_rep_range      : tuple (float_low, float_high)
        Range for the soft DPP repulsion strength λ_rep ≥ 0.
        Sampled log-uniformly.  Example: (0.01, 2.0)
        • 0 = no repulsion; too large blocks valid merges.
    master_seed           : int
        Seed for the search RNG — makes the run fully reproducible.
    verbose               : bool
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
        print("\nDMM-SVVS-NIG v5  Random Hyperparameter Search")
        print("=" * 76)
        print(f"  n_trials={n_trials},  master_seed={master_seed}")
        print(f"  K_max           : {K_max_range}  [uniform int]")
        print(f"  nig_sigma       : {nig_sigma_range}  [uniform float, σ ∈ (0,1)]")
        print(f"  nig_alpha       : {nig_alpha_range}  [log-uniform float]")
        print(f"  selection_prior : {selection_prior_range}  [uniform float]")
        print(f"  lambda_rep      : {lambda_rep_range}  [log-uniform float]")
        print("=" * 76)
        print(f"  {'#':>4}  {'ARI':>6}  {'NMI':>6}  {'K':>4}  {'sec':>5}  "
              f"{'K_max':>5}  {'sigma':>6}  {'alpha':>7}  {'sel':>5}  {'lrep':>6}")
        print("  " + "-" * 70)

    all_results = []
    best_ari    = -np.inf
    best_result = None
    best_config = None

    for trial in range(1, n_trials + 1):
        trial_seed = int(master_rng.integers(0, 2**31))
        config     = _sample_config(
                         master_rng,
                         K_max_range,
                         nig_sigma_range,
                         nig_alpha_range,
                         selection_prior_range,
                         lambda_rep_range)

        result  = _run_trial(X, true_labels, config, 42)
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
                  f"{result['K_estimated']:4d}  {result['elapsed']:5.1f}  "
                  f"{c['K_max']:5d}  {c['nig_sigma']:6.3f}  "
                  f"{c['nig_alpha']:7.4f}  {c['selection_prior']:5.3f}  "
                  f"{c['lambda_rep']:6.4f}{marker}")

    all_results.sort(key=lambda r: r["ari"], reverse=True)

    if verbose:
        print("\n" + "=" * 76)
        print(f"  Best ARI   : {best_ari:.4f}")
        if best_config is not None:
            print(f"  Best config:")
            print(f"    K_max           = {best_config['K_max']}")
            print(f"    nig_sigma       = {best_config['nig_sigma']:.4f}")
            print(f"    nig_alpha       = {best_config['nig_alpha']:.4f}")
            print(f"    selection_prior = {best_config['selection_prior']:.4f}")
            print(f"    lambda_rep      = {best_config['lambda_rep']:.4f}")

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
    extra_seed       : int   — seed used to generate the remaining restart seeds
    verbose          : bool

    Returns
    -------
    Fitted DMM_SVVS_Variational_v5 model
    """
    best_config = best_result["config"]
    trial_seed  = best_result["trial_seed"]

    # Build restart seeds: original trial seed first, then extras
    rng_extra   = np.random.default_rng(extra_seed)
    n_extra     = max(n_restarts_final - 1, 0)
    extra_seeds = rng_extra.integers(0, 2**31, size=n_extra).tolist()
    all_seeds   = [trial_seed] + extra_seeds   # length == n_restarts_final

    if verbose:
        print(f"\nFinal refit  (n_restarts={n_restarts_final},  "
              f"trial_seed={trial_seed})")
        print("=" * 76)
        print(f"  Config: {json.dumps(best_config, indent=4)}")
        print()

    best_model = None
    best_ari   = -np.inf
    best_icl   = -np.inf

    for restart_idx, seed in enumerate(all_seeds):
        params = {
            **FIXED_PARAMS,
            **best_config,
            "n_restarts":   1,
            "verbose":      0,
            "random_state": seed,
        }
        try:
            m = DMM_SVVS_Variational_v5(**params)
            m.fit(X)
            pred_r = m.predict(X)
            ari_r  = float(adjusted_rand_score(true_labels, pred_r))
            nmi_r  = float(normalized_mutual_info_score(true_labels, pred_r))
            icl_r  = float(np.sum(m.r * np.log(np.maximum(m.r, 1e-10))))

            if verbose:
                tag = " ← original trial" if restart_idx == 0 else ""
                print(f"  restart {restart_idx+1:2d}/{n_restarts_final}"
                      f"  seed={seed:10d}  ARI={ari_r:.4f}"
                      f"  NMI={nmi_r:.4f}  K={m.K}{tag}")

            if ari_r > best_ari or (ari_r == best_ari and icl_r > best_icl):
                best_ari   = ari_r
                best_icl   = icl_r
                best_model = m
        except Exception as exc:
            if verbose:
                print(f"  restart {restart_idx+1:2d}/{n_restarts_final}"
                      f"  seed={seed:10d}  FAILED: {exc}")

    if best_model is None:
        raise RuntimeError("All refit restarts failed.")

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
    print(f"  {'Rank':>4}  {'ARI':>6}  {'NMI':>6}  {'K':>4}  "
          f"{'K_max':>5}  {'sigma':>6}  {'alpha':>7}  {'sel':>5}  {'lrep':>6}")
    print("  " + "-" * 68)
    for rank, r in enumerate(results, 1):
        c = r["config"]
        print(f"  {rank:4d}  {r['ari']:6.3f}  {r['nmi']:6.3f}  "
              f"{r['K_estimated']:4d}  "
              f"{c['K_max']:5d}  {c['nig_sigma']:6.3f}  "
              f"{c['nig_alpha']:7.4f}  {c['selection_prior']:5.3f}  "
              f"{c['lambda_rep']:6.4f}")


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

        K_max_range           = (5, 20),
        nig_sigma_range       = (0.1, 0.9),
        nig_alpha_range       = (0.1, 10.0),
        selection_prior_range = (0.1, 0.9),
        lambda_rep_range      = (0.01, 2.0),

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
