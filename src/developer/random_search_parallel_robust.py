#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_search_parallel_robust.py
=================================
Parallel random search over four key hyperparameters of DMM_SVVS_Robust
to maximise ARI (Adjusted Rand Index).

Optimised parameters
--------------------
  n_components               : int   – initial (maximum) number of clusters
  weight_concentration_prior : float – DP concentration
                                       SMALLER → more clusters survive pruning
                                       LARGER  → fewer clusters survive pruning
  prune_threshold            : float – minimum stick-breaking weight to keep a cluster
  selection_prior            : float – prior probability that a feature is discriminating

Sampling strategy
-----------------
  n_components               : uniform integer in [K_lo, K_hi]
  weight_concentration_prior : log-uniform in [wcp_lo, wcp_hi]
                               (spans orders of magnitude; small values matter)
  prune_threshold            : log-uniform in [prune_lo, prune_hi]
  selection_prior            : uniform in [sel_lo, sel_hi]

Parallelisation
---------------
  Uses multiprocessing.Pool with an initializer that loads X into each
  worker process once — avoids re-pickling the (potentially large) data
  matrix for every job.  Only the small params dict and seed are sent
  per job (~200 bytes vs ~MB for X).

Live output
-----------
  Results are printed in trial order (1..n_trials) as soon as each trial's
  last seed result arrives, with a running "vs Best" column so you can
  see ARI improvement at a glance.

Usage
-----
  from random_search_parallel_robust import parallel_random_search_robust

  result = parallel_random_search_robust(
      X, true_labels,
      n_trials=20, n_seeds=3, n_jobs=-1
  )
  pred = result['best_model'].predict(X)
  print(result['best_params'])
"""

import os
import sys
import warnings
import numpy as np
from time import time
import multiprocessing
from typing import Optional, List, Dict, Any, Tuple

from sklearn.metrics import adjusted_rand_score
from sklearn.utils import check_array

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Per-worker global state — loaded once per process via pool initializer
# ─────────────────────────────────────────────────────────────────────────────

_WORKER_X:           np.ndarray           = None
_WORKER_TRUE_LABELS: Optional[np.ndarray] = None


def _worker_init(X_shared: np.ndarray,
                 true_labels_shared: Optional[np.ndarray]) -> None:
    """
    Pool initializer: store X and true_labels as process-globals.
    Called once per worker process — never per job.
    """
    global _WORKER_X, _WORKER_TRUE_LABELS
    import warnings
    warnings.filterwarnings("ignore")
    _WORKER_X           = X_shared
    _WORKER_TRUE_LABELS = true_labels_shared


# ─────────────────────────────────────────────────────────────────────────────
# Worker job functions  (top-level for pickling; X not in args)
# ─────────────────────────────────────────────────────────────────────────────

def _job_fit(args: Tuple) -> Dict[str, Any]:
    """
    Fit one DMM_SVVS_Robust model and return its score.

    args = (params_dict, seed, max_iter, trial_idx, seed_idx)

    The data matrix X and true_labels are read from process-global variables
    set by _worker_init — they are not re-pickled per job.
    """
    params, seed, max_iter, trial_idx, seed_idx = args

    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    from DMM_SVVS_robust import DMM_SVVS_Robust

    X           = _WORKER_X
    true_labels = _WORKER_TRUE_LABELS

    def _quality_proxy(model) -> float:
        """
        Unsupervised quality score when true_labels is None.
        Averages three signals:
          1. Confidence  — mean peak responsibility per sample
          2. Sharpness   — 1 minus normalised entropy of responsibilities
          3. Balance     — penalises heavily skewed cluster sizes
        """
        EPS = 1e-10
        # DMM_SVVS_Robust stores responsibilities in resp_ after fit
        r   = model.resp_
        confidence = float(r.max(axis=1).mean())
        K_eff      = max(model.n_components, 2)
        H          = -(r * np.log(r + EPS)).sum(axis=1).mean()
        sharpness  = float(1.0 - H / np.log(K_eff))
        sizes      = r.sum(axis=0) / (r.sum() + EPS)
        balance    = float(np.clip(
            1.0 - np.std(sizes) * np.sqrt(model.n_components), 0.0, 1.0
        ))
        return (confidence + sharpness + balance) / 3.0

    try:
        model = DMM_SVVS_Robust(
            n_components               = params['n_components'],
            weight_concentration_prior = params['weight_concentration_prior'],
            prune_threshold            = params['prune_threshold'],
            selection_prior            = params['selection_prior'],
            max_iter                   = max_iter,
            verbose                    = 0,
            random_state               = seed,
        )
        model.fit(X)

        if true_labels is not None:
            pred  = model.predict(X)
            score = float(adjusted_rand_score(true_labels, pred))
        else:
            score = _quality_proxy(model)

        return dict(
            trial_idx = trial_idx,
            seed_idx  = seed_idx,
            score     = score,
            K_est     = model.n_components,   # post-pruning K
            error     = None,
        )

    except Exception as e:
        return dict(
            trial_idx = trial_idx,
            seed_idx  = seed_idx,
            score     = -1.0,
            K_est     = params['n_components'],
            error     = str(e),
        )


def _job_refit(args: Tuple) -> Dict[str, Any]:
    """
    Refit at full max_iter and return the fitted model object.
    args = (params_dict, seed, max_iter)
    """
    params, seed, max_iter = args

    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    from DMM_SVVS_robust import DMM_SVVS_Robust

    X           = _WORKER_X
    true_labels = _WORKER_TRUE_LABELS

    def _quality_proxy(model) -> float:
        EPS = 1e-10
        r   = model.resp_
        confidence = float(r.max(axis=1).mean())
        K_eff      = max(model.n_components, 2)
        H          = -(r * np.log(r + EPS)).sum(axis=1).mean()
        sharpness  = float(1.0 - H / np.log(K_eff))
        sizes      = r.sum(axis=0) / (r.sum() + EPS)
        balance    = float(np.clip(
            1.0 - np.std(sizes) * np.sqrt(model.n_components), 0.0, 1.0
        ))
        return (confidence + sharpness + balance) / 3.0

    try:
        model = DMM_SVVS_Robust(
            n_components               = params['n_components'],
            weight_concentration_prior = params['weight_concentration_prior'],
            prune_threshold            = params['prune_threshold'],
            selection_prior            = params['selection_prior'],
            max_iter                   = max_iter,
            verbose                    = 0,
            random_state               = seed,
        )
        model.fit(X)

        if true_labels is not None:
            pred  = model.predict(X)
            score = float(adjusted_rand_score(true_labels, pred))
        else:
            score = _quality_proxy(model)

        return dict(score=score, model=model, K_est=model.n_components, error=None)

    except Exception as e:
        return dict(score=-1.0, model=None,
                    K_est=params['n_components'], error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_params(
    rng:       np.random.Generator,
    K_lo:      int,   K_hi:      int,
    wcp_lo:    float, wcp_hi:    float,
    prune_lo:  float, prune_hi:  float,
    sel_lo:    float, sel_hi:    float,
) -> Dict[str, Any]:
    """
    Draw one random hyperparameter configuration.

    n_components               : uniform integer in [K_lo, K_hi]
    weight_concentration_prior : log-uniform in [wcp_lo, wcp_hi]
                                 (Note: SMALLER = more clusters survive)
    prune_threshold            : log-uniform in [prune_lo, prune_hi]
    selection_prior            : uniform in [sel_lo, sel_hi]
    """
    n_comp = int(rng.integers(K_lo, K_hi + 1))

    # Log-uniform: gives equal probability to each order of magnitude.
    # Critical for weight_concentration_prior which can range 0.01–10.
    wcp   = float(np.exp(rng.uniform(np.log(wcp_lo),   np.log(wcp_hi))))
    prune = float(np.exp(rng.uniform(np.log(prune_lo), np.log(prune_hi))))
    sel   = float(rng.uniform(sel_lo, sel_hi))

    return dict(
        n_components               = n_comp,
        weight_concentration_prior = wcp,
        prune_threshold            = prune,
        selection_prior            = sel,
    )


def _resolve_n_jobs(n_jobs: int) -> int:
    n_cpu = os.cpu_count() or 1
    if n_jobs <= 0:
        return n_cpu
    return max(1, min(n_jobs, n_cpu))


# ─────────────────────────────────────────────────────────────────────────────
# Main parallel random search
# ─────────────────────────────────────────────────────────────────────────────

def parallel_random_search_robust(
    X:           np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    # ── search budget ──────────────────────────────────────────────────────
    n_trials:    int = 30,
    n_seeds:     int = 3,
    # ── parallelism ────────────────────────────────────────────────────────
    n_jobs:      int = -1,
    # ── training budget ────────────────────────────────────────────────────
    max_iter_search: int = 300,
    max_iter_final:  int = 1000,
    # ── search bounds ──────────────────────────────────────────────────────
    K_lo:      int   = 3,
    K_hi:      int   = 20,
    wcp_lo:    float = 0.01,   # weight_concentration_prior lower bound
    wcp_hi:    float = 10.0,   # weight_concentration_prior upper bound
    prune_lo:  float = 1e-3,
    prune_hi:  float = 0.20,
    sel_lo:    float = 0.05,
    sel_hi:    float = 0.95,
    # ── misc ───────────────────────────────────────────────────────────────
    seed:    int = 42,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Parallel random search over n_components, weight_concentration_prior,
    prune_threshold, and selection_prior for DMM_SVVS_Robust.

    Parameters
    ----------
    X               : (N, S) count data matrix
    true_labels     : (N,) ground-truth labels or None (unsupervised proxy)
    n_trials        : number of random hyperparameter configurations to try
    n_seeds         : seeds averaged per configuration (noise reduction)
    n_jobs          : worker processes  (-1 = all cores)
    max_iter_search : iterations during search phase (keep short, e.g. 200-400)
    max_iter_final  : iterations for the final best-model refit (e.g. 1000)
    K_lo / K_hi     : integer bounds for n_components (inclusive)
    wcp_lo / wcp_hi : float bounds for weight_concentration_prior (log-uniform)
                      Note: SMALLER values → more clusters survive pruning
    prune_lo/prune_hi : float bounds for prune_threshold (log-uniform)
    sel_lo / sel_hi : float bounds for selection_prior (uniform)
    seed            : master random seed (deterministic results)
    verbose         : 0 = silent, 1 = live per-trial lines + summary

    Returns
    -------
    dict:
      best_params   : winning hyperparameter dict
      best_score    : mean ARI (or proxy) of the best search-phase config
      best_model    : DMM_SVVS_Robust refitted at max_iter_final
      all_trials    : list of trial dicts sorted best-first
      score_history : mean_score per trial in original sampling order
      n_jobs_used   : actual worker count
    """
    X = check_array(X, dtype=np.float64)
    n_workers = _resolve_n_jobs(n_jobs)
    n_total   = n_trials * n_seeds
    mode_str  = "supervised (ARI)" if true_labels is not None \
                else "unsupervised (quality proxy)"

    if verbose >= 1:
        print()
        print("=" * 70)
        print("  DMM-SVVS-Robust Parallel Random Search")
        print(f"  Mode        : {mode_str}")
        print(f"  Data        : N={X.shape[0]}, S={X.shape[1]}")
        print(f"  Trials      : {n_trials}  |  Seeds/trial : {n_seeds}")
        print(f"  Total jobs  : {n_total}  |  Workers : {n_workers}")
        print(f"  max_iter    : {max_iter_search} (search) / {max_iter_final} (final)")
        print(f"  n_components    : [{K_lo}, {K_hi}]  (integer, uniform)")
        print(f"  wcp             : [{wcp_lo:.0e}, {wcp_hi}]  (log-uniform; smaller=more clusters)")
        print(f"  prune_threshold : [{prune_lo:.0e}, {prune_hi}]  (log-uniform)")
        print(f"  selection_prior : [{sel_lo}, {sel_hi}]  (uniform)")
        print(f"  Est. speedup    : ~{n_workers}×  over serial")
        print("=" * 70)

    # ── Pre-generate all (params, seed) pairs deterministically ──────────
    rng = np.random.default_rng(seed)

    trial_params: List[Dict[str, Any]] = []
    job_args:     List[Tuple]          = []

    for trial_idx in range(n_trials):
        params = _sample_params(
            rng,
            K_lo, K_hi,
            wcp_lo, wcp_hi,
            prune_lo, prune_hi,
            sel_lo, sel_hi,
        )
        trial_params.append(params)
        for seed_idx in range(n_seeds):
            worker_seed = (seed + trial_idx * 9973 + seed_idx * 997) % (2**31 - 1)
            job_args.append((params, worker_seed, max_iter_search, trial_idx, seed_idx))

    # ── Per-trial accumulators ────────────────────────────────────────────
    raw_results:         List[List[Dict]]  = [[] for _ in range(n_trials)]
    per_trial_received:  List[int]         = [0]  * n_trials
    per_trial_scores:    List[List[float]] = [[] for _ in range(n_trials)]
    per_trial_kests:     List[List[float]] = [[] for _ in range(n_trials)]

    best_score  = -np.inf
    best_params = None

    trial_records: List[Dict]  = []
    score_history: List[float] = [None] * n_trials

    if verbose >= 1:
        print(f"\n  Dispatching {n_total} jobs to {n_workers} workers ...")
        print(f"  (each trial line prints as soon as all its seeds finish)\n")
        hdr = (f"  {'Trial':<9} {'Score':>8}  {'K_init':>7}  {'wcp':>9}  "
               f"{'prune':>9}  {'sel':>7}  {'K_est':>6}  vs Best")
        print(hdr)
        print(f"  {'-' * (len(hdr) - 2)}")
        sys.stdout.flush()

    t_search_start = time()

    # ── Worker pool: X loaded once per process via initializer ───────────
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        processes   = n_workers,
        initializer = _worker_init,
        initargs    = (X, true_labels),
    ) as pool:

        # Submit all jobs
        handles = [pool.apply_async(_job_fit, (arg,)) for arg in job_args]

        # Poll for completions; print a trial line as its last seed arrives
        completed_flags = [False] * n_total
        n_collected     = 0

        while n_collected < n_total:
            for job_idx, handle in enumerate(handles):
                if completed_flags[job_idx]:
                    continue
                if not handle.ready():
                    continue

                completed_flags[job_idx] = True
                n_collected += 1

                try:
                    res = handle.get(timeout=0)
                except Exception as e:
                    t_idx    = job_args[job_idx][3]
                    s_idx    = job_args[job_idx][4]
                    res = dict(trial_idx=t_idx, seed_idx=s_idx,
                               score=-1.0,
                               K_est=job_args[job_idx][0]['n_components'],
                               error=str(e))

                t_idx = res['trial_idx']
                raw_results[t_idx].append(res)
                per_trial_received[t_idx] += 1
                per_trial_scores[t_idx].append(res['score'])
                per_trial_kests[t_idx].append(res['K_est'])

                # All seeds for this trial have arrived — aggregate and print
                if per_trial_received[t_idx] == n_seeds:
                    mean_score = float(np.mean(per_trial_scores[t_idx]))
                    mean_K     = float(np.mean(per_trial_kests[t_idx]))
                    params     = trial_params[t_idx]

                    score_history[t_idx] = mean_score
                    record = dict(
                        params      = params,
                        mean_score  = mean_score,
                        seed_scores = list(per_trial_scores[t_idx]),
                        mean_K      = mean_K,
                    )
                    trial_records.append(record)

                    prev_best   = best_score
                    is_new_best = mean_score > best_score
                    if is_new_best:
                        best_score  = mean_score
                        best_params = dict(params)

                    if verbose >= 1:
                        delta_str = (
                            "  *** NEW BEST ***" if is_new_best
                            else f"  {mean_score - prev_best:+.4f}"
                        )
                        p = params
                        print(
                            f"  [{t_idx+1:3d}/{n_trials}]  "
                            f"{mean_score:>8.4f}  "
                            f"{p['n_components']:>7d}  "
                            f"{p['weight_concentration_prior']:>9.5f}  "
                            f"{p['prune_threshold']:>9.5f}  "
                            f"{p['selection_prior']:>7.4f}  "
                            f"{mean_K:>6.1f}  "
                            f"{delta_str}"
                        )
                        sys.stdout.flush()

            if n_collected < n_total:
                import time as _time
                _time.sleep(0.05)

    search_time = time() - t_search_start

    # Fill any gaps (should not occur)
    score_history = [s if s is not None else -1.0 for s in score_history]

    # Sort best-first for the summary / return value
    trial_records.sort(key=lambda t: t['mean_score'], reverse=True)

    if verbose >= 1:
        print(f"\n  Search complete in {search_time:.1f}s  "
              f"({n_total} fits across {n_workers} workers)")
        print(f"  Best search score : {best_score:.4f}")
        print(f"  Best params       : {best_params}")

    # ── Refit best config at full max_iter in parallel ────────────────────
    n_refit_seeds = max(3, n_seeds)
    refit_seeds   = [(seed + i * 1009) % (2**31 - 1) for i in range(n_refit_seeds)]
    refit_args    = [(best_params, rs, max_iter_final) for rs in refit_seeds]

    if verbose >= 1:
        print(f"\n  Refitting best config: {n_refit_seeds} seeds "
              f"at max_iter={max_iter_final} in parallel ...")
        sys.stdout.flush()

    best_model       = None
    best_model_score = -np.inf

    with ctx.Pool(
        processes   = min(n_workers, n_refit_seeds),
        initializer = _worker_init,
        initargs    = (X, true_labels),
    ) as pool:
        refit_handles = [pool.apply_async(_job_refit, (arg,)) for arg in refit_args]
        for i, handle in enumerate(refit_handles):
            try:
                res = handle.get()
            except Exception as e:
                res = dict(score=-1.0, model=None,
                           K_est=best_params['n_components'], error=str(e))

            if verbose >= 1:
                err_str = f"  ERROR: {res.get('error')}" if res.get('error') else ""
                print(f"    refit seed {i}: score={res['score']:.4f}, "
                      f"K={res['K_est']}{err_str}")
                sys.stdout.flush()

            if res['score'] > best_model_score and res['model'] is not None:
                best_model_score = res['score']
                best_model       = res['model']

    # ── Final summary ─────────────────────────────────────────────────────
    if verbose >= 1:
        print()
        print("=" * 70)
        print("  DMM-SVVS-ROBUST — PARALLEL RANDOM SEARCH SUMMARY")
        print("=" * 70)
        print(f"  Trials evaluated   : {n_trials}  ({n_seeds} seeds each)")
        print(f"  Workers used       : {n_workers}")
        print(f"  Wall-clock time    : {search_time:.1f}s")
        print(f"  Est. serial time   : {search_time * n_workers:.1f}s")
        print(f"  Effective speedup  : ~{n_workers:.0f}×")
        print(f"  Best search score  : {best_score:.4f}")
        if best_model is not None:
            if true_labels is not None:
                pred  = best_model.predict(X)
                final = float(adjusted_rand_score(true_labels, pred))
                print(f"  Final ARI (full iter): {final:.4f}")
            print(f"  Final K estimated    : {best_model.n_components}")
        print(f"  Best params:")
        for k, v in best_params.items():
            print(f"    {k:<30s} = {v}")
        print("=" * 70)

        # Top-5 table
        print(f"\n  Top-5 configurations:")
        hdr2 = (f"  {'Rank':<5} {'Score':>8}  {'K_init':>7}  "
                f"{'wcp':>10}  {'prune':>10}  {'sel_prior':>9}  {'K_est':>6}")
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        for rank, t in enumerate(trial_records[:5], 1):
            p = t['params']
            print(
                f"  {rank:<5} {t['mean_score']:>8.4f}  "
                f"{p['n_components']:>7d}  "
                f"{p['weight_concentration_prior']:>10.6f}  "
                f"{p['prune_threshold']:>10.6f}  "
                f"{p['selection_prior']:>9.4f}  "
                f"{t['mean_K']:>6.1f}"
            )

    return dict(
        best_params   = best_params,
        best_score    = best_score,
        best_model    = best_model,
        all_trials    = trial_records,
        score_history = score_history,
        n_jobs_used   = n_workers,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Demo / smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from DMM_SVVS_robust import DMM_SVVS_Robust

    print("Parallel Random Search Demo — DMM_SVVS_Robust")
    print("=" * 70)

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

    print(f"Data: N={N}, S={S}, K_true={K_true}\n")

    # ── Baseline: default parameters ──────────────────────────────────────
    print("Baseline (default params):")
    t0 = time()
    baseline = DMM_SVVS_Robust(
        n_components=10,
        weight_concentration_prior=1.0,
        prune_threshold=0.01,
        selection_prior=0.3,
        max_iter=500,
        verbose=0,
        random_state=42,
    )
    baseline.fit(X)
    ari_base = adjusted_rand_score(true_labels, baseline.predict(X))
    nmi_base = normalized_mutual_info_score(true_labels, baseline.predict(X))
    print(f"  ARI={ari_base:.4f}  NMI={nmi_base:.4f}  "
          f"K={baseline.n_components}  [{time()-t0:.1f}s]\n")

    # ── Parallel random search ────────────────────────────────────────────
    result = parallel_random_search_robust(
        X            = X,
        true_labels  = true_labels,
        n_trials     = 20,
        n_seeds      = 3,
        n_jobs       = -1,
        max_iter_search = 200,
        max_iter_final  = 600,
        K_lo    = 3,    K_hi    = 15,
        wcp_lo  = 0.01, wcp_hi  = 5.0,
        prune_lo = 1e-3, prune_hi = 0.15,
        sel_lo   = 0.05, sel_hi   = 0.80,
        seed    = 42,
        verbose = 1,
    )

    # ── Final comparison ──────────────────────────────────────────────────
    best_model = result['best_model']
    if best_model is not None:
        ari_opt = adjusted_rand_score(true_labels, best_model.predict(X))
        nmi_opt = normalized_mutual_info_score(true_labels, best_model.predict(X))

        print(f"\nComparison:")
        print(f"  {'':30s}  {'ARI':>8}  {'NMI':>8}  {'K':>4}")
        print(f"  {'Baseline (defaults)':30s}  {ari_base:>8.4f}  "
              f"{nmi_base:>8.4f}  {baseline.n_components:>4d}")
        print(f"  {'Parallel search best':30s}  {ari_opt:>8.4f}  "
              f"{nmi_opt:>8.4f}  {best_model.n_components:>4d}")
        print(f"\n  ARI improvement: {ari_opt - ari_base:+.4f}")
        print(f"  Best params found:")
        for k, v in result['best_params'].items():
            print(f"    {k:<30s} = {v}")
