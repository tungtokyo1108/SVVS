#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_search_parallel_dmm_svvs.py
===================================
Parallel random search over four key hyperparameters of DMM_SVVS_Variational_v2
to maximise ARI (Adjusted Rand Index).

Parallelisation strategy
------------------------
Two performance problems are solved compared to a naive ProcessPoolExecutor:

  Problem 1 — Data transfer overhead (main cause of slowness)
    With N=336, S=3346, the data matrix X is ~9 MB.  A naive
    ProcessPoolExecutor pickles X and sends it to every worker for every
    job.  With 60 jobs that is ~540 MB of IPC traffic.
    Fix: use multiprocessing.Pool with an initializer that loads X into
    each worker process once.  Jobs only transmit the small params dict
    and seed integer (~200 bytes each).

  Problem 2 — No live per-trial output
    Collecting all futures before printing means nothing appears until
    the entire search finishes.
    Fix: track per-trial completion inside the result-collection loop.
    The moment a trial's last seed result arrives, aggregate and print
    that trial's line immediately with the running-best marker.

Speedup (expected on 10-core machine, N=336, S=3346)
  Serial  : n_trials × n_seeds × t_per_fit
  Parallel: ≈ (n_trials × n_seeds / n_workers) × t_per_fit  +  small overhead

Usage
-----
  from random_search_parallel_dmm_svvs import parallel_random_search_dmm_svvs

  result = parallel_random_search_dmm_svvs(
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
# Per-worker global state
# Loaded once per worker process via the pool initializer — never re-pickled.
# ─────────────────────────────────────────────────────────────────────────────

_WORKER_X:           np.ndarray           = None
_WORKER_TRUE_LABELS: Optional[np.ndarray] = None


def _worker_init(X_shared: np.ndarray, true_labels_shared: Optional[np.ndarray]) -> None:
    """
    Pool initializer: called once when each worker process starts.
    Stores X and true_labels as process-global variables so individual
    job functions do not need to receive them as arguments (no re-pickling).
    """
    global _WORKER_X, _WORKER_TRUE_LABELS
    import warnings
    warnings.filterwarnings("ignore")
    _WORKER_X           = X_shared
    _WORKER_TRUE_LABELS = true_labels_shared


# ─────────────────────────────────────────────────────────────────────────────
# Worker job functions  (top-level for pickling; X is NOT an argument)
# ─────────────────────────────────────────────────────────────────────────────

def _job_fit(args: Tuple) -> Dict[str, Any]:
    """
    Fit one model using the process-global X and true_labels.
    args = (params_dict, seed, max_iter, trial_idx, seed_idx)
    """
    params, seed, max_iter, trial_idx, seed_idx = args

    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from sklearn.metrics import adjusted_rand_score
    from DMM_SVVS_Variational_v2 import DMM_SVVS_Variational_v2

    X           = _WORKER_X
    true_labels = _WORKER_TRUE_LABELS

    def _quality_proxy(model):
        EPS = 1e-10
        r   = model.r
        confidence = float(r.max(axis=1).mean())
        K_eff      = max(model.K, 2)
        H          = -(r * np.log(r + EPS)).sum(axis=1).mean()
        sharpness  = float(1.0 - H / np.log(K_eff))
        sizes      = r.sum(axis=0) / (r.sum() + EPS)
        balance    = float(np.clip(1.0 - np.std(sizes) * np.sqrt(model.K), 0.0, 1.0))
        return (confidence + sharpness + balance) / 3.0

    try:
        model = DMM_SVVS_Variational_v2(
            K_max           = params['K_max'],
            nu              = params['nu'],
            prune_threshold = params['prune_threshold'],
            selection_prior = params['selection_prior'],
            max_iter        = max_iter,
            verbose         = 0,
            random_state    = seed,
        )
        model.fit(X)

        if true_labels is not None:
            pred  = model.predict(X)
            score = float(adjusted_rand_score(true_labels, pred))
        else:
            score = _quality_proxy(model)

        return dict(trial_idx=trial_idx, seed_idx=seed_idx,
                    score=score, K_est=model.K, error=None)

    except Exception as e:
        return dict(trial_idx=trial_idx, seed_idx=seed_idx,
                    score=-1.0, K_est=params['K_max'], error=str(e))


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
    from DMM_SVVS_Variational_v2 import DMM_SVVS_Variational_v2

    X           = _WORKER_X
    true_labels = _WORKER_TRUE_LABELS

    def _quality_proxy(model):
        EPS = 1e-10
        r   = model.r
        confidence = float(r.max(axis=1).mean())
        K_eff      = max(model.K, 2)
        H          = -(r * np.log(r + EPS)).sum(axis=1).mean()
        sharpness  = float(1.0 - H / np.log(K_eff))
        sizes      = r.sum(axis=0) / (r.sum() + EPS)
        balance    = float(np.clip(1.0 - np.std(sizes) * np.sqrt(model.K), 0.0, 1.0))
        return (confidence + sharpness + balance) / 3.0

    try:
        model = DMM_SVVS_Variational_v2(
            K_max           = params['K_max'],
            nu              = params['nu'],
            prune_threshold = params['prune_threshold'],
            selection_prior = params['selection_prior'],
            max_iter        = max_iter,
            verbose         = 0,
            random_state    = seed,
        )
        model.fit(X)

        if true_labels is not None:
            pred  = model.predict(X)
            score = float(adjusted_rand_score(true_labels, pred))
        else:
            score = _quality_proxy(model)

        return dict(score=score, model=model, K_est=model.K, error=None)

    except Exception as e:
        return dict(score=-1.0, model=None, K_est=params['K_max'], error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_params(
    rng:      np.random.Generator,
    K_lo:     int,   K_hi:     int,
    nu_lo:    float, nu_hi:    float,
    prune_lo: float, prune_hi: float,
    sel_lo:   float, sel_hi:   float,
) -> Dict[str, Any]:
    """
    Draw one random hyperparameter configuration.
      K_max           : uniform integer in [K_lo, K_hi]
      nu              : log-uniform in [nu_lo, nu_hi]
      prune_threshold : log-uniform in [prune_lo, prune_hi]
      selection_prior : uniform in [sel_lo, sel_hi]
    """
    K_max = int(rng.integers(K_lo, K_hi + 1))
    nu    = float(np.exp(rng.uniform(np.log(nu_lo),    np.log(nu_hi))))
    prune = float(np.exp(rng.uniform(np.log(prune_lo), np.log(prune_hi))))
    sel   = float(rng.uniform(sel_lo, sel_hi))
    return dict(K_max=K_max, nu=nu, prune_threshold=prune, selection_prior=sel)


def _resolve_n_jobs(n_jobs: int) -> int:
    n_cpu = os.cpu_count() or 1
    if n_jobs <= 0:
        return n_cpu
    return max(1, min(n_jobs, n_cpu))


# ─────────────────────────────────────────────────────────────────────────────
# Main parallel random search
# ─────────────────────────────────────────────────────────────────────────────

def parallel_random_search_dmm_svvs(
    X:           np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    # ── search budget ──────────────────────────────────────────────────────
    n_trials:    int = 30,
    n_seeds:     int = 3,
    # ── parallelism ────────────────────────────────────────────────────────
    n_jobs:      int = -1,
    # ── training budget ────────────────────────────────────────────────────
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
    Parallel random search over K_max, nu, prune_threshold, selection_prior.

    Parameters
    ----------
    X               : (N, S) count data matrix
    true_labels     : (N,) ground-truth labels or None (unsupervised proxy)
    n_trials        : number of random hyperparameter configurations to try
    n_seeds         : seeds averaged per configuration for noise reduction
    n_jobs          : worker processes  (-1 = all cores)
    max_iter_search : CAVI iterations during search  (keep short, e.g. 100-150)
    max_iter_final  : CAVI iterations for the final best-model refit
    K_lo/K_hi       : integer bounds for K_max (inclusive)
    nu_lo/nu_hi     : float bounds for nu  (log-uniform)
    prune_lo/prune_hi : float bounds for prune_threshold  (log-uniform)
    sel_lo/sel_hi   : float bounds for selection_prior  (uniform)
    seed            : master random seed
    verbose         : 0 = silent, 1 = live per-trial lines + summary

    Returns
    -------
    dict:
      best_params   : winning hyperparameter dict
      best_score    : mean ARI (or proxy) of the best search-phase config
      best_model    : DMM_SVVS_Variational_v2 refitted at max_iter_final
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
        print("=" * 68)
        print("  DMM-SVVS Parallel Random Search")
        print(f"  Mode        : {mode_str}")
        print(f"  Data        : N={X.shape[0]}, S={X.shape[1]}")
        print(f"  Trials      : {n_trials}  |  Seeds/trial : {n_seeds}")
        print(f"  Total jobs  : {n_total}  |  Workers : {n_workers}")
        print(f"  max_iter    : {max_iter_search} (search) / {max_iter_final} (final)")
        print(f"  K_max       : [{K_lo}, {K_hi}]")
        print(f"  nu          : [{nu_lo:.0e}, {nu_hi}]  (log-uniform)")
        print(f"  prune       : [{prune_lo:.0e}, {prune_hi}]  (log-uniform)")
        print(f"  sel_prior   : [{sel_lo}, {sel_hi}]  (uniform)")
        print(f"  Est. speedup: ~{n_workers}×  over serial")
        print("=" * 68)

    # ── Pre-generate all (params, seed) pairs deterministically ──────────
    # Done before any worker starts so the sampling sequence is reproducible.
    rng = np.random.default_rng(seed)

    trial_params: List[Dict[str, Any]] = []   # one entry per trial
    job_args:     List[Tuple]          = []   # one entry per (trial, seed) job

    for trial_idx in range(n_trials):
        params = _sample_params(
            rng, K_lo, K_hi, nu_lo, nu_hi, prune_lo, prune_hi, sel_lo, sel_hi
        )
        trial_params.append(params)
        for seed_idx in range(n_seeds):
            worker_seed = (seed + trial_idx * 9973 + seed_idx * 997) % (2**31 - 1)
            job_args.append((params, worker_seed, max_iter_search, trial_idx, seed_idx))

    # ── Collect raw results into raw_results[trial_idx] ──────────────────
    # per_trial_received[k] counts how many seed-results we have for trial k.
    raw_results:         List[List[Dict]] = [[] for _ in range(n_trials)]
    per_trial_received:  List[int]        = [0]  * n_trials
    per_trial_scores:    List[List[float]] = [[] for _ in range(n_trials)]
    per_trial_kests:     List[List[float]] = [[] for _ in range(n_trials)]

    # Running-best state (updated as trials complete)
    best_score  = -np.inf
    best_params = None

    trial_records: List[Dict] = []
    score_history: List[float] = [None] * n_trials   # filled as trials complete

    if verbose >= 1:
        print(f"\n  Dispatching {n_total} jobs to {n_workers} workers ...")
        print(f"  (each trial line prints as soon as all its seeds finish)\n")
        # Table header
        hdr = (f"  {'Trial':<9} {'Score':>8}  {'K_max':>6}  {'nu':>9}  "
               f"{'prune':>9}  {'sel':>7}  {'K_est':>6}  {'vs Best'}")
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

        # Submit all jobs as async tasks; keep handles in job_handle list
        handles = [pool.apply_async(_job_fit, (arg,)) for arg in job_args]

        # Poll for completed results; print a trial line as soon as its
        # last seed result has arrived.
        completed_flags = [False] * n_total
        n_collected     = 0

        while n_collected < n_total:
            for job_idx, handle in enumerate(handles):
                if completed_flags[job_idx]:
                    continue
                if not handle.ready():
                    continue

                # This job just finished — collect it
                completed_flags[job_idx] = True
                n_collected += 1

                try:
                    res = handle.get(timeout=0)
                except Exception as e:
                    trial_idx = job_args[job_idx][3]
                    seed_idx  = job_args[job_idx][4]
                    res = dict(trial_idx=trial_idx, seed_idx=seed_idx,
                               score=-1.0, K_est=job_args[job_idx][0]['K_max'],
                               error=str(e))

                t_idx = res['trial_idx']
                raw_results[t_idx].append(res)
                per_trial_received[t_idx] += 1
                per_trial_scores[t_idx].append(res['score'])
                per_trial_kests[t_idx].append(res['K_est'])

                # If all seeds for this trial are in, aggregate and print
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
                            f"{p['K_max']:>6d}  "
                            f"{p['nu']:>9.5f}  "
                            f"{p['prune_threshold']:>9.5f}  "
                            f"{p['selection_prior']:>7.4f}  "
                            f"{mean_K:>6.1f}  "
                            f"{delta_str}"
                        )
                        sys.stdout.flush()

            # Brief sleep to avoid busy-waiting burning a CPU core
            if n_collected < n_total:
                import time as _time
                _time.sleep(0.05)

    search_time = time() - t_search_start

    # Fill any score_history slots that stayed None (should not happen)
    score_history = [s if s is not None else -1.0 for s in score_history]

    # Sort trial records best-first for the summary table
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
                           K_est=best_params['K_max'], error=str(e))

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
        print("=" * 68)
        print("  PARALLEL RANDOM SEARCH — FINAL SUMMARY")
        print("=" * 68)
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
            print(f"  Final K estimated  : {best_model.K}")
        print(f"  Best params:")
        for k, v in best_params.items():
            print(f"    {k:<20s} = {v}")
        print("=" * 68)

        # Top-5 table
        print(f"\n  Top-5 configurations:")
        hdr2 = (f"  {'Rank':<5} {'Score':>8}  {'K_max':>6}  "
                f"{'nu':>10}  {'prune':>10}  {'sel_prior':>9}  {'K_est':>6}")
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        for rank, t in enumerate(trial_records[:5], 1):
            p = t['params']
            print(
                f"  {rank:<5} {t['mean_score']:>8.4f}  {p['K_max']:>6d}  "
                f"{p['nu']:>10.6f}  {p['prune_threshold']:>10.6f}  "
                f"{p['selection_prior']:>9.4f}  {t['mean_K']:>6.1f}"
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
    from DMM_SVVS_Variational_v2 import DMM_SVVS_Variational_v2

    print("Parallel Random Search Demo — DMM_SVVS_Variational_v2")
    print("=" * 68)

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

    # Baseline
    t0 = time()
    baseline = DMM_SVVS_Variational_v2(
        K_max=10, nu=0.3, prune_threshold=0.2, selection_prior=0.1,
        max_iter=400, verbose=0, random_state=42
    )
    baseline.fit(X)
    ari_base = adjusted_rand_score(true_labels, baseline.predict(X))
    nmi_base = normalized_mutual_info_score(true_labels, baseline.predict(X))
    print(f"Baseline: ARI={ari_base:.4f}  NMI={nmi_base:.4f}  "
          f"K={baseline.K}  [{time()-t0:.1f}s]\n")

    # Parallel random search
    result = parallel_random_search_dmm_svvs(
        X=X, true_labels=true_labels,
        n_trials=20, n_seeds=3, n_jobs=-1,
        max_iter_search=120, max_iter_final=400,
        K_lo=3, K_hi=12,
        nu_lo=1e-3, nu_hi=1.0,
        prune_lo=5e-3, prune_hi=0.30,
        sel_lo=0.05, sel_hi=0.80,
        seed=42, verbose=1,
    )

    best_model = result['best_model']
    if best_model is not None:
        ari_opt = adjusted_rand_score(true_labels, best_model.predict(X))
        nmi_opt = normalized_mutual_info_score(true_labels, best_model.predict(X))
        print(f"\nComparison:")
        print(f"  {'':28s}  {'ARI':>8}  {'NMI':>8}  {'K':>4}")
        print(f"  {'Baseline (defaults)':28s}  {ari_base:>8.4f}  "
              f"{nmi_base:>8.4f}  {baseline.K:>4d}")
        print(f"  {'Parallel search best':28s}  {ari_opt:>8.4f}  "
              f"{nmi_opt:>8.4f}  {best_model.K:>4d}")
        print(f"\n  ARI improvement: {ari_opt - ari_base:+.4f}")
