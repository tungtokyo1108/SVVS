#!/usr/bin/env python3
"""
Clustering Difficulty Test Suite with Visualization

This script tests DMM_SVVS_Fixed on datasets with varying difficulty levels:
- Easy: Well-separated clusters
- Medium: Some overlap between clusters
- Hard: Significant overlap, unbalanced sizes
- Very Hard: High overlap, noise, similar clusters

Each test includes visualization of results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    confusion_matrix
)
from scipy.stats import entropy
from time import time
import warnings
warnings.filterwarnings('ignore')

from DMM_SVVS_robust import DMM_SVVS_Robust
from DMM_Dirichlet_SVVS import DMM_SVVS
from DMM_SVVS_Variational import DMM_SVVS_Variational
from DMM_SVVS_Variational_v2 import DMM_SVVS_Variational_v2
from DMM_SVVS_Variational_v3 import DMM_SVVS_Variational_v3

import random_search_dmm_svvs as rsds
import random_search_parallel_dmm_svvs as rrsds

# =============================================================================
# DATA GENERATION FUNCTIONS
# =============================================================================

def generate_well_separated_clusters(N=200, S=100, K=3, seed=42):
    """
    EASY: Well-separated clusters with distinct distributions

    Characteristics:
    - High concentration (low variance within clusters)
    - Clear separation between clusters
    - Balanced cluster sizes
    """
    np.random.seed(seed)

    X = np.zeros((N, S))
    true_labels = np.random.choice(K, size=N)

    # Create very distinct cluster distributions
    cluster_centers = []
    for k in range(K):
        # High concentration = well-defined clusters
        # Each cluster focuses on different features
        center = np.zeros(S)
        feature_subset = np.arange(k * (S // K), (k + 1) * (S // K))
        center[feature_subset] = np.random.gamma(2.0, 1.0, size=len(feature_subset))
        center = center / center.sum()
        cluster_centers.append(center)

    # Generate samples
    for i in range(N):
        k = true_labels[i]
        total_count = np.random.poisson(1000)
        X[i] = np.random.multinomial(total_count, cluster_centers[k])

    difficulty_info = {
        'level': 'Easy',
        'separation': 'High',
        'balance': 'Balanced',
        'noise': 'None'
    }

    return X, true_labels, difficulty_info


def generate_moderate_overlap_clusters(N=200, S=100, K=3, seed=42, separation=1.0):
    """
    MEDIUM: Moderate signal with realistic DMM-SVVS generative model.

    Simulation design following Dang et al. (2022) and Mao & Ma (2022):
    - 20% of OTUs are discriminating (signal); 80% are shared background.
    - Dirichlet-Multinomial generative model with NegBin library sizes.
    - Balanced cluster sizes (uniform mixing proportions).

    Parameters
    ----------
    separation : float in [0.0, 1.0]
        Controls the distance between clusters in feature space.

        Mechanism: alpha_signal is interpolated between alpha_cross (separation=0,
        no discrimination — all clusters share the same distribution) and
        alpha_signal_max (separation=1.0, maximum designed signal strength).

            alpha_signal = alpha_cross + (alpha_signal_max - alpha_cross) * separation

        Ratio  alpha_signal / alpha_cross  drives PCA/t-SNE separation:
          separation = 1.0  →  ratio = 3.0 / 0.15 = 20×  (far apart, easy)
          separation = 0.5  →  ratio = 1.575 / 0.15 = 10×  (moderate gap)
          separation = 0.2  →  ratio = 0.72 / 0.15 = 4.8×  (slight overlap)
          separation = 0.05 →  ratio ≈ 1.1×  (nearly touching clusters)
          separation = 0.0  →  ratio = 1×  (clusters identical, ARI ≈ 0)

        Recommended starting points:
          - Clusters "just touching"   → separation ≈ 0.1
          - Moderate visible overlap   → separation ≈ 0.2 – 0.3
          - Clear but not trivial gap  → separation ≈ 0.5
          - Well-separated (original)  → separation = 1.0

    Characteristics (at separation=1.0):
    - Library size    : NegBin(mean=8000, r=8)  — realistic 16S sequencing depth
    - S_signal        : 20% of S — each cluster owns one equal block
    - alpha_signal_max: 3.0   — maximum within-cluster concentration
    - alpha_cross     : 0.15  — cross-cluster bleed (fixed)
    - alpha_background: 0.10  — shared background OTUs
    - Zero inflation  : none
    """
    rng = np.random.default_rng(seed)
    separation = float(np.clip(separation, 0.0, 1.0))

    # ── Cluster assignments (balanced) ────────────────────────────────────
    pi = np.ones(K) / K
    true_labels = rng.choice(K, size=N, p=pi)

    # ── Alpha values governed by separation ───────────────────────────────
    # alpha_cross is fixed; alpha_signal shrinks toward alpha_cross as
    # separation → 0 (clusters become indistinguishable).
    ALPHA_SIGNAL_MAX = 3.0
    ALPHA_CROSS      = 0.15
    ALPHA_BG         = 0.10

    alpha_signal = ALPHA_CROSS + (ALPHA_SIGNAL_MAX - ALPHA_CROSS) * separation

    # ── Signal / background feature structure  (Dang 2022 block design) ──
    S_signal    = max(K, int(round(S * 0.20)))   # 20% discriminating OTUs
    block       = S_signal // K

    # alpha[k] ∈ R^S : Dirichlet concentration for cluster k
    alpha = np.full((K, S), ALPHA_BG)
    signal_mask = np.zeros(S, dtype=bool)

    for k in range(K):
        start = k * block
        end   = start + block if k < K - 1 else S_signal
        signal_mask[start:end] = True
        alpha[k, start:end] = alpha_signal           # own block
        for kp in range(K):
            if kp != k:
                s2 = kp * block
                e2 = s2 + block if kp < K - 1 else S_signal
                alpha[k, s2:e2] = ALPHA_CROSS        # other cluster's block: cross-bleed

    # ── Library sizes: NegBin(mean=8000, r=8)  — Mao & Ma (2022) ─────────
    lib_r = 8
    lib_mean = 8_000
    lib_p = lib_r / (lib_r + lib_mean)
    lib_sizes = rng.negative_binomial(n=lib_r, p=lib_p, size=N)
    lib_sizes = np.maximum(lib_sizes, 100)

    # ── Dirichlet-Multinomial sampling ────────────────────────────────────
    X = np.zeros((N, S), dtype=np.float64)
    for i in range(N):
        k = true_labels[i]
        p_i = rng.dirichlet(alpha[k])
        X[i] = rng.multinomial(int(lib_sizes[i]), p_i)

    sep_ratio = alpha_signal / ALPHA_CROSS
    difficulty_info = {
        'level':           'Medium',
        'separation':      f'Moderate (separation={separation:.2f}, ratio={sep_ratio:.1f}x)',
        'balance':         'Balanced',
        'noise':           'Low',
        'signal_mask':     signal_mask,
        'S_signal':        S_signal,
        'separation_param': separation,
        'alpha_signal':    round(alpha_signal, 4),
        'alpha_cross':     ALPHA_CROSS,
        'sparsity':        float((X == 0).mean()),
    }

    return X, true_labels, difficulty_info


def generate_high_overlap_clusters(N=200, S=100, K=4, seed=42,
                                    separation=1.0,
                                    imbalance=0.5,
                                    zero_inflation=0.15,
                                    signal_fraction=0.15):
    """
    HARD: Weak signal, unbalanced clusters, zero inflation.

    Simulation design following Dang et al. (2022) Dataset C spirit
    (IBD, ~10K OTUs, K=2-3) and Mao & Ma (2022) "weak signal" regime.

    Parameters
    ----------
    separation : float in [0.0, 1.0]
        Controls the distance between clusters in feature space.

        alpha_signal is interpolated between alpha_cross (no discrimination,
        separation=0) and alpha_signal_max (full signal, separation=1.0):

            alpha_signal = alpha_cross + (alpha_signal_max - alpha_cross) * separation

        Ratio  alpha_signal / alpha_cross  drives PCA/t-SNE separation:
          separation = 1.0  →  ratio ≈ 9×   (far apart — "original Hard")
          separation = 0.5  →  ratio ≈ 5×   (moderate overlap)
          separation = 0.2  →  ratio ≈ 2.6× (significant overlap)
          separation = 0.05 →  ratio ≈ 1.2× (clusters nearly touching)
          separation = 0.0  →  ratio = 1×   (clusters identical, ARI ≈ 0)

    imbalance : float in [0.0, 1.0]
        Controls how unequal cluster sizes are.

        Implemented via the Dirichlet concentration parameter c for mixing
        proportions:  π ~ Dirichlet(c · 1_K)

          imbalance = 0.0  →  c = 10   (nearly equal cluster sizes)
          imbalance = 0.5  →  c = 0.5  (moderately unbalanced, original default)
          imbalance = 1.0  →  c = 0.1  (extremely unbalanced — one cluster
                                         may dominate, others may be tiny)

        The mapping is:  c = 10^(1 − 2·imbalance)
          imbalance=0.0  → c=10,   imbalance=0.25 → c≈3.2,
          imbalance=0.5  → c=1.0,  imbalance=0.75 → c≈0.32,
          imbalance=1.0  → c=0.1

    zero_inflation : float in [0.0, 1.0)
        Fraction of entries set to structural zero after DMM sampling,
        following SparseDOSSA2 (Ma et al. 2021).

          zero_inflation = 0.00  →  no extra zeros (only DMM natural zeros)
          zero_inflation = 0.15  →  15% structural zeros (original default)
          zero_inflation = 0.30  →  30% (gut microbiome level)
          zero_inflation = 0.50  →  50% (vaginal/sparse communities)

    signal_fraction : float in (0.0, 1.0]
        Fraction of S OTUs that are discriminating (cluster-specific).

        Each cluster owns one equal block of signal OTUs:
          signal_fraction = 0.05  →  5% discriminating OTUs (very hard)
          signal_fraction = 0.15  →  15% (original default)
          signal_fraction = 0.30  →  30% (moderate)
          signal_fraction = 0.50  →  50% (easy to distinguish)

        Minimum is always K OTUs (at least 1 per cluster).

    Characteristics (at all defaults):
    - Library size    : NegBin(mean=8000, r=5)  — high variance in sequencing depth
    - alpha_signal_max: 1.8   — maximum within-cluster concentration
    - alpha_cross     : 0.20  — cross-cluster bleed (fixed)
    - alpha_background: 0.05  — shared background OTUs
    """
    rng = np.random.default_rng(seed)
    separation      = float(np.clip(separation,      0.0, 1.0))
    imbalance       = float(np.clip(imbalance,       0.0, 1.0))
    zero_inflation  = float(np.clip(zero_inflation,  0.0, 0.99))
    signal_fraction = float(np.clip(signal_fraction, 1e-4, 1.0))

    # ── Cluster assignments ───────────────────────────────────────────────
    # c = 10^(1 - 2*imbalance):  imbalance=0→c=10 (equal), imbalance=1→c=0.1 (extreme)
    conc = 10.0 ** (1.0 - 2.0 * imbalance)
    raw_pi = rng.dirichlet(np.ones(K) * conc)
    true_labels = rng.choice(K, size=N, p=raw_pi)
    # Guarantee every cluster has at least 1 sample
    for k in range(K):
        if (true_labels == k).sum() == 0:
            true_labels[rng.integers(N)] = k

    # ── Alpha values governed by separation ───────────────────────────────
    ALPHA_SIGNAL_MAX = 1.8
    ALPHA_CROSS      = 0.20
    ALPHA_BG         = 0.05

    alpha_signal = ALPHA_CROSS + (ALPHA_SIGNAL_MAX - ALPHA_CROSS) * separation

    # ── Signal / background feature structure ─────────────────────────────
    S_signal = max(K, int(round(S * signal_fraction)))
    block    = S_signal // K

    alpha = np.full((K, S), ALPHA_BG)
    signal_mask = np.zeros(S, dtype=bool)

    for k in range(K):
        start = k * block
        end   = start + block if k < K - 1 else S_signal
        signal_mask[start:end] = True
        alpha[k, start:end] = alpha_signal
        for kp in range(K):
            if kp != k:
                s2 = kp * block
                e2 = s2 + block if kp < K - 1 else S_signal
                alpha[k, s2:e2] = ALPHA_CROSS

    # ── Library sizes: NegBin(mean=8000, r=5) ────────────────────────────
    lib_r = 5
    lib_mean = 8_000
    lib_p = lib_r / (lib_r + lib_mean)
    lib_sizes = rng.negative_binomial(n=lib_r, p=lib_p, size=N)
    lib_sizes = np.maximum(lib_sizes, 100)

    # ── Dirichlet-Multinomial sampling ────────────────────────────────────
    X = np.zeros((N, S), dtype=np.float64)
    for i in range(N):
        k = true_labels[i]
        p_i = rng.dirichlet(alpha[k])
        X[i] = rng.multinomial(int(lib_sizes[i]), p_i)

    # ── Structural zero inflation  (SparseDOSSA2 / Ma et al. 2021) ────────
    if zero_inflation > 0.0:
        zi_mask = rng.uniform(size=(N, S)) < zero_inflation
        X[zi_mask] = 0.0

    sep_ratio    = alpha_signal / ALPHA_CROSS
    cluster_sizes = np.bincount(true_labels, minlength=K)
    difficulty_info = {
        'level':             'Hard',
        'separation':        f'Low (sep={separation:.2f}, ratio={sep_ratio:.1f}x)',
        'balance':           f'imbalance={imbalance:.2f} (conc={conc:.2f}, '
                             f'sizes={cluster_sizes.tolist()})',
        'noise':             f'ZI={zero_inflation:.0%}',
        'signal_mask':       signal_mask,
        'S_signal':          S_signal,
        'signal_fraction':   signal_fraction,
        'separation_param':  separation,
        'imbalance_param':   imbalance,
        'zero_inflation':    zero_inflation,
        'alpha_signal':      round(alpha_signal, 4),
        'alpha_cross':       ALPHA_CROSS,
        'sparsity':          float((X == 0).mean()),
    }

    return X, true_labels, difficulty_info


def generate_very_hard_clusters(N=200, S=100, K=5, seed=42, separation=1.0):
    """
    VERY HARD: Minimal signal, extreme imbalance, high zero inflation.

    Simulation design following Dang et al. (2022) Dataset D spirit
    (Obesity, ~56K OTUs, K=2, extreme sparsity) and SparseDOSSA2
    (Ma et al. 2021, gut data: 85% zeros):
    - Only 5% of OTUs are discriminating; 95% are shared background.
    - Extremely unbalanced clusters via Dirichlet(0.3).
    - High zero inflation (30%) simulating real gut/vaginal microbiome sparsity.
    - Highly variable library sizes: NegBin(mean=6000, r=3).

    Parameters
    ----------
    separation : float in [0.0, 1.0]
        Controls the distance between clusters in feature space.

        Mechanism: alpha_signal is interpolated between alpha_cross (separation=0,
        no discrimination) and alpha_signal_max (separation=1.0):

            alpha_signal = alpha_cross + (alpha_signal_max - alpha_cross) * separation

        Ratio  alpha_signal / alpha_cross  drives PCA/t-SNE separation:
          separation = 1.0  →  ratio = 1.3 / 0.35 = 3.7×  (already weak signal)
          separation = 0.5  →  ratio = 0.825 / 0.35 = 2.4× (further overlap)
          separation = 0.2  →  ratio = 0.54 / 0.35 = 1.5×  (near-random)
          separation = 0.0  →  ratio = 1×  (clusters identical, ARI ≈ 0)

        Recommended starting points:
          - Near-random baseline       → separation ≈ 0.1
          - Substantial overlap        → separation ≈ 0.3
          - Original Very Hard design  → separation = 1.0

    Characteristics (at separation=1.0):
    - Library size    : NegBin(mean=6000, r=3)  — very high depth variance
    - S_signal        : max(3K, 5% of S) — minimal discriminating OTUs
    - alpha_signal_max: 1.3   — barely elevated within-cluster concentration
    - alpha_cross     : 0.35  — high bleed (clusters nearly identical)
    - alpha_background: 0.05  — very sparse background
    - Zero inflation  : 30%
    """
    rng = np.random.default_rng(seed)
    separation = float(np.clip(separation, 0.0, 1.0))

    # ── Cluster assignments (extremely unbalanced) — Dirichlet(0.3) ──────
    raw_pi = rng.dirichlet(np.ones(K) * 0.3)
    true_labels = rng.choice(K, size=N, p=raw_pi)
    # Ensure every cluster has at least 1 sample
    for k in range(K):
        if (true_labels == k).sum() == 0:
            true_labels[rng.integers(N)] = k

    # ── Alpha values governed by separation ───────────────────────────────
    ALPHA_SIGNAL_MAX = 1.3
    ALPHA_CROSS      = 0.35
    ALPHA_BG         = 0.05

    alpha_signal = ALPHA_CROSS + (ALPHA_SIGNAL_MAX - ALPHA_CROSS) * separation

    # ── Signal / background feature structure ─────────────────────────────
    S_signal    = max(3 * K, int(round(S * 0.05)))  # 5% or at least 3 per cluster
    block       = S_signal // K

    alpha = np.full((K, S), ALPHA_BG)
    signal_mask = np.zeros(S, dtype=bool)

    for k in range(K):
        start = k * block
        end   = start + block if k < K - 1 else S_signal
        signal_mask[start:end] = True
        alpha[k, start:end] = alpha_signal
        for kp in range(K):
            if kp != k:
                s2 = kp * block
                e2 = s2 + block if kp < K - 1 else S_signal
                alpha[k, s2:e2] = ALPHA_CROSS

    # ── Library sizes: NegBin(mean=6000, r=3)  — very high variance ──────
    lib_r = 3
    lib_mean = 6_000
    lib_p = lib_r / (lib_r + lib_mean)
    lib_sizes = rng.negative_binomial(n=lib_r, p=lib_p, size=N)
    lib_sizes = np.maximum(lib_sizes, 100)

    # ── Dirichlet-Multinomial sampling ────────────────────────────────────
    X = np.zeros((N, S), dtype=np.float64)
    for i in range(N):
        k = true_labels[i]
        p_i = rng.dirichlet(alpha[k])
        X[i] = rng.multinomial(int(lib_sizes[i]), p_i)

    # ── Zero inflation: 30% structural zeros  (SparseDOSSA2 / gut data) ──
    zi_mask = rng.uniform(size=(N, S)) < 0.30
    X[zi_mask] = 0.0

    sep_ratio = alpha_signal / ALPHA_CROSS
    difficulty_info = {
        'level':            'Very Hard',
        'separation':       f'Minimal (separation={separation:.2f}, ratio={sep_ratio:.1f}x)',
        'balance':          'Extremely unbalanced',
        'noise':            'High',
        'signal_mask':      signal_mask,
        'S_signal':         S_signal,
        'separation_param': separation,
        'alpha_signal':     round(alpha_signal, 4),
        'alpha_cross':      ALPHA_CROSS,
        'sparsity':         float((X == 0).mean()),
    }

    return X, true_labels, difficulty_info


def generate_nested_clusters(N=200, S=100, K=3, seed=42):
    """
    SPECIAL: Nested/Hierarchical structure

    One large cluster with smaller sub-clusters inside
    Tests ability to handle hierarchical structure
    """
    np.random.seed(seed)

    # Create hierarchical structure - dynamically based on K
    # First cluster gets more weight, others split remaining
    cluster_probs = np.ones(K) * 0.5 / (K - 1) if K > 1 else np.ones(K)
    cluster_probs[0] = 0.5  # Main cluster gets 50%
    cluster_probs = cluster_probs / cluster_probs.sum()
    true_labels = np.random.choice(K, size=N, p=cluster_probs)

    X = np.zeros((N, S))

    # Main cluster: broad distribution
    alpha_main = np.random.gamma(1.0, 1.0, size=S)
    alpha_main = alpha_main / alpha_main.sum()

    # Create sub-clusters dynamically based on K
    alphas = [alpha_main]  # First cluster is main

    if K > 1:
        # Each sub-cluster emphasizes different feature subsets
        feature_subsets_per_cluster = S // K
        for k in range(1, K):
            alpha_sub = alpha_main.copy()
            # Each sub-cluster focuses on a different feature range
            start_idx = k * feature_subsets_per_cluster
            end_idx = min((k + 1) * feature_subsets_per_cluster, S)
            subset = np.arange(start_idx, end_idx)
            alpha_sub[subset] *= 2.0
            alpha_sub = alpha_sub / alpha_sub.sum()
            alphas.append(alpha_sub)

    for i in range(N):
        k = true_labels[i]
        total_count = np.random.poisson(1000)
        X[i] = np.random.multinomial(total_count, alphas[k])

    difficulty_info = {
        'level': 'Special (Nested)',
        'separation': 'Hierarchical',
        'balance': 'Unbalanced',
        'noise': 'Low'
    }

    return X, true_labels, difficulty_info


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_clustering_results(X, true_labels, pred_labels,
                                 metrics, model, save_path=None):
    """
    Create comprehensive visualization of clustering results

    Includes:
    - 2D projection (PCA/t-SNE) of true vs predicted clusters
    - Confusion matrix
    - Cluster size comparison
    - Metrics summary
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(f"Clustering Results ",
                 fontsize=16, fontweight='bold')

    # 1. PCA projection - True labels
    ax1 = fig.add_subplot(gs[0, 0])
    plot_2d_projection(X, true_labels, ax1, method='PCA', title='True Clusters (PCA)')

    # 2. PCA projection - Predicted labels
    ax2 = fig.add_subplot(gs[0, 1])
    plot_2d_projection(X, pred_labels, ax2, method='PCA', title='Predicted Clusters (PCA)')

    # 3. t-SNE projection - True labels
    ax3 = fig.add_subplot(gs[0, 2])
    plot_2d_projection(X, true_labels, ax3, method='t-SNE', title='True Clusters (t-SNE)')

    # 4. t-SNE projection - Predicted labels
    ax4 = fig.add_subplot(gs[1, 0])
    plot_2d_projection(X, pred_labels, ax4, method='t-SNE', title='Predicted Clusters (t-SNE)')

    # 5. Confusion matrix
    ax5 = fig.add_subplot(gs[1, 1])
    plot_confusion_matrix(true_labels, pred_labels, ax5)

    # 6. Cluster size comparison
    ax6 = fig.add_subplot(gs[1, 2])
    plot_cluster_sizes(true_labels, pred_labels, ax6)

    # 7. Metrics summary
    #ax7 = fig.add_subplot(gs[2, :])
    #plot_metrics_summary(difficulty_info, metrics, model, ax7)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")

    plt.close()

    return fig


def plot_2d_projection(X, labels, ax, method='PCA', title=''):
    """Plot 2D projection of high-dimensional data"""
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) // 4))

    X_2d = reducer.fit_transform(np.log1p(X))

    # Plot each cluster with different color
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                  c=[colors[i]], label=f'Cluster {label}',
                  alpha=0.6, s=30, edgecolors='k', linewidths=0.5)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


def plot_confusion_matrix(true_labels, pred_labels, ax):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_labels, pred_labels)

    # Normalize by row (true label)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto')

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.2f})',
                          ha="center", va="center", color="black", fontsize=9)

    ax.set_title('Confusion Matrix\n(True vs Predicted)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Set ticks
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_cluster_sizes(true_labels, pred_labels, ax):
    """Compare cluster sizes"""
    true_counts = np.bincount(true_labels)
    pred_counts = np.bincount(pred_labels)

    K_true = len(true_counts)
    K_pred = len(pred_counts)

    x = np.arange(max(K_true, K_pred))
    width = 0.35

    # Pad with zeros if needed
    if K_true < len(x):
        true_counts = np.concatenate([true_counts, np.zeros(len(x) - K_true)])
    if K_pred < len(x):
        pred_counts = np.concatenate([pred_counts, np.zeros(len(x) - K_pred)])

    ax.bar(x - width/2, true_counts[:len(x)], width, label='True', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, pred_counts[:len(x)], width, label='Predicted', alpha=0.8, color='coral')

    ax.set_title('Cluster Size Comparison', fontsize=11, fontweight='bold')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Samples')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x)


def plot_metrics_summary(difficulty_info, metrics, model, ax):
    """Display summary of metrics and difficulty characteristics"""
    ax.axis('off')

    # Create text summary
    summary_text = f"""
    DIFFICULTY CHARACTERISTICS:
    • Level: {difficulty_info['level']}
    • Separation: {difficulty_info['separation']}
    • Balance: {difficulty_info['balance']}
    • Noise: {difficulty_info['noise']}

    CLUSTERING PERFORMANCE:
    • K_true: {metrics['K_true']} → K_estimated: {metrics['K_estimated']}
    • Adjusted Rand Index (ARI): {metrics['ARI']:.3f}
    • Normalized Mutual Info (NMI): {metrics['NMI']:.3f}
    • Silhouette Score: {metrics['Silhouette']:.3f}

    COMPUTATIONAL:
    • Time: {metrics['Time']:.2f} seconds
    • Converged: {metrics['Converged']}
    • Iterations: {metrics['Iterations']}

    INTERPRETATION:
    • ARI = 1.0: Perfect clustering
    • ARI > 0.8: Excellent
    • ARI > 0.6: Good
    • ARI > 0.4: Fair
    • ARI < 0.4: Poor
    """

    # Color code based on ARI
    if metrics['ARI'] >= 0.8:
        color = 'green'
        verdict = "✓ EXCELLENT"
    elif metrics['ARI'] >= 0.6:
        color = 'blue'
        verdict = "✓ GOOD"
    elif metrics['ARI'] >= 0.4:
        color = 'orange'
        verdict = "⚠ FAIR"
    else:
        color = 'red'
        verdict = "✗ POOR"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax.text(0.75, 0.5, verdict, transform=ax.transAxes,
            fontsize=24, fontweight='bold', color=color,
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))



# =============================================================================
# MAIN - Can run line by line for debugging
# =============================================================================

if __name__ == "__main__":
    # You can run the full suite

    # OR run individual tests line by line:

    # # Easy test
    K=3
    
    X, true_labels, difficulty_info = generate_well_separated_clusters(N=400, S=5000, K=K)
    
    X, true_labels, difficulty_info = generate_moderate_overlap_clusters(N=400, S=5000, K=K, separation=0.1)
    
    X, true_labels, difficulty_info = generate_high_overlap_clusters(N=400, S=5000, K=K, 
                                                                     separation=0.2,
                                                                     imbalance=0.0,
                                                                     zero_inflation=0.80,
                                                                     signal_fraction=0.15)
    
    X, true_labels, difficulty_info = generate_very_hard_clusters(N=400, S=500, K=K, separation=1.0)
    
    model = DMM_SVVS_Robust(
        n_components=K+3,           # Overspecify
        weight_concentration_prior=10,  # 0.01-0.1=more clusters, 1-10=fewer
        prune_threshold=0.1,
        selection_prior=0.3,
        min_clusters=2,                # Never go below this
        burnin_iterations=100,         # No pruning before this
        max_iter=1000,
        verbose=1
    )
    
    model = DMM_SVVS_Variational(
        K_max=K+3,
        nu=10,
        zeta=1.0,
        eta=1.0,
        prune_threshold=0.02,
        selection_prior=0.3,
        max_iter=200,
        verbose=1,
        random_state=42
    )
    
    model = DMM_SVVS_Variational_v2(
        K_max=12,
        nu=1.343116,
        zeta=1.0,
        eta=1.0,
        prune_threshold=0.255117,
        selection_prior=0.3596,
        max_iter=200,
        verbose=1,
        random_state=42
    )
    
    model = DMM_SVVS_Variational_v3(
            K_max=20,
            py_discount=0.5,
            py_concentration=10,
            selection_prior=0.244140,
            prune_threshold=0.1255,
            zeta=0.5,
            eta=0.5,
            beta_start=0.2,
            anneal_iters=60,
            merge_every=15,
            n_restarts=5,
            max_iter=300,
            verbose=1,
            random_state=42
        )
    
    dmm = DMM_SVVS(n_components = 10, max_iter = 100, init_params = "random")
    log_resp_, clus_update, prob_selected, sel_update = dmm.fit_predict(X)
    
    start_time = time()
    model.fit(X)
    elapsed = time() - start_time

    # Predictions
    pred_labels = model.predict(X)

    # Compute metrics
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    try:
        silhouette = silhouette_score(np.log1p(X), pred_labels)
    except:
        silhouette = 0.0

    metrics = {
        'K_true': K,
        #'K_estimated': model.n_components,
        'ARI': ari,
        'NMI': nmi,
        'Silhouette': silhouette,
        'Time': elapsed,
        #'Converged': model.converged_,
        #'Iterations': model.n_iter_
    }
    
    print(f"\nResults:")
    #print(f"  K: {K} → {model.n_components}")
    print(f"  ARI: {ari:.3f}")
    print(f"  NMI: {nmi:.3f}")
    print(f"  Silhouette: {silhouette:.3f}")
    #print(f"  Time: {elapsed:.2f}s")
    #print(f"  Converged: {model.converged_}")

    # Visualize
    visualize_clustering_results(
            X, true_labels, pred_labels,
            difficulty_info, metrics, model,
            save_path=None
        )

###############################################################################
import pandas as pd
from sklearn.utils import check_array
import csv

dataset_A_count = pd.read_csv("datasetA_count.csv", index_col=0)
dataset_A_meta = pd.read_csv("datasetA_meta.csv", index_col=0)

dataset_A_count = pd.read_csv("CDI_count_data.csv", index_col=0)
dataset_A_meta = pd.read_csv("cdi_meta.csv", index_col=0)


dataset_A_count = pd.read_csv("Blueberry_ASVs_table.tsv", sep='\t', index_col=0).T
dataset_A_meta = pd.read_csv("Blueberry_metadata.tsv", sep='\t', index_col=0)


X = check_array(dataset_A_count, dtype=[np.float64, np.float32])
dmm = DMM_SVVS(n_components = 10, max_iter = 300, init_params = "random")
log_resp_, clus_update, prob_selected, sel_update = dmm.fit_predict(X)

resp_ = clus_update[300]
log_resp_max_ = resp_.argmax(axis=1)
df_cluster = {'Diseases': dataset_A_meta['Label'], 'Predicted_cluster': log_resp_max_}
clus_labeled = pd.DataFrame(data=df_cluster)
clus_labeled["True_cluster"] = clus_labeled["Diseases"].apply(lambda x: 2 
                                          if x == "D" else 4)
ARI_score = adjusted_rand_score(clus_labeled["Predicted_cluster"], clus_labeled["True_cluster"])


unique_labels = dataset_A_meta['DiseaseState'].unique()
label_map = {label: i for i, label in enumerate(unique_labels)}
print(f"Label mapping: {label_map}")
true_labels = dataset_A_meta['DiseaseState'].map(label_map).values

unique_labels = dataset_A_meta['comparison'].unique()
label_map = {label: i for i, label in enumerate(unique_labels)}
print(f"Label mapping: {label_map}")
true_labels = dataset_A_meta['comparison'].map(label_map).values

adjusted_rand_score(true_labels, log_resp_max_)

pred_labels = log_resp_max_

######

model = DMM_SVVS_Robust(
    n_components=12,           # Overspecify
    weight_concentration_prior=0.595603,  # 0.01-0.1=more clusters, 1-10=fewer
    prune_threshold=0.182132,
    selection_prior=0.4957,
    min_clusters=2,                # Never go below this
    burnin_iterations=10,         # No pruning before this
    max_iter=200,
    verbose=1
)

model = DMM_SVVS_Variational(
    K_max=8,
    nu=10,
    zeta=1.0,
    eta=1.0,
    prune_threshold=0.2,
    selection_prior=0.3,
    max_iter=200,
    verbose=1,
    random_state=42
)

model = DMM_SVVS_Variational_v2(
    K_max=17,
    nu=8.448216,
    zeta=1.0,
    eta=1.0,
    prune_threshold=0.381477,
    selection_prior=0.6759, 
    max_iter=200,
    verbose=1,
    random_state=42
)

from DMM_SVVS_Variational_v2_1 import DMM_SVVS_Variational_v2_1

model = DMM_SVVS_Variational_v2_1(
    K_max=5,
    nu=0.6373,
    zeta=1.0,
    eta=1.0,
    prune_threshold=0.125478,
    selection_prior=0.1125,
    max_iter=200,
    verbose=1,
    random_state=42
)

from DMM_SVVS_Variational_v2_3 import DMM_SVVS_Variational_v2_3

model = DMM_SVVS_Variational_v2_3(
    K_max=11,
    py_discount=0.194639,
    py_concentration=0.857911,
    selection_prior=0.231146,
    prune_threshold=0.465407,
    per_sample_f=False,
    max_iter=300,
    verbose=1,
    random_state=94067870
)

from DMM_SVVS_Variational_v2_4 import DMM_SVVS_Variational_v2_4

model = DMM_SVVS_Variational_v2_4(
    K_max=15,
    mfm_delta=2.3004,
    mfm_gamma=1.3267,
    selection_prior=0.200,
    per_sample_f=False,
    prune_threshold=0.87191,
    zeta=1.0,
    eta=1.0,
    xi_1=1.0,
    xi_2=1.0,
    tol=1e-4,
    max_iter=500
)

from DMM_SVVS_Variational_v2_5 import DMM_SVVS_Variational_v2_5

model = DMM_SVVS_Variational_v2_5(
    K_max=8,
    nig_sigma=0.866847,
    nig_alpha=3.036012,
    selection_prior=0.765325,
    per_sample_f=False,
    prune_threshold=0.120472,
    zeta=1.0,
    eta=1.0,
    xi_1=1.0,
    xi_2=1.0,
    tol=1e-4,
    max_iter=500,
    random_state=48970940
)


model = DMM_SVVS_Variational_v3(
        K_max=10,
        py_discount=0.560,
        py_concentration=0.340,
        selection_prior=0.875,
        prune_threshold=0.5535,
        zeta=0.5,
        eta=0.5,
        beta_start=0.2,
        anneal_iters=60,
        merge_every=15,
        n_restarts=1,
        max_iter=300,
        verbose=1,
        random_state=42
    )

from DMM_SVVS_Variational_v4 import DMM_SVVS_Variational_v4

model = DMM_SVVS_Variational_v4(
    K_max=19, mfm_delta=1.5445, mfm_gamma=1.1089,
    zeta=0.1, eta=0.1, xi_1=2.0, xi_2=1.0, selection_prior=0.924,
    beta_start=0.2, anneal_iters=60,
    merge_every=15, n_restarts=1, max_iter=300, verbose=1, random_state=42)

from DMM_SVVS_Variational_v5 import DMM_SVVS_Variational_v5

model = DMM_SVVS_Variational_v5(
        K_max=18,
        nig_sigma=0.7068, nig_alpha=2.7474,
        zeta=0.1, eta=0.1, xi_1=2.0, xi_2=1.0, selection_prior=0.4457,
        lambda_rep=0.2776,
        beta_start=0.2, anneal_iters=60,
        merge_every=15, n_restarts=1, max_iter=300, verbose=1, random_state=42)

from DMM_SVVS_Variational_v6 import DMM_SVVS_Variational_v6

model = DMM_SVVS_Variational_v6(
    K_max=11,
    nu=8.4683,
    zeta=1.0,
    eta=1.0,
    prune_threshold=0.65965,
    # ── SSL hyperparameters (replaces xi_1/xi_2/selection_prior) ──
    lambda0=277.41,
    lambda1=0.0148,
    kappa=0.388,
    init_xi=0.352,
    max_iter=200,
    verbose=1,
    random_state=1720056778
)

from DMM_SVVS_Variational_v6_1 import DMM_SVVS_Variational_v6_1

model = DMM_SVVS_Variational_v6_1(
    K_max=11,
    nu=1.686201,
    zeta=1.0,
    eta=1.0,
    prune_threshold=0.651822,
    # ── SSL hyperparameters (replaces xi_1/xi_2/selection_prior) ──
    lambda0=20.984864,
    lambda1=0.024534,
    kappa=0.338209,
    xi_ema=0.224204,
    max_iter=200,
    verbose=1,
    random_state=1021568550
)

###############################################################################

start_time = time()
model.fit(X)
elapsed = time() - start_time

# Predictions
pred_labels = model.predict(X)

true_labels = true_labels

# Compute metrics
ari = adjusted_rand_score(true_labels, pred_labels)
nmi = normalized_mutual_info_score(true_labels, pred_labels)

try:
    silhouette = silhouette_score(np.log1p(X), pred_labels)
except:
    silhouette = 0.0

metrics = {
    #'K_true': K,
    #'K_estimated': model.n_components,
    'ARI': ari,
    'NMI': nmi,
    'Silhouette': silhouette,
    'Time': elapsed,
    #'Converged': model.converged_,
    #'Iterations': model.n_iter_
}

print(f"\nResults:")
#print(f"  K: {K} → {model.n_components}")
print(f"  ARI: {ari:.3f}")
print(f"  NMI: {nmi:.3f}")
print(f"  Silhouette: {silhouette:.3f}")
#print(f"  Time: {elapsed:.2f}s")
#print(f"  Converged: {model.converged_}")

###############################################################################

result = rsds.random_search_dmm_svvs(
        X            = X,
        true_labels  = true_labels,
        n_trials     = 50,          # small for demo; use 30-50 in practice
        n_seeds      = 3,
        max_iter_search = 120,
        max_iter_final  = 400,
        K_lo   = 3,   K_hi   = 20,
        nu_lo  = 1.0, nu_hi  = 10,
        prune_lo = 0.10, prune_hi = 0.80,
        sel_lo = 0.10,   sel_hi   = 0.80,
        seed    = 42,
        verbose = 1,
    )

result = rrsds.parallel_random_search_dmm_svvs(
        X            = X,
        true_labels  = true_labels,
        n_trials     = 20,           # same as serial for fair comparison
        n_seeds      = 3,
        n_jobs       = -1,           # use all cores
        max_iter_search = 100,
        max_iter_final  = 400,
        K_lo   = 3,    K_hi   = 15,
        nu_lo  = 1.0, nu_hi  = 10,
        prune_lo = 0.10, prune_hi = 0.50,
        sel_lo = 0.10,   sel_hi   = 0.50,
        seed    = 42,
        verbose = 1,
    )

print(f"\n  Top-5 configurations:")
header = (f"  {'Rank':<5} {'Score':>8}  {'K_max':>6}  "
                f"{'nu':>10}  {'prune':>10}  {'sel_prior':>9}  "
                  f"{'K_est':>6}  {'Time':>6}")
print(header)
print("  " + "-" * (len(header) - 2))
for rank, t in enumerate(result['all_trials'][:5], 1):
    p = t['params']
    print(
        f"  {rank:<5} {t['mean_score']:>8.4f}  {p['K_max']:>6d}  "
        f"{p['nu']:>10.6f}  {p['prune_threshold']:>10.6f}  "
        f"{p['selection_prior']:>9.4f}  {t['mean_K']:>6.1f}  "
        f"{t['elapsed']:>5.1f}s")

import random_search_v2_1 as resv2

results = resv2.random_search(
        X            = X,
        true_labels  = true_labels,
        n_trials     = 50,

        # ── User-specified search ranges ──────────────────────────────────
        K_max_range           = (3, 15),
        nu_range              = (1.0, 10),
        zeta_range            = (1.0, 1.0),
        eta_range             = (1.0, 1.0),
        selection_prior_range = (0.1, 0.8),
        prune_threshold_range = (0.1, 0.8),

        master_seed = 42,
        verbose     = True,
    )

import random_search_parallel_robust as rspr 

result = rspr.parallel_random_search_robust(
        X            = X,
        true_labels  = true_labels,
        n_trials     = 20,
        n_seeds      = 3,
        n_jobs       = -1,
        max_iter_search = 200,
        max_iter_final  = 600,
        K_lo    = 3,    K_hi    = 15,
        wcp_lo  = 1.0, wcp_hi  = 10.0,
        prune_lo = 0.1, prune_hi = 0.80,
        sel_lo   = 0.1, sel_hi   = 0.80,
        seed    = 42,
        verbose = 1,
    )


import random_search_v3 as rsv3

results = rsv3.random_search(
        X            = X,
        true_labels  = true_labels,
        n_trials     = 100,

        # ── User-specified search ranges ──────────────────────────────────
        K_max_range            = (5, 20),
        py_discount_range      = (0.0, 0.8),
        py_concentration_range = (0.1, 10),
        selection_prior_range  = (0.1, 0.9),
        prune_threshold_range  = (0.1, 0.9),

        master_seed = 42,
        verbose     = True,
    )

rsv3.print_top_k(results, top_k=10)

best_model = rsv3.refit_best(
        X                = X,
        true_labels      = true_labels,
        best_result      = results["best_result"],
        n_restarts_final = 10,
        extra_seed       = 42,
        verbose          = True,
    )

import random_search_v4 as rsv4

results = rsv4.random_search(
        X            = X,
        true_labels  = true_labels,
        n_trials     = 30,

        # ── Specify ranges for each hyperparameter ──────────────────
        K_max_range           = (5, 20),       # int range [low, high]
        mfm_delta_range       = (0.1, 5.0),    # float range, log-uniform
        mfm_gamma_range       = (0.5, 10.0),   # float range, log-uniform
        selection_prior_range = (0.1, 0.95),  # float range [low, high)

        master_seed  = 42,
        verbose      = True,
    )

import random_search_v5 as rsv5

results = rsv5.random_search(
        X            = X,
        true_labels  = true_labels,
        n_trials     = 30,

        K_max_range           = (5, 20),
        nig_sigma_range       = (0.1, 0.9),
        nig_alpha_range       = (0.1, 10.0),
        selection_prior_range = (0.1, 0.9),
        lambda_rep_range      = (0.01, 2.0),

        master_seed = 42,
        verbose     = True,
    )

###############################################################################

import random_search_dmm_svvs_v2_3 as rsv2_3

results = rsv2_3.random_search(
    X           = X,
    true_labels = true_labels,
    n_trials    = 100,

    # Search ranges (override defaults as needed for your data)
    K_max_range            = (3, 20),
    py_discount_range      = (0.0, 1.0),
    py_concentration_range = (0.1, 10.0),
    selection_prior_range  = (0.1, 0.95),
    prune_threshold_range  = (0.1, 0.95),

    master_seed = 42,
    verbose     = True,
)

rsv2_3.print_top_k(results, top_k=10)


import random_search_dmm_svvs_v2_4 as rsv2_4

results = rsv2_4.random_search(
    X           = X,           # (N, S) count matrix
    true_labels = true_labels, # (N,)   ground-truth cluster labels
    n_trials    = 100,

    # Override any range you like; defaults are used for the rest
    K_max_range           = (3, 20),
    mfm_delta_range       = (1.0, 10.0),
    mfm_gamma_range       = (1.0, 10.0),
    selection_prior_range = (0.1, 0.95),
    prune_threshold_range = (0.1, 0.95),

    master_seed = 42,
    verbose     = True,
)

rsv2_4.print_top_k(results, top_k=10)


import random_search_dmm_svvs_v2_5 as rsv2_5

results = rsv2_5.random_search(
    X           = X,           # (N, S) count matrix
    true_labels = true_labels, # (N,)   ground-truth cluster labels
    n_trials    = 100,

    # Override any range you like; defaults are used for the rest
    K_max_range           = (3, 20),
    nig_sigma_range       = (0.1, 0.9),
    nig_alpha_range       = (1.0, 10.0),
    selection_prior_range = (0.1, 0.95),
    prune_threshold_range = (0.1, 0.95),

    master_seed = 42,
    verbose     = True,
)

rsv2_5.print_top_k(results, top_k=10)


import random_search_dmm_svvs_v6 as rsv6

results = rsv6.random_search(
    X           = X,           # (N, S) count matrix
    true_labels = true_labels, # (N,)   ground-truth cluster labels
    n_trials    = 100,

    # Override any range you like; defaults are used for the rest
    K_max_range           = (3, 15),
    nu_range              = (1.0, 10.0),
    lambda0_range         = (10.0, 500.0),
    lambda1_range         = (0.01, 10.0),
    kappa_range           = (0.01, 0.5),
    init_xi_range         = (0.01, 0.5),
    prune_threshold_range = (0.1, 0.95),

    master_seed = 42,
    verbose     = True,
)

rsv6.print_top_k(results, top_k=10)

best_model = rsv6.refit_best(
    X           = X,
    true_labels = true_labels,
    best_result = results["best_result"],
    n_restarts  = 5,
    max_iter    = 500,
    verbose     = True,
)

import random_search_dmm_svvs_v6_1 as rsv6_1

results = rsv6_1.random_search(
    X           = X,           # (N, S) count matrix
    true_labels = true_labels, # (N,)   ground-truth cluster labels
    n_trials    = 100,

    # Override any range you like; defaults are used for the rest
    K_max_range           = (3, 15),
    nu_range              = (1.0, 10.0),
    lambda0_range         = (10.0, 500.0),
    lambda1_range         = (0.01, 10.0),
    kappa_range           = (0.01, 0.5),
    init_xi_range         = (0.01, 0.5),
    prune_threshold_range = (0.1, 0.95),

    master_seed = 42,
    verbose     = True,
)

rsv6_1.print_top_k(results, top_k=10)

###############################################################################

visualize_clustering_results(
        X, true_labels, pred_labels,
         metrics, model,
        save_path=None
    )






































