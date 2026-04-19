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

###############################################################################
import pandas as pd
from sklearn.utils import check_array
import csv

dataset_A_count = pd.read_csv("datasetA_count.csv", index_col=0)
dataset_A_meta = pd.read_csv("datasetA_meta.csv", index_col=0)

dataset_A_count = pd.read_csv("CDI_count_data.csv", index_col=0)
dataset_A_meta = pd.read_csv("cdi_meta.csv", index_col=0)


dataset_A_count = pd.read_csv("ob_goodrich_ASVs_table.tsv", sep='\t', index_col=0).T
dataset_A_meta = pd.read_csv("ob_goodrich_metadata.tsv", sep='\t', index_col=0)
# Align metadata row order to match count matrix sample order

#--------------------------------------------------------------------------------
min_samples = 200
idx = (dataset_A_count).sum(axis=0) >= min_samples
dataset_A_count = dataset_A_count.loc[:, idx]

dataset_A_meta = dataset_A_meta.loc[dataset_A_count.index]
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
    K_max=11,
    nu=4.660183,
    zeta=1.0,
    eta=1.0,
    prune_threshold=0.266344,
    selection_prior=0.4957, 
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
    K_max=15,
    py_discount=0.975622,
    py_concentration=3.328736,
    selection_prior=0.208897,
    prune_threshold=0.275647,
    per_sample_f=True,
    max_iter=300,
    verbose=1,
    random_state=1688060241
)

from DMM_SVVS_Variational_v2_4 import DMM_SVVS_Variational_v2_4

model = DMM_SVVS_Variational_v2_4(
    K_max=7,
    mfm_delta=1.390023,
    mfm_gamma=5.980919,
    selection_prior=0.268474,
    per_sample_f=True,
    prune_threshold=0.776877,
    zeta=1.0,
    eta=1.0,
    xi_1=1.0,
    xi_2=1.0,
    tol=1e-4,
    max_iter=500,
    random_state=517174316
)

from DMM_SVVS_Variational_v2_5 import DMM_SVVS_Variational_v2_5

model = DMM_SVVS_Variational_v2_5(
    K_max=7,
    nig_sigma=0.214418,
    nig_alpha=5.980919,
    selection_prior=0.268474,
    per_sample_f=True,
    prune_threshold=0.776877,
    zeta=1.0,
    eta=1.0,
    xi_1=1.0,
    xi_2=1.0,
    tol=1e-4,
    max_iter=500,
    random_state=517174316
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
    K_max=7, mfm_delta=1.390023, mfm_gamma=5.980919,
    zeta=0.1, eta=0.1, xi_1=2.0, xi_2=1.0, selection_prior=0.268474,
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
    K_max=6,
    nu=6.703350,
    zeta=1.0,
    eta=1.0,
    prune_threshold=0.348092,
    # ── SSL hyperparameters (replaces xi_1/xi_2/selection_prior) ──
    lambda0=76.666704,
    lambda1=0.207364,
    kappa=0.449119,
    init_xi=0.078722,
    max_iter=200,
    verbose=1,
    random_state=46411588
)

from DMM_SVVS_Variational_v6_1 import DMM_SVVS_Variational_v6_1

model = DMM_SVVS_Variational_v6_1(
    K_max=4,
    nu=3.195442,
    zeta=1.0,
    eta=1.0,
    prune_threshold=0.679471,
    # ── SSL hyperparameters (replaces xi_1/xi_2/selection_prior) ──
    lambda0=11.700808,
    lambda1=0.040304,
    kappa=0.375142,
    xi_ema=0.318706,
    max_iter=200,
    verbose=1,
    random_state=268271558
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
        n_trials     = 100,          # small for demo; use 30-50 in practice
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






































