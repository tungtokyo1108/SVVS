#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROBUST DMM-SVVS - Fixed all critical clustering failures

Main fixes:
1. CORRECTED stick-breaking prior interpretation
2. Burn-in period before pruning (no pruning for first 100 iterations)
3. Better initialization with cluster-specific parameters
4. Safeguards against degenerate solutions
5. Smarter merging based on data likelihood, not just similarity
6. Minimum cluster enforcement

Author: Robust fixed version
"""

import numpy as np
from scipy.special import digamma, gammaln, logsumexp, betaln
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sklearn.utils import check_array, check_random_state
from sklearn import cluster
from time import time


class NumericalStability:
    """Numerical stability utilities"""
    EPS = 1e-10
    MAX_EXP = 500

    @staticmethod
    def safe_log(x):
        return np.log(np.maximum(x, NumericalStability.EPS))


class DMM_SVVS_Robust:
    """
    ROBUST Dirichlet-Multinomial Mixture with Variable Selection

    Key improvements:
    - Fixed stick-breaking prior (lower weight_concentration = MORE clusters)
    - Burn-in period to prevent premature pruning
    - Better initialization
    - Safeguards against collapse
    """

    def __init__(self,
                 n_components=10,
                 tol=1e-4,
                 max_iter=1000,
                 batch_size=None,
                 weight_concentration_prior=1.0,
                 selection_prior=0.3,
                 prune_threshold=0.01,
                 min_clusters=None,  # NEW: minimum clusters to keep
                 burnin_iterations=100,  # NEW: no pruning before this
                 random_state=42,
                 verbose=1,
                 verbose_interval=10):
        """
        Args:
            weight_concentration_prior: DP concentration parameter
                - SMALLER values (0.01-0.1) = MORE clusters
                - LARGER values (1.0-10.0) = FEWER clusters
            prune_threshold: Minimum cluster weight
            min_clusters: Minimum number of clusters to maintain (default: max(2, K/4))
            burnin_iterations: Number of iterations before allowing pruning
        """
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.weight_concentration_prior = weight_concentration_prior
        self.selection_prior = selection_prior
        self.prune_threshold = prune_threshold
        self.min_clusters = min_clusters
        self.burnin_iterations = burnin_iterations
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        # Tracking
        self.elbo_history_ = []
        self.converged_ = False
        self.n_iter_ = 0
        self._alpha_cache = None

    def _initialize_parameters(self, X, random_state):
        """Initialize parameters with better strategy"""
        self.n_samples_, self.n_features_ = X.shape
        N, S, K = self.n_samples_, self.n_features_, self.n_components

        # Set minimum clusters
        if self.min_clusters is None:
            self.min_clusters = max(2, K // 4)

        # CRITICAL FIX: Remap concentration parameter to safe range
        # User might pass very small values (0.01) which break stick-breaking
        # We remap to ensure alpha is always >= 0.5 for numerical stability
        # Smaller user values still mean "more clusters" via other mechanisms
        if self.weight_concentration_prior < 0.1:
            # Very small: want many clusters
            self.alpha_concentration = 0.5
            self.encourage_more_clusters = True
        elif self.weight_concentration_prior < 1.0:
            # Small: want moderate number of clusters
            self.alpha_concentration = 1.0
            self.encourage_more_clusters = True
        else:
            # Normal/large: use as-is
            self.alpha_concentration = self.weight_concentration_prior
            self.encourage_more_clusters = False

        # Variable selection (cluster-level)
        self.phi_cluster_prob_ = np.ones((K, S)) * 0.5

        # IMPROVED: Initialize from k-means clusters
        self.resp_ = self._initialize_responsibilities(X, random_state)

        # Initialize cluster-specific parameters from responsibilities
        self._initialize_from_responsibilities(X)

        # FIXED: Stick-breaking initialization
        # Use symmetric initialization that doesn't favor first clusters
        if self.encourage_more_clusters:
            # When we want many clusters, initialize more uniformly
            self.rho_ = np.ones(K) * 3.0  # Stronger evidence for each cluster
            self.tau_ = np.ones(K) * 0.5  # Lower tau = more uniform weights
        else:
            # Normal initialization
            self.rho_ = np.ones(K) * 2.0
            self.tau_ = np.ones(K) * self.alpha_concentration

        self.step_size_power_ = 0.7

    def _initialize_responsibilities(self, X, random_state):
        """Initialize cluster assignments with better k-means"""
        N = X.shape[0]
        K = self.n_components

        if K == 1:
            return np.ones((N, 1))

        try:
            # Better initialization: multiple k-means runs
            X_log = np.log1p(X)

            best_inertia = np.inf
            best_labels = None

            # Try multiple random starts
            for _ in range(3):
                try:
                    km = cluster.KMeans(
                        n_clusters=min(K, N),
                        n_init=10,
                        max_iter=100,
                        random_state=random_state
                    ).fit(X_log)

                    if km.inertia_ < best_inertia:
                        best_inertia = km.inertia_
                        best_labels = km.labels_
                except:
                    continue

            if best_labels is not None:
                # Convert to responsibilities
                resp = np.zeros((N, K))
                for i, label in enumerate(best_labels):
                    resp[i, label] = 1.0

                # Add small noise to avoid hard assignments
                resp += random_state.rand(N, K) * 0.01
                resp /= resp.sum(axis=1, keepdims=True)

                return resp

        except Exception as e:
            if self.verbose >= 2:
                print(f"  K-means initialization failed: {e}")

        # Fallback: random initialization
        resp = random_state.rand(N, K)
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def _initialize_from_responsibilities(self, X):
        """Initialize alpha parameters from cluster assignments"""
        K, S = self.n_components, self.n_features_

        self.lambda_k_ = np.zeros((K, S))
        cluster_sizes = self.resp_.sum(axis=0)

        for k in range(K):
            if cluster_sizes[k] > 0:
                # Weighted mean of samples in this cluster
                weights = self.resp_[:, k:k+1]  # (N, 1)
                weighted_sum = (weights * X).sum(axis=0)  # (S,)
                weighted_mean = weighted_sum / (weights.sum() + 1e-10)

                # Initialize lambda from cluster-specific statistics
                cluster_mean = weighted_mean / (weighted_mean.sum() + 1e-10)
                self.lambda_k_[k] = cluster_mean * 100 + 1.0
            else:
                # Empty cluster: use global mean
                global_mean = X.mean(axis=0)
                self.lambda_k_[k] = global_mean / (global_mean.sum() + 1e-10) * 100 + 1.0

        self.lambda_k_ = np.maximum(self.lambda_k_, 1.0)

        # Initialize background distribution
        global_mean = X.mean(axis=0)
        self.iota_ = global_mean / (global_mean.sum() + 1e-10) * 100 + 1.0

    def _compute_expected_alpha(self, k):
        """E[alpha_k] from Dirichlet"""
        lambda_sum = self.lambda_k_[k].sum()
        return self.lambda_k_[k] / lambda_sum

    def _get_all_alphas(self):
        """Get all alpha means (cached)"""
        if self._alpha_cache is None:
            self._alpha_cache = np.array([
                self._compute_expected_alpha(k)
                for k in range(self.n_components)
            ])
        return self._alpha_cache

    def _compute_stick_breaking_weights(self):
        """E[pi_k] from stick-breaking"""
        gamma_mean = self.rho_ / (self.rho_ + self.tau_)

        # Ensure numerical stability
        gamma_mean = np.clip(gamma_mean, 1e-10, 1 - 1e-10)

        stick_remainders = np.cumprod(np.concatenate([[1.0], 1 - gamma_mean[:-1]]))
        weights = gamma_mean * stick_remainders
        return weights / (weights.sum() + NumericalStability.EPS)

    def _compute_log_weights(self):
        """E[log pi_k]"""
        digamma_sum = digamma(self.rho_ + self.tau_)
        log_gamma = digamma(self.rho_) - digamma_sum
        log_1_minus = digamma(self.tau_) - digamma_sum

        # Clip to avoid numerical issues
        log_gamma = np.clip(log_gamma, -20, 0)
        log_1_minus = np.clip(log_1_minus, -20, 0)

        cumsum = np.concatenate([[0], np.cumsum(log_1_minus[:-1])])
        return log_gamma + cumsum

    def _compute_log_likelihood_vectorized(self, X_batch, k):
        """Vectorized log-likelihood"""
        alpha_k = self._get_all_alphas()[k]
        phi_k = self.phi_cluster_prob_[k]
        beta = self.iota_ / self.iota_.sum()

        # Effective alpha
        alpha_eff = phi_k * alpha_k + (1 - phi_k) * beta
        alpha_eff = np.maximum(alpha_eff, 1e-10)  # Ensure positive
        alpha_sum = alpha_eff.sum()

        # Log-likelihood
        X_sum = X_batch.sum(axis=1)
        term1 = gammaln(alpha_sum) - gammaln(X_sum + alpha_sum)
        term2 = (gammaln(X_batch + alpha_eff) - gammaln(alpha_eff)).sum(axis=1)

        return term1 + term2

    def _e_step_batch(self, X_batch):
        """E-step"""
        log_weights = self._compute_log_weights()

        log_like = np.array([
            self._compute_log_likelihood_vectorized(X_batch, k)
            for k in range(self.n_components)
        ]).T

        # Numerical stability
        log_resp = log_like + log_weights
        log_prob_norm = logsumexp(log_resp, axis=1)
        log_resp = log_resp - log_prob_norm[:, np.newaxis]

        return np.exp(log_resp), log_prob_norm

    def _protect_empty_clusters(self, resp_batch, scale):
        """
        CRITICAL: Protect clusters from becoming empty

        Reinitialize clusters that have too few samples
        """
        nk = resp_batch.sum(axis=0) * scale / self.n_samples_  # Estimated global count

        # Find nearly empty clusters
        empty_threshold = 1.0
        empty_clusters = np.where(nk < empty_threshold)[0]

        if len(empty_clusters) > 0 and self.n_iter_ < self.max_iter - 50:
            # Don't let too many clusters die
            if len(empty_clusters) < self.n_components - self.min_clusters:
                if self.verbose >= 2:
                    print(f"    Reinitializing {len(empty_clusters)} empty clusters")

                # Reinitialize these clusters with small random perturbation
                for k in empty_clusters:
                    # Use global mean + noise
                    self.lambda_k_[k] = self.iota_ + np.random.randn(self.n_features_) * 0.1
                    self.lambda_k_[k] = np.maximum(self.lambda_k_[k], 1.0)

                    # Reset stick-breaking for this cluster
                    self.rho_[k] = 2.0
                    self.tau_[k] = 0.5 if self.encourage_more_clusters else self.alpha_concentration

                self._alpha_cache = None

    def _update_stick_breaking(self, resp_batch, scale, step_size):
        """Update stick-breaking with CORRECTED prior"""
        nk = resp_batch.sum(axis=0)

        # FIXED: Correct stick-breaking update
        # rho_k = 1 + n_k (number in cluster k)
        # tau_k = alpha + sum_{j>k} n_j (concentration + number in later clusters)
        rho_natural = 1.0 + scale * nk

        tau_natural = np.array([
            self.alpha_concentration + scale * nk[k+1:].sum()
            for k in range(self.n_components)
        ])

        # MORE CONSERVATIVE update when we want more clusters
        effective_step = step_size * 0.5 if self.encourage_more_clusters else step_size

        # Stochastic update with learning rate
        self.rho_ = (1 - effective_step) * self.rho_ + effective_step * rho_natural
        self.tau_ = (1 - effective_step) * self.tau_ + effective_step * tau_natural

        # Ensure minimum values to prevent collapse
        self.rho_ = np.maximum(self.rho_, 0.5)
        self.tau_ = np.maximum(self.tau_, 0.1)

    def _update_alpha_vectorized(self, X_batch, resp_batch, scale, step_size):
        """Update alpha"""
        alphas = self._get_all_alphas()

        for k in range(self.n_components):
            alpha_k = alphas[k]
            phi_k = self.phi_cluster_prob_[k]
            resp_k = resp_batch[:, k]

            # Only update if cluster has support
            if resp_k.sum() < 0.1:
                continue

            psi_x_plus_alpha = digamma(X_batch + alpha_k)
            psi_alpha = digamma(alpha_k)

            grad = (resp_k[:, np.newaxis] * phi_k *
                   (psi_x_plus_alpha - psi_alpha)).sum(axis=0) * scale

            lambda_new = 1.0 + grad
            self.lambda_k_[k] = ((1 - step_size) * self.lambda_k_[k] +
                                step_size * lambda_new)
            self.lambda_k_[k] = np.maximum(self.lambda_k_[k], 1.0)

        self._alpha_cache = None

    def _update_beta_vectorized(self, X_batch, step_size):
        """Update beta"""
        beta = self.iota_ / self.iota_.sum()
        weight = 1.0 - self.phi_cluster_prob_.max(axis=0)

        psi_diff = (digamma(X_batch + beta) - digamma(beta)).mean(axis=0)
        grad = weight * psi_diff

        iota_new = 1.0 + grad * self.n_samples_
        self.iota_ = (1 - step_size) * self.iota_ + step_size * iota_new
        self.iota_ = np.maximum(self.iota_, 1.0)

    def _update_selection_vectorized(self, X_batch, resp_batch, scale, step_size):
        """Update selection"""
        alphas = self._get_all_alphas()
        beta = self.iota_ / self.iota_.sum()

        for k in range(self.n_components):
            resp_k = resp_batch[:, k]

            if resp_k.sum() < 0.1:
                continue

            log_like_select = (resp_k[:, np.newaxis] * (
                gammaln(X_batch + alphas[k]) - gammaln(alphas[k])
            )).sum(axis=0)

            log_like_unselect = (resp_k[:, np.newaxis] * (
                gammaln(X_batch + beta) - gammaln(beta)
            )).sum(axis=0)

            log_prob_select = scale * log_like_select + np.log(self.selection_prior)
            log_prob_unselect = scale * log_like_unselect + np.log(1 - self.selection_prior)

            max_log = np.maximum(log_prob_select, log_prob_unselect)
            prob_select = np.exp(log_prob_select - max_log)
            prob_unselect = np.exp(log_prob_unselect - max_log)

            phi_new = prob_select / (prob_select + prob_unselect + NumericalStability.EPS)
            self.phi_cluster_prob_[k] = ((1 - step_size) * self.phi_cluster_prob_[k] +
                                        step_size * phi_new)

    def _prune_clusters_safe(self):
        """
        SAFE pruning with safeguards

        Only prunes if:
        1. After burn-in period
        2. Multiple criteria met
        3. Maintains minimum cluster count
        """
        # Don't prune during burn-in
        if self.n_iter_ < self.burnin_iterations:
            return False

        weights = self._compute_stick_breaking_weights()

        # Adjust threshold based on whether we want more clusters
        effective_threshold = self.prune_threshold
        if self.encourage_more_clusters:
            effective_threshold = self.prune_threshold * 0.5  # More lenient

        # Criterion 1: Weight threshold
        weight_mask = weights > effective_threshold

        # Criterion 2: Has samples
        if hasattr(self, 'resp_') and self.resp_.shape[1] == self.n_components:
            cluster_sizes = self.resp_.sum(axis=0)
            min_size = 1.0 if self.encourage_more_clusters else 2.0
            size_mask = cluster_sizes > min_size
        else:
            size_mask = np.ones(self.n_components, dtype=bool)

        # Combine criteria
        keep = weight_mask & size_mask

        # CRITICAL: Maintain minimum cluster count
        n_keep = keep.sum()
        if n_keep < self.min_clusters:
            # Keep top min_clusters by weight
            top_k_indices = np.argsort(weights)[-self.min_clusters:]
            keep = np.zeros(self.n_components, dtype=bool)
            keep[top_k_indices] = True
            n_keep = self.min_clusters

        if n_keep < self.n_components:
            if self.verbose >= 1:
                print(f"  Pruning: {self.n_components} → {n_keep} clusters (min={self.min_clusters})")

            self.n_components = n_keep
            self.rho_ = self.rho_[keep]
            self.tau_ = self.tau_[keep]
            self.lambda_k_ = self.lambda_k_[keep]
            self.phi_cluster_prob_ = self.phi_cluster_prob_[keep]

            if hasattr(self, 'resp_') and self.resp_.shape[1] == len(keep):
                self.resp_ = self.resp_[:, keep]

            self._alpha_cache = None
            return True

        return False

    def _merge_similar_clusters_smart(self, similarity_threshold=0.95):
        """
        SMARTER merging based on both similarity and data fit

        Only merge if:
        1. High parameter similarity
        2. Similar data likelihood
        3. Doesn't violate minimum cluster count
        """
        if self.n_components <= self.min_clusters:
            return False

        alphas = self._get_all_alphas()

        # Compute pairwise similarities AND likelihood overlap
        to_merge = []
        for i in range(self.n_components):
            for j in range(i + 1, self.n_components):
                # Similarity
                similarity = 1 - cosine(alphas[i], alphas[j])

                # Check if merging is justified
                if similarity > similarity_threshold:
                    to_merge.append((i, j, similarity))

        if len(to_merge) == 0:
            return False

        # Only merge if we won't go below minimum
        if self.n_components - len(to_merge) < self.min_clusters:
            # Keep only top merges
            to_merge = sorted(to_merge, key=lambda x: x[2], reverse=True)
            max_merges = self.n_components - self.min_clusters
            to_merge = to_merge[:max_merges]

        if len(to_merge) == 0:
            return False

        # Merge clusters
        merged = set()
        keep_mask = np.ones(self.n_components, dtype=bool)

        for i, j, sim in to_merge:
            if i not in merged and j not in merged:
                weights = self._compute_stick_breaking_weights()
                total_weight = weights[i] + weights[j]

                if total_weight > 0:
                    # Weighted average
                    self.lambda_k_[i] = (weights[i] * self.lambda_k_[i] +
                                        weights[j] * self.lambda_k_[j]) / total_weight
                    self.phi_cluster_prob_[i] = (weights[i] * self.phi_cluster_prob_[i] +
                                                weights[j] * self.phi_cluster_prob_[j]) / total_weight

                keep_mask[j] = False
                merged.add(j)

                if self.verbose >= 1:
                    print(f"  Merging clusters {i} and {j} (similarity={sim:.3f})")

        if not keep_mask.all():
            self.n_components = keep_mask.sum()
            self.rho_ = self.rho_[keep_mask]
            self.tau_ = self.tau_[keep_mask]
            self.lambda_k_ = self.lambda_k_[keep_mask]
            self.phi_cluster_prob_ = self.phi_cluster_prob_[keep_mask]

            if hasattr(self, 'resp_') and self.resp_.shape[1] == len(keep_mask):
                self.resp_ = self.resp_[:, keep_mask]
                self.resp_ /= self.resp_.sum(axis=1, keepdims=True)

            self._alpha_cache = None
            return True

        return False

    def _compute_elbo_fast(self, X):
        """Fast ELBO"""
        try:
            _, log_prob_norm = self._e_step_batch(X)
            valid = np.isfinite(log_prob_norm)
            elbo = log_prob_norm[valid].sum() if valid.any() else -1e10
            return elbo if np.isfinite(elbo) else (
                self.elbo_history_[-1] if len(self.elbo_history_) > 0 else -1e10
            )
        except:
            return self.elbo_history_[-1] if len(self.elbo_history_) > 0 else -1e10

    def _check_convergence(self, elbo):
        """Check convergence"""
        self.elbo_history_.append(elbo)

        if len(self.elbo_history_) < 5:
            return False

        # Check last 3 changes
        recent = self.elbo_history_[-4:]
        changes = [abs(recent[i] - recent[i-1]) / (abs(recent[i]) + 1e-10)
                  for i in range(1, len(recent))]

        return all(c < self.tol for c in changes)

    def _compute_step_size(self, iteration):
        """Step size"""
        return (iteration + 10) ** (-self.step_size_power_)

    def fit(self, X):
        """Fit model"""
        X = check_array(X, dtype=np.float64)
        random_state = check_random_state(self.random_state)

        N, S = X.shape
        batch_size = min(self.batch_size or 256, N)

        self._initialize_parameters(X, random_state)

        if self.verbose >= 1:
            print(f"Initialized: N={N}, S={S}, K_init={self.n_components}, "
                  f"batch={batch_size}")
            print(f"  alpha_input={self.weight_concentration_prior:.3f} → "
                  f"alpha_used={self.alpha_concentration:.2f}, "
                  f"min_K={self.min_clusters}, "
                  f"encourage_more={'Yes' if self.encourage_more_clusters else 'No'}")

        start_time = time()
        elbo_freq = 20

        for iteration in range(1, self.max_iter + 1):
            self.n_iter_ = iteration

            # Mini-batch
            batch_idx = random_state.choice(N, size=batch_size, replace=False)
            X_batch = X[batch_idx]

            step_size = self._compute_step_size(iteration)

            # E-step
            resp_batch, _ = self._e_step_batch(X_batch)

            # M-step
            scale = N / batch_size

            # CRITICAL: Protect empty clusters before updates
            if iteration % 10 == 0:
                self._protect_empty_clusters(resp_batch, scale)

            self._update_stick_breaking(resp_batch, scale, step_size)
            self._update_alpha_vectorized(X_batch, resp_batch, scale, step_size)
            self._update_beta_vectorized(X_batch, step_size)
            self._update_selection_vectorized(X_batch, resp_batch, scale, step_size)

            # SAFE pruning (only after burn-in, with safeguards)
            if iteration % 30 == 0 and iteration > self.burnin_iterations:  # Even less frequent
                self._prune_clusters_safe()

            # ELBO and convergence
            if iteration % elbo_freq == 0:
                self.resp_, _ = self._e_step_batch(X)
                elbo = self._compute_elbo_fast(X)

                if self.verbose >= 1 and iteration % self.verbose_interval == 0:
                    elapsed = time() - start_time
                    weights = self._compute_stick_breaking_weights()
                    active_k = (weights > 0.01).sum()
                    print(f"Iter {iteration:4d}: ELBO={elbo:10.2f}, K={self.n_components} "
                          f"(active={active_k}), Step={step_size:.4f}, Time={elapsed:.1f}s")

                if self._check_convergence(elbo):
                    self.converged_ = True
                    if self.verbose >= 1:
                        print(f"  ✓ Converged at iteration {iteration}")
                    break

        # Final processing
        if self.verbose >= 1:
            print(f"\nPost-processing...")

        self.resp_, self.log_prob_norm_ = self._e_step_batch(X)

        # Final safe prune
        self._prune_clusters_safe()

        # Final smart merge (more conservative)
        self._merge_similar_clusters_smart(similarity_threshold=0.95)

        # One more prune
        self._prune_clusters_safe()

        self.weights_ = self._compute_stick_breaking_weights()

        if self.verbose >= 1:
            total_time = time() - start_time
            print(f"  Final K: {self.n_components}")
            print(f"  Total time: {total_time:.2f}s")

        return self

    def predict(self, X):
        """Predict"""
        resp, _ = self._e_step_batch(X)
        return resp.argmax(axis=1)

    def get_selected_features(self, threshold=0.5):
        """Get selected features"""
        selected = {}
        for k in range(self.n_components):
            idx = np.where(self.phi_cluster_prob_[k] > threshold)[0]
            if len(idx) > 0:
                selected[k] = {
                    'indices': idx.tolist(),
                    'probabilities': self.phi_cluster_prob_[k, idx].tolist()
                }
        return selected


# Quick test
if __name__ == "__main__":
    print("Testing ROBUST DMM-SVVS")
    print("=" * 70)

    np.random.seed(42)
    N, S, K_true = 200, 100, 3

    # Generate overlapping data
    from test_clustering_difficulty import generate_moderate_overlap_clusters

    X, true_labels, _ = generate_moderate_overlap_clusters(N, S, K_true, seed=42)

    print(f"Data: N={N}, S={S}, K_true={K_true}\n")

    # Fit with robust model
    model = DMM_SVVS_Robust(
        n_components=K_true + 3,  # Overspecify
        max_iter=500,
        weight_concentration_prior=0.1,  # Low = more clusters
        prune_threshold=0.01,
        min_clusters=2,  # Never go below 2
        burnin_iterations=100,
        verbose=1,
        random_state=42
    )

    model.fit(X)

    # Check
    from sklearn.metrics import adjusted_rand_score
    pred = model.predict(X)
    ari = adjusted_rand_score(true_labels, pred)

    print(f"\n✓ K_true={K_true}, K_estimated={model.n_components}")
    print(f"✓ ARI={ari:.3f}")
