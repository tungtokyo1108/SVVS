#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMM-SVVS with Quantum Annealing Variational Bayes (QAVB)
=========================================================
Extends DMM_SVVS_Variational_v2 by replacing the standard softmax E-step
with a quantum density-matrix cluster assignment following:

    Miyahara & Roychowdhury, PNAS 2023 (DOI: 10.1073/pnas.2212660120)

Adaptation to DMM-SVVS
-----------------------
The classical energy for sample i at cluster k is:

    H_cl[i,k] = -(E[log π_k] + E[log p(x_i | z_i=k)])

QAVB replaces the softmax E-step with a Gibbs density matrix:

    rho_i = exp(-β*(1-s)*diag(H_cl_norm[i]) - β*s*H_qu) / Z_i
    r[i,k] = rho_i[k,k]   (diagonal elements = soft assignments)

where H_cl_norm[i] is the per-sample energy shifted and normalised to [0,1]
so that the quantum Hamiltonian H_qu (eigenvalues in [-2, 2]) has a comparable
scale to the classical term — making quantum tunnelling meaningful regardless
of the absolute magnitude of the DMM log-likelihood.

Key design fixes for DMM (vs. direct GMM transplant)
------------------------------------------------------
FIX A — Energy normalisation:
    DMM log-likelihood differences can be O(1000×) larger than H_qu
    eigenvalues.  Without normalisation, the quantum term is negligible
    and QAVB degenerates to standard softmax.  We normalise each sample's
    energy vector to [0, 1] so classical and quantum terms compete on equal
    footing throughout the annealing schedule.

FIX B — k-means anchor for M-step:
    With s=1 (full quantum start), r_quantum is uniform, which would make
    every M-step update wash out the k-means cluster structure in λ_star.
    We preserve cluster-specific parameters by blending:
        r_mstep = (1-s_t)*r_quantum + s_t*r_kmeans
    so M-step parameters transition smoothly from k-means-anchored to
    quantum-guided as the quantum phase completes.

Annealing schedule  (t = 0-indexed iteration)
----------------------------------------------
    s_t   = s0 * max(1 - t/τ1, 0)             quantum strength: s0 → 0 by τ1
    β_t   = β0                   if t ≤ τ1
          = 1 + (β0-1)*(τ2-t)/(τ2-τ1)  τ1 < t ≤ τ2   (thermal annealing)
          = 1                    if t > τ2

Phase summary
~~~~~~~~~~~~~
  [1, τ1]:    Quantum annealing  — s: s0→0, quantum density-matrix E-step
  (τ1, τ2]:   Thermal annealing  — s=0, β: β0→1, temperature-scaled softmax
  (τ2, ∞):    Standard CAVI      — β=1, s=0

Three classes provided
----------------------
    DMM_SVVS_DAVB            — deterministic annealing VB (no quantum term;
                               simpler, robust, recommended first step)
    DMM_SVVS_ClassicalQAVB   — QAVB via scipy.linalg.expm (no extra deps)
    DMM_SVVS_PennyLaneQAVB   — QAVB via PennyLane quantum-circuit simulation
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp
from sklearn.utils import check_array, check_random_state
from time import time
import sys
import os

# ── Import base class ────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from DMM_SVVS_Variational_v2 import DMM_SVVS_Variational_v2, NumericalStability


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cyclic_hamiltonian(K: int) -> np.ndarray:
    """
    K×K cyclic adjacency matrix (quantum Hamiltonian H_qu).

    H_qu[k, (k+1)%K] = H_qu[(k+1)%K, k] = 1  for all k.

    When K is small, expm(-β·H_qu) has uniform diagonal for a symmetric
    cyclic matrix, so at s=1 all samples get equal cluster assignments
    regardless of data — the quantum phase is data-agnostic by design.
    """
    H = np.zeros((K, K))
    for k in range(K):
        H[k, (k + 1) % K] = 1.0
        H[(k + 1) % K, k] = 1.0
    return H


def _annealing_schedule(t: int, beta0: float, s0: float,
                        tau1: int, tau2: int):
    """
    Return (beta_t, s_t) for 0-indexed iteration t.

    s_t  : quantum strength, decreases s0 → 0 linearly over [0, tau1].
    beta_t: inverse temperature, stays at beta0 until tau1, then
            decreases to 1.0 linearly over (tau1, tau2].
    """
    s_t = s0 * max(1.0 - t / tau1, 0.0)

    if t <= tau1:
        beta_t = float(beta0)
    elif t <= tau2:
        frac   = (tau2 - t) / (tau2 - tau1)
        beta_t = 1.0 + (beta0 - 1.0) * frac
    else:
        beta_t = 1.0

    return float(beta_t), float(s_t)


# ─────────────────────────────────────────────────────────────────────────────
# Mixin: shared CAVI loop logic for all annealed classes
# ─────────────────────────────────────────────────────────────────────────────

class _AnnealedDMMMixin:
    """
    Common fit-loop infrastructure shared by DAVB and QAVB classes.

    Subclasses must implement:
        _compute_r_annealed(X, beta_t, s_t) → ndarray (N, K)
            Returns new cluster responsibilities (does NOT set self.r).
        _r_for_mstep(r_annealed, s_t) → ndarray (N, K)
            Returns the r to use for M-step updates (may differ from
            r_annealed, e.g. blended with k-means anchor).
    """

    # ── Shared temperature-scaled CAVI E-step ────────────────────────────

    def _update_r_classical(self, X: np.ndarray, beta_t: float = 1.0):
        """
        Temperature-scaled softmax E-step (used in thermal and CAVI phases).

        log r[i,k]  ∝  β * (E[log π_k] + E[log p(x_i|z=k)])
        beta_t = 1  →  standard CAVI.
        """
        EPS = NumericalStability.EPS
        E_log_pi = self._E_log_pi()
        ll       = self._expected_log_lik(X)
        log_r    = beta_t * (E_log_pi[None, :] + ll)
        log_r   -= logsumexp(log_r, axis=1, keepdims=True)
        self.r   = np.exp(log_r)
        self.r   = np.maximum(self.r, EPS)
        self.r  /= self.r.sum(axis=1, keepdims=True)

    # ── Common fit loop (overrides DMM_SVVS_Variational_v2.fit) ─────────

    def fit(self, X):
        X            = check_array(X, dtype=np.float64)
        random_state = check_random_state(self.random_state)

        self._initialize_parameters(X, random_state)

        if self.verbose >= 1:
            self._print_fit_header()
            print("=" * 70)

        t0 = time()

        for iteration in range(1, self.max_iter + 1):
            self.n_iter = iteration
            self._clear_cache()

            # 0-indexed t for annealing schedule
            beta_t, s_t = _annealing_schedule(
                iteration - 1, self.beta0, self.s0, self.tau1, self.tau2
            )

            # ── E-step ────────────────────────────────────────────────────
            if s_t > 1e-8:
                # Quantum / annealing phase: compute new r
                r_new = self._compute_r_annealed(X, beta_t, s_t)
                # r for M-step may differ (e.g. blended with k-means anchor)
                self.r = self._r_for_mstep(r_new, s_t)
            else:
                # Classical phase: standard β-scaled softmax
                self._update_r_classical(X, beta_t)
                r_new = self.r  # same as M-step r

            # ── M-step ────────────────────────────────────────────────────
            self._update_f(X)
            self._update_theta()
            self._update_xi_star()
            self._update_lambda_star(X)
            self._update_iota_star(X)

            # Restore quantum r (for ELBO and next E-step),
            # not the M-step blend
            if s_t > 1e-8:
                self.r = r_new

            # ── Pruning (deferred until after quantum phase) ──────────────
            if iteration >= self.prune_start and iteration % self.prune_every == 0:
                self._prune_empty_clusters()

            # ── ELBO + convergence check ──────────────────────────────────
            if iteration % 10 == 0:
                elbo = self._compute_elbo(X)
                self.elbo_history.append(elbo)

                if self.verbose >= 1:
                    phase = ("quantum"   if s_t > 1e-8          else
                             "annealing" if beta_t > 1.0 + 1e-6 else
                             "CAVI")
                    print(f"Iter {iteration:4d}: ELBO={elbo:14.2f}, K={self.K}, "
                          f"β={beta_t:.2f}, s={s_t:.3f}, [{phase}], "
                          f"Time={time()-t0:.1f}s")

                # Convergence only after annealing completes + at least one prune
                if (iteration > self.tau2 and
                        self._pruned_at_least_once and
                        len(self.elbo_history) >= 3):
                    recent  = self.elbo_history[-3:]
                    changes = [abs(recent[i] - recent[i - 1]) /
                               (abs(recent[i]) + 1e-10)
                               for i in range(1, len(recent))]
                    if all(c < self.tol for c in changes):
                        self.converged = True
                        if self.verbose >= 1:
                            print(f"\n✓ Converged at iteration {iteration}")
                        break

        # Final pruning and weight computation
        self._prune_empty_clusters()
        self.weights_ = self._compute_weights()

        if self.verbose >= 1:
            print(f"\nFinal: K={self.K}, "
                  f"weights={np.round(self.weights_, 4)}, "
                  f"time={time()-t0:.2f}s, "
                  f"converged={self.converged}")

        return self


# ─────────────────────────────────────────────────────────────────────────────
# 1. Deterministic Annealing VB (DAVB) — no quantum term
# ─────────────────────────────────────────────────────────────────────────────

class DMM_SVVS_DAVB(_AnnealedDMMMixin, DMM_SVVS_Variational_v2):
    """
    DMM-SVVS with Deterministic Annealing Variational Bayes (DAVB).

    Replaces the standard softmax E-step with a temperature-scaled version:

        log r[i,k] ∝  β_t * (E[log π_k] + E[log p(x_i | z=k)])

    β_t anneals from β0 (≪ 1, nearly uniform) to 1.0 (standard CAVI).
    Starting from soft, near-uniform assignments avoids the k-means local
    optima that standard CAVI can get trapped in.

    No quantum term (s=0 always) — no PennyLane dependency.

    Parameters (beyond DMM_SVVS_Variational_v2)
    --------------------------------------------
    beta0 : float
        Initial inverse temperature (e.g. 0.01).  β → 1.0 by tau2.
    s0    : float
        Kept at 0.0 (DAVB has no quantum term; s0 is ignored internally).
    tau1  : int
        Unused for DAVB (no quantum phase); kept for API compatibility.
    tau2  : int
        Iteration at which β reaches 1.0 (annealing complete).
    """

    def __init__(
        self,
        K_max=10,
        nu='auto',
        zeta=1.0,
        eta=1.0,
        xi_1=1.0,
        xi_2=1.0,
        selection_prior=0.3,
        tol=1e-4,
        max_iter=500,
        prune_threshold=0.02,
        min_clusters=None,
        prune_start=10,
        prune_every=5,
        verbose=1,
        random_state=42,
        # Annealing
        beta0=0.01,
        s0=0.0,    # DAVB: no quantum term
        tau1=1,    # DAVB: no quantum phase (set to 1 to avoid /0)
        tau2=150,
    ):
        super().__init__(
            K_max=K_max, nu=nu, zeta=zeta, eta=eta,
            xi_1=xi_1, xi_2=xi_2,
            selection_prior=selection_prior,
            tol=tol, max_iter=max_iter,
            prune_threshold=prune_threshold,
            min_clusters=min_clusters,
            prune_start=prune_start,
            prune_every=prune_every,
            verbose=verbose,
            random_state=random_state,
        )
        self.beta0 = float(beta0)
        self.s0    = 0.0           # DAVB: purely classical
        self.tau1  = int(tau1)
        self.tau2  = int(tau2)

    def _print_fit_header(self):
        print(f"\nStarting DAVB — DMM-SVVS")
        print(f"  β0={self.beta0}, τ2={self.tau2}, "
              f"prune_start={self.prune_start}")

    # DAVB has no quantum E-step — _compute_r_annealed is never called
    # because s0=0 → s_t=0 always → the quantum branch is never entered.
    def _compute_r_annealed(self, X, beta_t, s_t):
        raise NotImplementedError("DAVB has no quantum E-step")

    def _r_for_mstep(self, r_annealed, s_t):
        return r_annealed  # not used in DAVB (s_t always 0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Classical QAVB  (scipy.linalg.expm)
# ─────────────────────────────────────────────────────────────────────────────

class DMM_SVVS_ClassicalQAVB(_AnnealedDMMMixin, DMM_SVVS_Variational_v2):
    """
    DMM-SVVS with Quantum Annealing VB using direct matrix exponentials.

    Quantum E-step (during [1, tau1])
    ----------------------------------
    For each sample i:
        h_i      = H_cl[i] - min_k H_cl[i]     (shift: h ≥ 0)
        h_norm_i = h_i / max(h_i, ε)            [FIX A: normalise to [0,1]]
        M_i      = diag(clip(-β(1-s)h_norm_i, -500, 0)) - β·s·H_qu
        rho_i    = expm(M_i) / trace(expm(M_i))
        r[i,k]   = diag(rho_i)[k].clip(ε)

    M-step anchor  [FIX B]
    -----------------------
    To prevent λ_star from collapsing when r is quantum-mixed (uniform-like),
    the M-step uses a blend of the quantum r and the initial k-means r:

        r_mstep = (1 - s_t) * r_quantum + s_t * r_kmeans

    At s_t = 1 (start): M-step uses k-means assignments (cluster-specific).
    At s_t = 0 (end):   M-step uses quantum assignments (data-driven).

    Parameters (beyond DMM_SVVS_Variational_v2)
    --------------------------------------------
    beta0 : float
        Initial inverse temperature.  Default 30.0 (as in the PNAS paper).
    s0    : float
        Initial quantum strength (1.0 = fully quantum start).
    tau1  : int
        Iteration at which s reaches 0 (quantum annealing ends).
    tau2  : int
        Iteration at which β reaches 1 (thermal annealing ends; > tau1).
    """

    def __init__(
        self,
        K_max=10,
        nu='auto',
        zeta=1.0,
        eta=1.0,
        xi_1=1.0,
        xi_2=1.0,
        selection_prior=0.3,
        tol=1e-4,
        max_iter=600,
        prune_threshold=0.02,
        min_clusters=None,
        prune_start=10,
        prune_every=5,
        verbose=1,
        random_state=42,
        # QAVB-specific
        beta0=30.0,
        s0=1.0,
        tau1=100,
        tau2=200,
    ):
        # Pruning must not fire during the quantum phase
        effective_prune_start = max(int(prune_start), int(tau1) + int(prune_every))

        super().__init__(
            K_max=K_max, nu=nu, zeta=zeta, eta=eta,
            xi_1=xi_1, xi_2=xi_2,
            selection_prior=selection_prior,
            tol=tol, max_iter=max_iter,
            prune_threshold=prune_threshold,
            min_clusters=min_clusters,
            prune_start=effective_prune_start,
            prune_every=prune_every,
            verbose=verbose,
            random_state=random_state,
        )
        self.beta0    = float(beta0)
        self.s0       = float(s0)
        self.tau1     = int(tau1)
        self.tau2     = int(tau2)
        self.H_qu     = None
        self.r_kmeans = None   # k-means anchor for M-step [FIX B]

    # ── Quantum Hamiltonian ───────────────────────────────────────────────

    def _rebuild_H_qu(self):
        self.H_qu = _cyclic_hamiltonian(self.K)

    # ── Override initialisation ───────────────────────────────────────────

    def _initialize_parameters(self, X, random_state):
        # Parent initialises λ_star from k-means AND r from k-means
        super()._initialize_parameters(X, random_state)

        # Save k-means r as anchor for M-step [FIX B]
        self.r_kmeans = self.r.copy()

        # Build quantum Hamiltonian
        self._rebuild_H_qu()

    # ── Override pruning to rebuild H_qu ─────────────────────────────────

    def _prune_empty_clusters(self):
        pruned = super()._prune_empty_clusters()
        if pruned:
            self._rebuild_H_qu()
            # r_kmeans is only used when s_t > 0 (before pruning fires),
            # so no reshape needed here.
        return pruned

    # ── Quantum E-step (density matrix diagonal) ──────────────────────────

    def _compute_r_annealed(self, X: np.ndarray, beta_t: float, s_t: float) -> np.ndarray:
        """
        Quantum soft assignments via Gibbs density matrix.

        FIX A: per-sample energy normalisation to [0,1] ensures the quantum
        Hamiltonian (eigenvalues ≈ ±2) competes on equal footing with the
        classical term regardless of DMM log-likelihood magnitude.
        """
        EPS      = NumericalStability.EPS
        E_log_pi = self._E_log_pi()
        ll       = self._expected_log_lik(X)
        H_cl     = -(E_log_pi[None, :] + ll)   # (N, K) energy to minimise

        N, K = H_cl.shape
        r    = np.zeros((N, K))

        for i in range(N):
            h = H_cl[i] - H_cl[i].min()           # shift: h ≥ 0

            # [FIX A] Normalise to [0,1] so quantum term is meaningful
            h_range = h.max()
            h_norm  = h / h_range if h_range > 1e-10 else h

            M_diag = np.clip(-beta_t * (1.0 - s_t) * h_norm, -500.0, 0.0)
            M      = np.diag(M_diag) - beta_t * s_t * self.H_qu

            rho    = expm(M)
            tr     = np.trace(rho)
            rho    = rho / tr if abs(tr) > 1e-10 else np.eye(K) / K

            diag   = np.real(np.diag(rho)).clip(EPS)
            r[i]   = diag / diag.sum()

        return r

    # ── Blended M-step r [FIX B] ─────────────────────────────────────────

    def _r_for_mstep(self, r_annealed: np.ndarray, s_t: float) -> np.ndarray:
        """
        Blend quantum r with k-means anchor for M-step parameters.

        r_mstep = (1 - s_t) * r_quantum + s_t * r_kmeans

        At s_t=1 (full quantum start): M-step uses k-means → λ_star
        retains cluster-specific structure from k-means initialisation.
        At s_t=0: M-step uses fully quantum r → standard data-driven update.
        """
        EPS     = NumericalStability.EPS
        r_blend = (1.0 - s_t) * r_annealed + s_t * self.r_kmeans
        r_blend  = np.maximum(r_blend, EPS)
        r_blend /= r_blend.sum(axis=1, keepdims=True)
        return r_blend

    def _print_fit_header(self):
        print(f"\nStarting Classical QAVB — DMM-SVVS")
        print(f"  β0={self.beta0}, s0={self.s0}, "
              f"τ1={self.tau1}, τ2={self.tau2}, "
              f"prune_start={self.prune_start}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PennyLane QAVB  (quantum circuit purification)
# ─────────────────────────────────────────────────────────────────────────────

class DMM_SVVS_PennyLaneQAVB(DMM_SVVS_ClassicalQAVB):
    """
    DMM-SVVS QAVB implemented via PennyLane quantum circuit simulation.

    Produces identical numerical results to DMM_SVVS_ClassicalQAVB on a
    noiseless state-vector simulator; on real quantum hardware shot noise and
    gate errors would be present.

    Quantum architecture  (per data point i)
    -----------------------------------------
    System register:  n_sys = ceil(log2(K)) qubits   → K-dimensional cluster space
    Ancilla register: n_sys qubits                   → purification
    Total:            2·n_sys qubits

    Circuit
    -------
    1. Build M_i and compute normalised Gibbs density matrix rho_i (K_pad×K_pad).
    2. Eigendecompose rho_i: rho_i = Σ_k λ_k |φ_k⟩⟨φ_k|.
    3. Purification: |Ψ_i⟩ = Σ_k √λ_k |φ_k⟩_sys ⊗ |k⟩_anc.
    4. PennyLane StatePrep → trace out ancilla → rho_sys.
    5. r[i,k] = diag(rho_sys)[k].

    Parameters: same as DMM_SVVS_ClassicalQAVB.
    Requires:   pennylane >= 0.30  (pip install pennylane)
    """

    # ── PennyLane device management ───────────────────────────────────────

    def _init_pennylane_device(self):
        """Build PennyLane device and QNode for current K."""
        try:
            import pennylane as qml
        except ImportError:
            raise ImportError(
                "PennyLane is required.  Install via:  pip install pennylane"
            )

        self._qml     = qml
        self.n_sys    = max(1, int(np.ceil(np.log2(self.K))))
        self.K_pad    = 2 ** self.n_sys
        self.n_tot    = 2 * self.n_sys

        # Pad quantum Hamiltonian to K_pad×K_pad
        self.H_qu_pad = np.zeros((self.K_pad, self.K_pad))
        self.H_qu_pad[:self.K, :self.K] = self.H_qu

        self._dev   = qml.device("default.qubit",
                                  wires=list(range(self.n_tot)))
        self._qnode = qml.QNode(self._circuit, self._dev)

    def _circuit(self, purif_state):
        qml = self._qml
        qml.StatePrep(purif_state,
                      wires=list(range(self.n_tot)),
                      pad_with=0.0)
        return qml.density_matrix(wires=list(range(self.n_sys)))

    # ── Override initialisation and pruning ──────────────────────────────

    def _initialize_parameters(self, X, random_state):
        super()._initialize_parameters(X, random_state)
        self._init_pennylane_device()

    def _prune_empty_clusters(self):
        pruned = super()._prune_empty_clusters()
        if pruned:
            self._init_pennylane_device()
        return pruned

    # ── Gibbs density matrix (padded to K_pad×K_pad) ─────────────────────

    def _gibbs_density_matrix_padded(
        self, h_norm: np.ndarray, beta_t: float, s_t: float
    ) -> np.ndarray:
        """
        K_pad×K_pad normalised density matrix for the purification circuit.
        h_norm is already shifted and normalised to [0,1].
        """
        K, K_pad = self.K, self.K_pad
        M_diag   = np.clip(-beta_t * (1.0 - s_t) * h_norm, -500.0, 0.0)

        M_pad = np.zeros((K_pad, K_pad))
        M_pad[:K, :K] = np.diag(M_diag) - beta_t * s_t * self.H_qu

        rho_unnorm = expm(M_pad)
        tr         = np.trace(rho_unnorm)
        return rho_unnorm / tr if abs(tr) > 1e-10 else np.eye(K_pad) / K_pad

    # ── Purification state vector ─────────────────────────────────────────

    def _build_purification(self, rho_pad: np.ndarray) -> np.ndarray:
        """
        Purification |Ψ⟩ = Σ_k √λ_k |φ_k⟩_sys ⊗ |k⟩_anc of rho_pad.
        Returns a K_pad²-dimensional complex state vector.
        """
        K_pad = self.K_pad

        eigenvalues, eigenvectors = np.linalg.eigh(rho_pad)
        eigenvalues = eigenvalues.clip(0)
        s = eigenvalues.sum()
        if s > 1e-10:
            eigenvalues /= s

        purif = np.zeros(K_pad * K_pad, dtype=complex)
        for k in range(K_pad):
            amp = np.sqrt(eigenvalues[k])
            if amp < 1e-15:
                continue
            for j in range(K_pad):
                purif[j * K_pad + k] += amp * eigenvectors[j, k]

        norm = np.linalg.norm(purif)
        return purif / norm if norm > 1e-10 else purif

    # ── Per-sample quantum E-step via PennyLane circuit ──────────────────

    def _pennylane_diagonal(
        self, h_norm: np.ndarray, beta_t: float, s_t: float
    ) -> np.ndarray:
        EPS     = NumericalStability.EPS
        rho_pad = self._gibbs_density_matrix_padded(h_norm, beta_t, s_t)
        purif   = self._build_purification(rho_pad)
        rho_sys = self._qnode(purif)                    # (K_pad, K_pad)
        diag    = np.real(np.diag(rho_sys))[:self.K].clip(EPS)
        return diag / diag.sum()

    # ── Override quantum E-step to use PennyLane ─────────────────────────

    def _compute_r_annealed(self, X: np.ndarray, beta_t: float, s_t: float) -> np.ndarray:
        """Quantum assignments via PennyLane circuit (replaces ClassicalQAVB expm)."""
        EPS      = NumericalStability.EPS
        E_log_pi = self._E_log_pi()
        ll       = self._expected_log_lik(X)
        H_cl     = -(E_log_pi[None, :] + ll)   # (N, K)

        N, K = H_cl.shape
        r    = np.zeros((N, K))
        for i in range(N):
            h       = H_cl[i] - H_cl[i].min()
            h_range = h.max()
            h_norm  = h / h_range if h_range > 1e-10 else h
            r[i]    = self._pennylane_diagonal(h_norm, beta_t, s_t)
        return r

    def _print_fit_header(self):
        print(f"\nStarting PennyLane QAVB — DMM-SVVS")
        print(f"  β0={self.beta0}, s0={self.s0}, "
              f"τ1={self.tau1}, τ2={self.tau2}, "
              f"prune_start={self.prune_start}")


# ─────────────────────────────────────────────────────────────────────────────
# Verification: PennyLane circuit matches classical expm
# ─────────────────────────────────────────────────────────────────────────────

def verify_pennylane_matches_classical(K=5, N_test=10, random_state=7):
    """
    Unit test: PennyLane circuit diagonal must match scipy.linalg.expm diagonal.
    Tests with random energy vectors and random (beta, s) pairs.
    """
    rng  = np.random.default_rng(random_state)
    H_qu = _cyclic_hamiltonian(K)

    # Build a minimal PennyLane QAVB shell (no data)
    m        = DMM_SVVS_PennyLaneQAVB.__new__(DMM_SVVS_PennyLaneQAVB)
    m.K      = K
    m.H_qu   = H_qu
    m._init_pennylane_device()

    max_diff = 0.0
    EPS      = 1e-10

    for _ in range(N_test):
        h_raw  = rng.standard_normal(K)
        beta_t = float(rng.uniform(0.5, 5.0))
        s_t    = float(rng.uniform(0.0, 1.0))

        h       = h_raw - h_raw.min()
        h_range = h.max()
        h_norm  = h / h_range if h_range > 1e-10 else h

        # Classical
        M_diag = np.clip(-beta_t * (1 - s_t) * h_norm, -500.0, 0.0)
        M      = np.diag(M_diag) - beta_t * s_t * H_qu
        rho    = expm(M)
        rho   /= np.trace(rho)
        r_cl   = np.real(np.diag(rho)).clip(EPS)
        r_cl  /= r_cl.sum()

        # PennyLane
        r_pl   = m._pennylane_diagonal(h_norm, beta_t, s_t)

        diff     = np.max(np.abs(r_cl - r_pl))
        max_diff = max(max_diff, diff)

    print(f"\nPennyLane vs Classical verification (K={K}, {N_test} random tests)")
    print(f"  Max element-wise difference: {max_diff:.2e}")
    passed = max_diff < 1e-8
    print(f"  {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print("=" * 70)
    print("Smoke-test: DMM-SVVS Standard VI (v2) vs DAVB vs QAVB variants")
    print("=" * 70)

    # ── Synthetic DMM data (block-structured, moderate difficulty) ────────
    rng     = np.random.default_rng(42)
    N, S, K_true = 150, 200, 3
    block   = S // K_true

    alpha = np.full((K_true, S), 0.1)
    for k in range(K_true):
        alpha[k, k * block:(k + 1) * block] = 3.0

    true_labels = rng.choice(K_true, size=N)
    X = np.array([
        rng.multinomial(5000, rng.dirichlet(alpha[true_labels[i]]))
        for i in range(N)
    ], dtype=float)

    print(f"\nDataset: N={N}, S={S}, K_true={K_true}\n")

    results = {}

    # ── 1. Standard VI (v2) ───────────────────────────────────────────────
    print("--- Standard VI (v2) ---")
    from DMM_SVVS_Variational_v2 import DMM_SVVS_Variational_v2

    m = DMM_SVVS_Variational_v2(
        K_max=10, nu='auto', max_iter=300, verbose=1, random_state=42, selection_prior=0.3,
        prune_threshold=0.2,
    )
    m.fit(X)
    pred = m.predict(X)
    results["Standard VI (v2)"] = (
        adjusted_rand_score(true_labels, pred),
        normalized_mutual_info_score(true_labels, pred),
        m.K,
    )
    print(f"→ ARI={results['Standard VI (v2)'][0]:.3f}, "
          f"NMI={results['Standard VI (v2)'][1]:.3f}, K={m.K}\n")

    # ── 2. DAVB ───────────────────────────────────────────────────────────
    print("--- DAVB (Deterministic Annealing VB) ---")
    m = DMM_SVVS_DAVB(
        K_max=10, nu='auto', max_iter=400,
        beta0=0.01, tau2=150,
        verbose=1, random_state=42,
    )
    m.fit(X)
    pred = m.predict(X)
    results["DAVB"] = (
        adjusted_rand_score(true_labels, pred),
        normalized_mutual_info_score(true_labels, pred),
        m.K,
    )
    print(f"→ ARI={results['DAVB'][0]:.3f}, "
          f"NMI={results['DAVB'][1]:.3f}, K={m.K}\n")

    # ── 3. Classical QAVB ────────────────────────────────────────────────
    print("--- Classical QAVB (scipy expm + energy normalisation) ---")
    m = DMM_SVVS_ClassicalQAVB(
        K_max=10, nu='auto', max_iter=500,
        beta0=30.0, s0=1.0, tau1=100, tau2=200,
        verbose=1, random_state=42, selection_prior=0.3,
        prune_threshold=0.2,
    )
    m.fit(X)
    pred = m.predict(X)
    results["Classical QAVB"] = (
        adjusted_rand_score(true_labels, pred),
        normalized_mutual_info_score(true_labels, pred),
        m.K,
    )
    print(f"→ ARI={results['Classical QAVB'][0]:.3f}, "
          f"NMI={results['Classical QAVB'][1]:.3f}, K={m.K}\n")

    # ── 4. PennyLane QAVB ────────────────────────────────────────────────
    print("--- PennyLane QAVB (quantum circuit) ---")
    m = DMM_SVVS_PennyLaneQAVB(
        K_max=10, nu='auto', max_iter=100,
        beta0=30.0, s0=1.0, tau1=60, tau2=120,
        verbose=1, random_state=42, selection_prior=0.3,
        prune_threshold=0.2,
    )
    m.fit(X)
    pred = m.predict(X)
    results["PennyLane QAVB"] = (
        adjusted_rand_score(true_labels, pred),
        normalized_mutual_info_score(true_labels, pred),
        m.K,
    )
    print(f"→ ARI={results['PennyLane QAVB'][0]:.3f}, "
          f"NMI={results['PennyLane QAVB'][1]:.3f}, K={m.K}\n")

    # ── Summary table ────────────────────────────────────────────────────
    print("=" * 70)
    print(f"{'Method':<30} {'ARI':>8} {'NMI':>8} {'K':>5}")
    print("-" * 70)
    for name, (ari, nmi, k) in results.items():
        print(f"{name:<30} {ari:8.3f} {nmi:8.3f} {k:5d}")
    print("=" * 70)
    print(f"True K = {K_true}")

    # ── Verification ─────────────────────────────────────────────────────
    print("\n--- Verification: PennyLane circuit == Classical expm ---")
    verify_pennylane_matches_classical(K=5, N_test=15)
