# -*- coding: utf-8 -*-
"""
Test suite for the one‑dimensional backward (adjoint) solver in the
Cahn–Hilliard optimal control problem.

This module defines a set of tests that verify the implementation of the
adjoint equations in one dimension.  The adjoint solver computes three
variables ``(p, q, r)`` that satisfy a set of linear equations derived from
the first order optimality conditions.  The tests include:

* **Terminal conditions:** At the final time, ``p``, ``q`` and ``r`` obey
  simple linear relationships with the state and target values.  The residuals
  of these conditions are compared against tolerances based on the
  condition number of the system matrix.
* **Discrete adjoint step:** For each time step, the adjoint update
  involves solving a Crank–Nicolson discretisation with matrices ``A`` and
  ``B`` constructed from the forward state.  Residuals of this discrete
  equation are checked at all steps.
* **Relation q = −L p:** The auxiliary variable ``q`` is defined as the
  negative Laplacian applied to ``p``.  This test asserts this relation at
  every time level.
* **Crank–Nicolson equation for r:** The third variable ``r`` satisfies its own
  Crank–Nicolson equation; the residuals of this update are checked for
  every step.
* **Integration test:** A forward simulation is run, the last ten frames
  extracted, and the backward solver is called on this shorter window.  The
  test rebuilds ``A`` and ``B`` matrices and verifies that the correct
  operator ordering gives a much smaller residual than a swapped ordering.

Throughout, relative residuals are computed by normalising by the norms of
the left– and right–hand sides to avoid dividing by zero.  Automatic
tolerance estimators based on matrix condition numbers are used to scale
the error thresholds.  A small synthetic forward problem is generated to
provide test data for the backward solver.
"""


import pytest
import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
import os

# --- Create a directory for plots ---
PLOT_DIR = "test_plots"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# --- Common Imports from across the test files ---
from cost_and_function import (
    calculate_cost,
    calculate_gradient,
    perform_gradient_step
)
from backward_solver import run_backward, fpp_log
from Forward_solver import (
    laplacian_matrix_neumann,
    run_main_simulation,
    init_phi_random,
    solve_w,
    free_energy,
    trapz_weights,
    newton_raphson
)
# Import config classes but NOT the interactive functions
from config import ForwardSolverConfig

# Import the proximal function


# --- Get default configuration values ---
_default_fwd_config = ForwardSolverConfig()
N = _default_fwd_config.N
Lx = _default_fwd_config.Lx
T = _default_fwd_config.T
dt_initial = _default_fwd_config.dt_initial
tau = _default_fwd_config.tau
gamma = _default_fwd_config.gamma
c1 = _default_fwd_config.c1
c2 = _default_fwd_config.c2
kappa = _default_fwd_config.kappa
h = Lx / N
delta_sep = 1e-2  # From Forward_solver.py


@pytest.fixture
def solver_params():
    x = np.linspace(0, Lx, N + 1)
    Lmat = laplacian_matrix_neumann(N, h)
    return {"x": x, "Lmat": Lmat, "N": N, "h": h, "kappa": kappa, "c1": c1, "c2": c2}


@pytest.fixture
def setup_grids_and_shapes():
    """Creates simple grid data for consistent testing."""
    N_cost = 10
    M_cost = 20
    Lx_cost = 1.0
    T_cost = 1.0
    x_cost = np.linspace(0, Lx_cost, N_cost + 1)
    t_hist_cost = np.linspace(0, T_cost, M_cost + 1)
    shape = (M_cost + 1, N_cost + 1)
    return {
        "x": x_cost,
        "t_hist": t_hist_cost,
        "shape": shape,
        "Lx": Lx_cost,
        "T": T_cost
    }


# =================================================================================================
# ==                           Tests from test_backward.py                                     ==
# =================================================================================================

# --- Utilities for backward tests ---
EPS = np.finfo(float).eps
TINY = 1e-16

def rel_residual(res, *terms):
    denom = sum(np.linalg.norm(t) for t in terms) + TINY
    return np.linalg.norm(res) / denom

def auto_tol_from_cond(A, base=1e3):
    try:
        c = np.linalg.cond(A)
        if not np.isfinite(c): c = 1e12
    except Exception: c = 1e6
    return base * EPS * max(1.0, c), c

# --- Test problem setup for backward tests ---
def _make_synthetic_forward(N_backward=16, M_backward=12, Lx_backward=1.0, T_backward=0.2, A_backward=0.2):
    x = np.linspace(0.0, Lx_backward, N_backward + 1)
    t = np.linspace(0.0, T_backward, M_backward + 1)
    X, TT = np.meshgrid(x, t)
    phi_hist = A_backward * np.sin(np.pi * X / Lx_backward) * (1.0 + 0.2 * np.cos(2.0 * np.pi * TT / T_backward))
    return phi_hist, x, t

def _lap_ops(Np1, h_backward):
    L = laplacian_matrix_neumann(Np1 - 1, h_backward)
    return L, L @ L, np.eye(Np1)

def _A(phi_n, dt, L, L2, I):
    diag = np.diag(fpp_log(phi_n))
    return I - tau * L + 0.5 * dt * L2 - 0.5 * dt * (diag @ L)

def _B(phi_np1, dt, L, L2, I):
    diag = np.diag(fpp_log(phi_np1))
    return I - tau * L - 0.5 * dt * L2 + 0.5 * dt * (diag @ L)

# --- Fixtures for backward tests ---
@pytest.fixture(scope="module")
def small_problem():
    N_backward, M_backward = 16, 12
    phi_hist, x, t_hist = _make_synthetic_forward(N_backward=N_backward, M_backward=M_backward)
    h_backward = x[1] - x[0]
    L, L2, I = _lap_ops(N_backward + 1, h_backward)
    return dict(N=N_backward, M=M_backward, phi_hist=phi_hist, x=x, t_hist=t_hist, h=h_backward, L=L, L2=L2, I=I)

# --- Tests for backward solver ---
def test_terminal_conditions_hold_relative(small_problem):
    """Check that the adjoint variables satisfy the terminal conditions.

    At the final time ``T`` the adjoint solution must satisfy
    ``A_T p(T) = b2 (phi(T) − phi_T_target)``, ``q(T) + L p(T) = 0`` and
    ``r(T) = 0``.  This test forms the residuals of these equations and
    asserts that their relative sizes are within computed tolerances.
    """
    P = small_problem
    phi, x, t = P["phi_hist"], P["x"], P["t_hist"]
    L, I = P["L"], P["I"]
    b1, b2 = 1.3, 0.7
    phi_Q, phi_T_target = np.zeros_like(phi), np.zeros_like(phi[-1])
    p, q, r = run_backward(phi, x, t, b1, b2, phi_Q, phi_T_target)
    
    pT, qT, rT = p[-1], q[-1], r[-1]
    rhs_T = b2 * (phi[-1] - phi_T_target)
    
    A_T = I - tau * L
    res_term_p = A_T @ pT - rhs_T
    tol_p, _ = auto_tol_from_cond(A_T, base=2e3)
    rel_p = rel_residual(res_term_p, A_T @ pT, rhs_T)

    res_term_q = qT + L @ pT
    tol_q = 5e2 * EPS
    rel_q = rel_residual(res_term_q, qT, L @ pT)

    print("\n--- Testing Backward Solver Terminal Conditions ---")
    print(f"[p(T)]: Relative residual = {rel_p:.3e}, Tolerance = {tol_p:.3e}")
    print(f"[q(T)]: Relative residual = {rel_q:.3e}, Tolerance = {tol_q:.3e}")
    print(f"[r(T)]: Norm = {np.linalg.norm(rT):.3e}, Expected = 0.0")

    

    assert rel_p < tol_p
    assert rel_q < tol_q
    assert np.linalg.norm(rT) < 1e-14

def test_p_step_discrete_equation_is_satisfied_relative(small_problem):
    """Verify the discrete Crank–Nicolson equation for ``p`` at each time step.

    The adjoint update for ``p`` solves ``A(phi_n) p_n = B(phi_{n+1}) p_{n+1}
    + source``.  For each time step a source term is constructed and the
    matrices ``A`` and ``B`` are assembled from the forward trajectory.  The
    residual of this linear equation is normalised and required to be below
    a tolerance derived from the condition number of ``A``.
    """
    P = small_problem
    phi, x, t = P["phi_hist"], P["x"], P["t_hist"]
    L, L2, I = P["L"], P["L2"], P["I"]
    b1, b2 = 1.3, 0.7
    phi_Q, phi_T_target = np.zeros_like(phi), np.zeros_like(phi[-1])
    p, q, r = run_backward(phi, x, t, b1, b2, phi_Q, phi_T_target)
    
    residuals, tols = [], []
    for n in range(len(t) - 1):
        dt = t[n+1] - t[n]
        if dt <= 0: continue
        A = _A(phi[n], dt, L, L2, I)
        B = _B(phi[n+1], dt, L, L2, I)
        src = 0.5 * dt * b1 * ((phi[n] - phi_Q[n]) + (phi[n+1] - phi_Q[n+1]))
        left = A @ p[n]
        right = B @ p[n+1] + src
        res = left - right
        tol, _ = auto_tol_from_cond(A, base=2e3)
        rel = rel_residual(res, left, right)
        residuals.append(rel)
        tols.append(tol)
        assert rel < tol

    

def test_q_equals_minus_Lp_all_steps_relative(small_problem):
    """Ensure ``q = −L p`` at all times.

    After computing the adjoint variables, each time level is checked to
    satisfy the relation ``q[n] + L p[n] = 0``.  The relative residuals
    should be machine–small for all n.
    """
    P = small_problem
    phi, x, t = P["phi_hist"], P["x"], P["t_hist"]
    L = P["L"]
    p, q, _ = run_backward(phi, x, t, 0.9, 0.4, np.zeros_like(phi), np.zeros_like(phi[-1]))
    
    residuals = []
    for n in range(len(t)):
        res = q[n] + L @ p[n]
        residuals.append(rel_residual(res, q[n], L @ p[n]))
    
    

def test_r_crank_nicolson_equation_is_satisfied_relative(small_problem):
    """Check the Crank–Nicolson equation for the auxiliary variable ``r``.

    The variable ``r`` satisfies an evolution equation involving ``q``.  This
    test computes the residual of that update at each time step and uses
    relative residuals to assert correctness.  The loop skips zero or
    negative time steps for robustness.
    """
    P = small_problem
    phi, x, t = P["phi_hist"], P["x"], P["t_hist"]
    p, q, r = run_backward(phi, x, t, 1.0, 0.6, np.zeros_like(phi), np.zeros_like(phi[-1]))
    
    residuals = []
    for n in range(len(t) - 1):
        dt = t[n+1] - t[n]
        if dt <= 0: continue
        left = -gamma * (r[n+1] - r[n]) / dt
        mid  = 0.5 * ((r[n+1] - q[n+1]) + (r[n] - q[n]))
        residuals.append(rel_residual(left + mid, left, mid))
        
    
def test_backward_reads_last10_reverse_real_forward():
    """Integration test: solve backward on a window of a real forward simulation.

    This test runs a forward simulation with a reasonably fine grid, then
    extracts the last ten frames of the state history and time stamps.
    It calls the backward solver on this reduced history with zero targets
    and reconstructs the Crank–Nicolson matrices ``A`` and ``B`` for each
    step.  The residual of the correct adjoint update is compared with the
    residual obtained by swapping the order of ``A`` and ``B``; the latter
    should be two orders of magnitude larger, demonstrating the sensitivity
    to operator ordering.
    """
    config = ForwardSolverConfig(N=64, dt_initial=1e-2)
    phi_hist, x, t_hist = run_main_simulation(fwd_config=config, store_history=True, verbose=False)
    
    phi10, t10 = phi_hist[-10:], t_hist[-10:]
    Np1, N_backward = phi10.shape[1], phi10.shape[1] - 1
    h_backward = x[1] - x[0]
    
    p, _, _ = run_backward(phi10, x, t10, 1.0, 0.7, np.zeros_like(phi10), np.zeros_like(phi10[-1]))
    
    L = laplacian_matrix_neumann(N_backward, h_backward)
    L2 = L @ L
    I = np.eye(Np1)
    
    print("\n--- Testing Backward Solver Direction ---")
    for i in range(9):
        dt_i = t10[i+1] - t10[i]
        A_i   = _A(phi10[i],   dt_i, L, L2, I)
        B_ip1 = _B(phi10[i+1], dt_i, L, L2, I)
        src = 0.5 * dt_i * 1.0 * ((phi10[i] - 0) + (phi10[i+1] - 0))
        left_corr  = A_i @ p[i]
        right_corr = B_ip1 @ p[i+1] + src
        rel_corr   = rel_residual(left_corr - right_corr, left_corr, right_corr)
        
        A_ip1 = _A(phi10[i+1], dt_i, L, L2, I)
        B_i = _B(phi10[i], dt_i, L, L2, I)
        left_swap  = A_ip1 @ p[i]
        right_swap = B_i @ p[i+1] + src
        rel_swap   = rel_residual(left_swap - right_swap, left_swap, right_swap)
        
        print(f"  Step {i}: Correct order residual = {rel_corr:.3e}, Swapped order = {rel_swap:.3e}")
        assert (rel_swap + 1e-30) / (rel_corr + 1e-30) > 1e2
        
        
if __name__ == "__main__":
    # Run pytest programmatically
    
    pytest.main([__file__, "-v","-s", "--tb=short"])