# -*- coding: utf-8 -*-
"""
Unit tests for the one‑dimensional cost and gradient functions in the
Cahn–Hilliard optimal control problem.

This module exercises the functions defined in ``cost_and_function``
which assemble the objective functional and its gradient for the 1‑D case.
The cost functional comprises several terms weighted by scalar parameters:

1. **J1 (tracking term):** Penalises the difference between the state
   trajectory ``phi_hist`` and a desired target ``phi_Q_target`` over
   space and time.
2. **J2 (terminal term):** Penalises the difference between the final state
   ``phi_hist[-1]`` and a terminal target ``phi_T_target``.
3. **J3 (energy term):** Proportional to the squared control ``u``.
4. **J4 (sparsity term):** Proportional to the ℓ¹ norm of the control.

The tests below isolate each of these contributions by choosing the inputs
appropriately.  Additional tests verify the simple gradient computation and
the basic gradient descent update used prior to applying the proximal
operator.  A fixture constructs small grids and time histories to keep the
computations inexpensive.
"""

import pytest
import numpy as np

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

from Forward_solver import     laplacian_matrix_neumann
    
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






def test_calculate_gradient():
    """Check that the gradient of the smooth part of the cost is computed correctly.

    For the 1‑D problem the gradient of the smooth control energy term J₃
    with respect to ``u`` is ``r + b3 * u``.  This test builds small
    arrays for the residual ``r`` and control ``u``, chooses a value of
    ``b3``, and compares the returned gradient against the expected sum.
    """
    """Verifies that grad_smooth = r + b3 * u."""
    r = np.array([[1.0, 2.0], [3.0, 4.0]])
    u = np.array([[0.5, 0.1], [-0.2, 0.0]])
    b3 = 10.0
    
    expected_gradient = r + b3 * u
    actual_gradient = calculate_gradient(r, u, b3)
    
    print("\n--- Testing Gradient Calculation ---")
    print("Expected Gradient:\n", expected_gradient)
    print("Actual Gradient:\n", actual_gradient)
    
    
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-7)

def test_perform_gradient_step():
    """Ensure that a gradient descent step is implemented properly.

    The function ``perform_gradient_step`` returns ``u_current − alpha * grad``.
    This test constructs a small control array and gradient and checks that
    the returned array matches the expected one for a chosen step size ``alpha``.
    """
    """Verifies that u_temp = u_current - alpha * grad."""
    u_current = np.array([[1.0, 1.0], [1.0, 1.0]])
    grad_smooth = np.array([[0.2, -0.1], [0.5, 0.0]])
    alpha = 5.0
    
    expected_u_temp = u_current - alpha * grad_smooth
    actual_u_temp = perform_gradient_step(u_current, grad_smooth, alpha)
    
    print("\n--- Testing Gradient Step ---")
    print("Expected u_temp:\n", expected_u_temp)
    print("Actual u_temp:\n", actual_u_temp)
    
    
    np.testing.assert_allclose(actual_u_temp, expected_u_temp, rtol=1e-7)

# --- Tests for Cost Function ---

def test_calculate_cost_all_zero(setup_grids_and_shapes):
    """Sanity check: zero inputs should yield zero cost.

    When all input arrays ``phi_hist``, ``u``, ``phi_Q_target`` and
    ``phi_T_target`` are zero, the cost functional must return zero
    irrespective of the parameter values.  This test verifies that.
    """
    """If all inputs are zero, the cost must be zero."""
    data = setup_grids_and_shapes
    phi_hist = np.zeros(data["shape"])
    u = np.zeros(data["shape"])
    phi_Q_target = np.zeros(data["shape"])
    phi_T_target = np.zeros(data["x"].shape)

    cost = calculate_cost(
        phi_hist, u, phi_Q_target, phi_T_target,
        data["x"], data["t_hist"],
        b1=1.0, b2=1.0, b3=1.0, kappa=1.0, verbose=False
    )
    
    print("\n--- Testing Cost (All Zero) ---")
    print(f"Expected Cost: 0.0, Actual Cost: {cost}")
    assert cost == 0.0

def test_calculate_cost_tracking_only(setup_grids_and_shapes):
    """Isolate the J1 tracking term.

    A constant state ``phi_hist`` with value 2.0 and zero control and targets
    yields a cost ``0.5 * b1 * ∫ phi_hist² dx dt``.  The test computes this
    expected value analytically on a small grid and asserts that
    ``calculate_cost`` matches it.
    """
    """Tests the J1 (space-time tracking) term in isolation."""
    data = setup_grids_and_shapes
    phi_hist = np.full(data["shape"], 2.0) 
    u = np.zeros(data["shape"])
    phi_Q_target = np.zeros(data["shape"])
    phi_T_target = np.zeros(data["x"].shape)
    
    b1 = 0.5
    expected_cost = 0.25 * (2.0**2) * data["Lx"] * data["T"]

    cost = calculate_cost(
        phi_hist, u, phi_Q_target, phi_T_target,
        data["x"], data["t_hist"],
        b1=b1, b2=0.0, b3=0, kappa=0.0, verbose=False
    )
    
    print("\n--- Testing Cost (Tracking Only) ---")
    print(f"Expected Cost: {expected_cost}, Actual Cost: {cost}")
    
   
    assert np.isclose(cost, expected_cost)

def test_calculate_cost_terminal_only(setup_grids_and_shapes):
    """Isolate the J2 terminal term.

    When the state is zero except at the final time where it equals a
    constant 3.0, the cost reduces to ``0.5 * b2 * ∫ phi_T² dx``.  The test
    constructs such a state and compares the computed cost against the
    analytic expression ``0.5 * b2 * 3² * Lx``.
    """
    """Tests the J2 (terminal tracking) term in isolation."""
    data = setup_grids_and_shapes
    phi_hist = np.zeros(data["shape"])
    phi_hist[-1, :] = 3.0
    u = np.zeros(data["shape"])
    phi_Q_target = np.zeros(data["shape"])
    phi_T_target = np.zeros(data["x"].shape)
    
    b2 = 0.8
    expected_cost = 0.4 * (3.0**2) * data["Lx"]

    cost = calculate_cost(
        phi_hist, u, phi_Q_target, phi_T_target,
        data["x"], data["t_hist"],
        b1=0.0, b2=b2, b3=0.0, kappa=0.0, verbose=False
    )
    
    print("\n--- Testing Cost (Terminal Only) ---")
    print(f"Expected Cost: {expected_cost}, Actual Cost: {cost}")
    
    
    
    assert np.isclose(cost, expected_cost)

def test_calculate_cost_energy_only(setup_grids_and_shapes):
    """Isolate the J3 control energy term.

    With zero state and constant control ``u``, the cost is
    ``0.5 * b3 * ∫ u² dx dt``.  A constant control of value 2.0 on a small
    grid is used to compute the expected cost, which is then compared
    against the value returned by ``calculate_cost``.
    """
    """Tests the J3 (control energy) term in isolation."""
    data = setup_grids_and_shapes
    phi_hist = np.zeros(data["shape"])
    u = np.full(data["shape"], 2.0)
    phi_Q_target = np.zeros(data["shape"])
    phi_T_target = np.zeros(data["x"].shape)

    b3 = 0.1
    expected_cost = 0.05 * (2.0**2) * data["Lx"] * data["T"]
    
    cost = calculate_cost(
        phi_hist, u, phi_Q_target, phi_T_target,
        data["x"], data["t_hist"],
        b1=0.0, b2=0.0, b3=b3, kappa=0.0, verbose=False
    )

    print("\n--- Testing Cost (Energy Only) ---")
    print(f"Expected Cost: {expected_cost}, Actual Cost: {cost}")
    assert np.isclose(cost, expected_cost)

def test_calculate_cost_sparsity_only(setup_grids_and_shapes):
    """Isolate the J4 ℓ¹ sparsity term.

    When the state and targets are zero and the control ``u`` is a constant
    negative value, the cost equals ``kappa * ∫ |u| dx dt``.  This test
    computes that integral on a small grid and checks that
    ``calculate_cost`` returns the same number.
    """
    """Tests the J4 (L1/sparsity) term in isolation."""
    data = setup_grids_and_shapes
    phi_hist = np.zeros(data["shape"])
    u = np.full(data["shape"], -3.0)
    phi_Q_target = np.zeros(data["shape"])
    phi_T_target = np.zeros(data["x"].shape)

    kappa_cost = 0.01
    expected_cost = kappa_cost * np.abs(-3.0) * data["Lx"] * data["T"]
    
    cost = calculate_cost(
        phi_hist, u, phi_Q_target, phi_T_target,
        data["x"], data["t_hist"],
        b1=0.0, b2=0.0, b3=0.0, kappa=kappa_cost, verbose=False
    )
    
    print("\n--- Testing Cost (Sparsity Only) ---")
    print(f"Expected Cost: {expected_cost}, Actual Cost: {cost}")
    assert np.isclose(cost, expected_cost)
    
    
if __name__ == "__main__":
    # Run pytest programmatically
    
    pytest.main([__file__, "-v","-s", "--tb=short"])