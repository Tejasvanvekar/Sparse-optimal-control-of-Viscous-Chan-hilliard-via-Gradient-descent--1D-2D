# -*- coding: utf-8 -*-
"""
Combined test suite for one‑dimensional proximal (ISTA) and gradient update
functions.

This script amalgamates tests from several smaller files (``test_cost.py``,
``test_backward.py``, ``test_proximal_elliptic.py``, and
``Test_Forward_solver.py``) into a single module that requires no user
interaction.  It focuses on verifying the proximal operator used in the
iterative shrinkage/thresholding algorithm (ISTA) for the ℓ¹ sparsity term
and the simple gradient descent update preceding the proximal step.  All
solver parameters are taken from default ``ForwardSolverConfig`` objects.

The tests cover:

* **Closed form proximal step:** For ``alpha = 1`` and no box bounds, one
  iteration of ISTA must coincide with the soft thresholding operator.
* **Soft threshold then clip:** When box constraints are applied, the
  proximal update reduces to soft thresholding followed by projection onto
  the interval ``[lbound, ubound]``.
* **Monotonic objective decrease:** Repeated ISTA steps with a step size
  ``alpha <= 1/L`` should yield a nonincreasing objective ``0.5||u − y||²
  + kappa * ||u||₁``.
* **Fixed point characterisation:** A point obtained by soft thresholding
  ``y`` is a fixed point of the ISTA map; applying the proximal update
  should return the same point.

Fixtures provide small grids and shapes for testing.  Plotting is disabled
to make the tests suitable for automated execution.  The default forward
configuration is used solely for obtaining parameter values; the forward
solver is not exercised directly in this file.
"""

import pytest
import numpy as np

import os

# --- Create a directory for plots ---
PLOT_DIR = "test_plots"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# --- Common Imports from across the test files ---
from cost_and_function import perform_gradient_step

from Forward_solver import  laplacian_matrix_neumann
# Import config classes but NOT the interactive functions
from config import ForwardSolverConfig
# Import the proximal function
from GD_1D import perform_proximal_and_projection

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
# ==                      Tests from test_proximal_elliptic.py                                 ==
# =================================================================================================
def soft_threshold(v, lam):
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)

def obj_l2_l1(u, y, kappa_prox):
    return 0.5 * np.sum((u - y) ** 2) + kappa_prox * np.sum(np.abs(u))

@pytest.mark.parametrize("shape", [(80,), (20, 30)])
def test_ista_one_step_equals_closed_form(shape):
    """Check that one ISTA step equals soft thresholding in the unconstrained case.

    For a quadratic objective ``0.5||u − y||²`` and an ℓ¹ penalty
    ``kappa_test * ||u||₁`` with step size ``alpha = 1``, the ISTA update
    ``u_next`` starting from a random ``u`` reduces to ``soft_threshold(y, kappa_test)``.
    This test verifies that the proximal implementation matches the closed
    form solution for arrays of various shapes.
    """
    rng = np.random.default_rng(7)
    y = rng.normal(0.0, 0.7, size=shape)
    kappa_test = 0.12
    alpha = 1.0

    u = rng.normal(0.0, 0.5, size=shape)
    grad = u - y
    u_temp = perform_gradient_step(u, grad, alpha)

    BIG = 1e12
    u_next = perform_proximal_and_projection(u_temp, alpha, kappa_test, -BIG, BIG)
    u_star = soft_threshold(y, kappa_test)
    assert np.allclose(u_next, u_star, atol=1e-12, rtol=0.0)

@pytest.mark.parametrize("shape,lbound,ubound", [((100,), -0.6, 0.8), ((15, 40), -0.5, 0.5)])
def test_ista_one_step_with_box_equals_soft_then_clip(shape, lbound, ubound):
    """Verify that one ISTA step with bounds equals soft thresholding then clipping.

    When box constraints ``[lbound, ubound]`` are enforced, the proximal
    update starting from ``u = 0`` with ``alpha = 1`` and ``grad = −y``
    should return ``clip(soft_threshold(y, kappa_test), lbound, ubound)``.
    This test checks that property for different shapes and bound values.
    """
    rng = np.random.default_rng(11)
    y = rng.normal(0.0, 0.9, size=shape)
    kappa_test = 0.2
    alpha = 1.0

    u = np.zeros(shape)
    grad = u - y
    u_temp = perform_gradient_step(u, grad, alpha)
    u_next = perform_proximal_and_projection(u_temp, alpha, kappa_test, lbound, ubound)

    u_star = np.clip(soft_threshold(y, kappa_test), lbound, ubound)
    assert np.allclose(u_next, u_star, atol=1e-12, rtol=0.0)

def test_ista_objective_monotone_nonincreasing():
    """Ensure that the ISTA objective does not increase over iterations.

    The objective ``0.5||u − y||² + kappa_prox * ||u||₁`` should decrease
    (up to tiny numerical noise) when applying the ISTA update with
    ``alpha <= 1/L`` for the Lipschitz constant ``L=1``.  Starting from
    ``u = 0``, the test performs 150 iterations and records the objective
    values; differences are checked to be nonpositive within a small
    tolerance.  Some intermediate values are printed for debugging.
    """
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 1.0, size=(120,))
    kappa_prox = 0.1
    alpha = 0.9
    u = np.zeros_like(y)
    vals = []
    print("\n--- Testing ISTA Objective Decrease ---")
    for i in range(150):
        cost = obj_l2_l1(u, y, kappa_prox)
        vals.append(cost)
        if i % 10 == 0:
            print(f"Iteration {i}: Objective = {cost:.6f}")
        grad = u - y
        u_temp = perform_gradient_step(u, grad, alpha)
        u = perform_proximal_and_projection(u_temp, alpha, kappa_prox, -1e12, 1e12)

    diffs = np.diff(vals)
    assert np.all(diffs <= 1e-12 + 1e-12*np.abs(vals[:-1]))
    
    

@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0])
def test_fixed_point_characterization(alpha):
    """Check the fixed point property of the ISTA mapping.

    For ``u_star = soft_threshold(y, kappa_test)`` the ISTA mapping
    satisfies ``u_star = prox_{alpha kappa ||·||₁}(u_star − alpha(u_star − y))``.
    This test constructs ``u_star`` from a random ``y`` and verifies that
    applying the proximal update returns the same vector, confirming the
    characterisation of fixed points of the ISTA iteration.
    """
    rng = np.random.default_rng(21)
    y = rng.normal(0.0, 0.8, size=(60,))
    kappa_test = 0.15

    u_star = soft_threshold(y, kappa_test)
    u_temp = u_star - alpha * (u_star - y)
    u_fp = perform_proximal_and_projection(u_temp, alpha, kappa_test, -1e12, 1e12)

    assert np.allclose(u_fp, u_star, atol=1e-12, rtol=0.0)






if __name__ == "__main__":
    # Run pytest programmatically
    
    pytest.main([__file__, "-v","-s", "--tb=short"])