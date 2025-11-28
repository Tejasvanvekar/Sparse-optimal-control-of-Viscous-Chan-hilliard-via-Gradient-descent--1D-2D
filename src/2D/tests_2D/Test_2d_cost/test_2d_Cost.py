# -*- coding: utf-8 -*-
"""
Test suite for verifying cost functional components in the 2‑D Cahn–Hilliard
optimal control framework.

This module exercises the routines in ``cost2_and_function`` that assemble
and differentiate the objective functional for the two–dimensional solver.
The cost functional includes four contributions:

1. **Space–time tracking term (J1)** which penalises deviations of the state
   ``phi`` from a desired target ``phi_Q`` across the full simulation horizon.
2. **Terminal tracking term (J2)** that measures how close the final state
   ``phi(T)`` is to a prescribed terminal target ``phi_T``.
3. **Control energy term (J3)** proportional to the squared control effort
   ``u``.
4. **L1 sparsity term (J4)** that promotes sparse (piecewise constant) control
   through an ℓ¹ penalty with coefficient ``kappa_sparsity``.

The tests in this file are designed to isolate each contribution and ensure
that the implemented functions ``calculate_cost``, ``calculate_gradient`` and
``proximal_step`` behave as expected.  A fixture constructs small grids and
histories for ease of integration.  Additional tests check the behaviour of
the Neumann Laplacian and the gradient of the free energy to ensure that
important mathematical properties hold under the chosen discretisation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Project functions
from cost2_and_function import (
    calculate_cost,
    calculate_gradient,
    proximal_step,
)

from Forward2_solver import (
    laplacian_matrix_neumann,
    apply_laplacian,
    
    regularized_log,
    
    initialize_mu,
    
)
# Configured input (like 1-D tests)
from config import ForwardSolverConfig, OptimizationConfig
#import io, re, numpy as np
#from contextlib import redirect_stdout, redirect_stder

# -------- helpers (same as 1-D style) --------
EPS = np.finfo(float).eps
TINY = 1e-16
DELTA_SEP = 1e-2

def rel_residual(res, *terms):
    denom = sum(np.linalg.norm(t) for t in terms) + TINY
    return np.linalg.norm(res) / denom

def auto_tol_from_cond(A, base=1e3):
    """Estimate tol from cond(A). Safe for small test matrices (dense if needed)."""
    try:
        c = np.linalg.cond(A.toarray())
        if not np.isfinite(c):
            c = 1e12
    except Exception:
        c = 1e6
    return base * EPS * max(1.0, c), c

def auto_tol_scalar(base=1e3):
    return base * EPS

# ========= fixtures =========

@pytest.fixture(scope="module")
def cfg():
    """Default 2-D forward config (like 1-D tests use ForwardSolverConfig defaults)."""
    return ForwardSolverConfig()

@pytest.fixture
def solver_params_2d(cfg: ForwardSolverConfig):
    """Provides grid & operators for 2-D tests, derived from config (no globals)."""
    Nx, Ny = cfg.Nx, cfg.Ny
    Lx, Ly = cfg.Lx, cfg.Ly
    hx, hy = Lx / Nx, Ly / Ny

    x = np.linspace(0.0, Lx, Nx + 1)
    y = np.linspace(0.0, Ly, Ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Lmat = laplacian_matrix_neumann(Nx, Ny, hx, hy)

    return {
        "Nx": Nx, "Ny": Ny,
        "Lx": Lx, "Ly": Ly,
        "hx": hx, "hy": hy,
        "x": x, "y": y, "X": X, "Y": Y, "Lmat": Lmat,
        "kappa": cfg.kappa, "c1": cfg.c1, "c2": cfg.c2,
        "tau": cfg.tau, "gamma": cfg.gamma,
    }

@pytest.fixture
def grid_setup():
    """Small synthetic grids for cost-term isolation tests (kept minimal)."""
    Nx_cost, Ny_cost, M_cost = 4, 3, 5
    Lx_cost, Ly_cost, T_cost = 1.0, 1.0, 1.0
    x = np.linspace(0.0, Lx_cost, Nx_cost + 1)
    y = np.linspace(0.0, Ly_cost, Ny_cost + 1)
    t_hist = np.linspace(0.0, T_cost, M_cost + 1)
    shape = (M_cost + 1, Nx_cost + 1, Ny_cost + 1)
    return dict(Nx=Nx_cost, Ny=Ny_cost, M=M_cost,
                Lx=Lx_cost, Ly=Ly_cost, T=T_cost,
                x=x, y=y, t_hist=t_hist, shape=shape)



#Cost & function


def test_neumann_bc_constant_nullspace_2d(solver_params_2d):
    """Confirm that the Neumann Laplacian annihilates constant fields.

    For Neumann boundary conditions, the Laplacian operator has a nullspace
    spanned by constant functions.  This test applies the discrete Laplacian
    to an array of ones and asserts that the result is (numerically) zero at
    all interior points.
    """
    """Neumann Laplacian must annihilate a constant field (including boundaries)."""
    p = solver_params_2d
    L = p["Lmat"]
    Nx, Ny = p["Nx"], p["Ny"]
    const = np.ones((Nx + 1, Ny + 1))
    out = apply_laplacian(L, const, Nx, Ny)
    assert_allclose(out, 0.0, atol=1e-12, err_msg="Δ(1) ≠ 0 under Neumann BCs in 2D.")


def test_energy_gradient_consistency_2d(solver_params_2d, cfg: ForwardSolverConfig):
    """Check that the computed chemical potential matches the variational formula.

    The gradient of the free energy with respect to ``phi`` is given by
    ``mu = −kappa Δ phi + c1 log((1+phi)/(1−phi)) − 2 c2 phi − w``.
    This test constructs random fields ``phi`` and ``w`` strictly inside the
    admissible interval and compares the implementation of ``initialize_mu``
    against this analytic expression to machine precision.
    """
    """Check μ equals δE/δφ = -κΔφ + c1*log((1+φ)/(1-φ)) - 2c2 φ - w."""
    p = solver_params_2d
    Nx, Ny = p["Nx"], p["Ny"]
    L = p["Lmat"]
    hx, hy = p["hx"], p["hy"]
    kappa, c1, c2 = p["kappa"], p["c1"], p["c2"]
    delta = 1e-2
    rng = np.random.default_rng(123)
    # Random φ strictly inside admissible interval and random w
    phi = 0.6 * (rng.random((Nx + 1, Ny + 1)) - 0.5)
    w = 0.1 * (rng.random((Nx + 1, Ny + 1)) - 0.5)
    # μ from implementation
    mu_impl = initialize_mu(phi, w, c1, c2, kappa, L, Nx, Ny, delta)
    # μ from variational formula
    lap_phi = apply_laplacian(L, phi, Nx, Ny)
    mu_var = -kappa * lap_phi + c1 * regularized_log(phi, delta) - 2.0 * c2 * phi - w
    err = np.linalg.norm(mu_impl - mu_var) / (np.linalg.norm(mu_var) + 1e-30)
    assert err < 1e-12, f"Energy gradient mismatch in 2D (rel err={err:.2e})"






def test_calculate_gradient_2d():
    """Test the gradient assembly for the smooth part of the cost.

    Given residual ``r`` and control ``u``, the gradient of the smooth part
    J₃ of the objective with respect to the control is ``r + b3 * u``.
    This test builds small arrays and an ``OptimizationConfig`` with the
    desired parameter ``b3`` and asserts exact agreement between the
    implementation and the expected expression.
    """
    r = np.array([[[1.0, -0.5],[2.0, 0.1]], [[0.0, 0.0],[0.3, -1.0]]])
    u = np.array([[[ -0.2,  0.4],[0.0, 0.1]], [[1.0, -0.5],[0.2,  0.2]]])
    b3 = 5.5
    # In the new API, calculate_gradient takes an OptimizationConfig object.
    # Construct a minimal config with the desired b3 value. Other parameters are unused here.
    opt_cfg = OptimizationConfig(b1=0.0, b2=0.0, b3=b3, kappa_sparsity=0.0)
    expected = r + b3 * u
    actual = calculate_gradient(r, u, opt_cfg)
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

def test_perform_gradient_step_2d():
    """Verify that a gradient descent step is implemented by the proximal operator.

    When the ℓ¹ penalty (sparsity) is disabled by setting
    ``kappa_sparsity = 0``, the proximal update reduces to a plain gradient
    descent step followed by clipping into the admissible bounds ``[u_min, u_max]``.
    This test constructs a control field and gradient and checks that the
    update computed by ``proximal_step`` matches ``clip(u − α * grad)`` exactly.
    """
    u_current = np.array([[[0.5,-0.5],[0.2,0.2]],[[-1.0,1.0],[0.0,0.0]]])
    grad      = np.array([[[0.1,-0.1],[0.5,0.0]],[[-0.2,0.2],[1.0,-1.0]]])
    alpha = 2.0
    # The proximal_step now uses an OptimizationConfig and performs projection into [u_min, u_max].
    # Use kappa_sparsity=0.0 to avoid soft-thresholding so the update reduces to a gradient step followed by clipping.
    opt_cfg = OptimizationConfig(b1=0.0, b2=0.0, b3=0.0, kappa_sparsity=0.0)
    expected = np.clip(u_current - alpha * grad, opt_cfg.u_min, opt_cfg.u_max)
    actual = proximal_step(u_current, grad, alpha, opt_cfg)
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

def test_calculate_cost_all_zero_2d(grid_setup):
    """Cost should vanish when all inputs are zero.

    The objective functional vanishes if the state, control and targets are
    identically zero.  This sanity check ensures that no stray constants
    pollute the cost computation.  All cost terms have unit coefficients here
    although their values do not matter when the fields are zero.
    """
    d = grid_setup
    phi_hist = np.zeros(d["shape"])
    u = np.zeros(d["shape"])
    phi_Q = np.zeros_like(phi_hist)
    phi_T = np.zeros((d["Nx"] + 1, d["Ny"] + 1))
    # Build optimization configuration matching the b1,b2,b3,kappa values used in the test.
    opt_cfg = OptimizationConfig(b1=1.0, b2=1.0, b3=1.0, kappa_sparsity=1.0)
    cost = calculate_cost(phi_hist, u, phi_Q, phi_T, d["x"], d["y"], d["t_hist"], opt_cfg)
    assert cost == pytest.approx(0.0)

def test_calculate_cost_tracking_only_2d(grid_setup):
    """Isolate the space–time tracking term ``J1``.

    A nonzero constant state ``phi_hist`` with zero control and zero targets
    results in a cost of ``0.5 * b1 * ∫∫ phi_hist² dx dy dt``.  The test
    constructs such a scenario on a small grid and compares against the
    analytic formula ``0.5 * b1 * (value)² * Lx * Ly * T``.
    """
    d = grid_setup
    phi_hist = np.full(d["shape"], 2.0)
    u = np.zeros(d["shape"])
    phi_Q = np.zeros_like(phi_hist)
    phi_T = np.zeros((d["Nx"] + 1, d["Ny"] + 1))
    b1 = 0.5
    expected = 0.5 * b1 * (2.0**2) * d["Lx"] * d["Ly"] * d["T"]
    opt_cfg = OptimizationConfig(b1=b1, b2=0.0, b3=0.0, kappa_sparsity=0.0)
    cost = calculate_cost(phi_hist, u, phi_Q, phi_T, d["x"], d["y"], d["t_hist"], opt_cfg)
    assert np.isclose(cost, expected, rtol=1e-10)

def test_calculate_cost_terminal_only_2d(grid_setup):
    """Isolate the terminal tracking term ``J2``.

    When the state is zero except at the final time step, the cost reduces
    to ``0.5 * b2 * ∫ phi_T² dx dy`` with ``phi_T`` being a constant.  The
    test verifies that ``calculate_cost`` produces the expected value.
    """
    d = grid_setup
    phi_hist = np.zeros(d["shape"])
    phi_hist[-1,:,:] = 3.0
    u = np.zeros(d["shape"])
    phi_Q = np.zeros_like(phi_hist)
    phi_T = np.zeros((d["Nx"] + 1, d["Ny"] + 1))
    b2 = 0.8
    expected = 0.5 * b2 * (3.0**2) * d["Lx"] * d["Ly"]
    opt_cfg = OptimizationConfig(b1=0.0, b2=b2, b3=0.0, kappa_sparsity=0.0)
    cost = calculate_cost(phi_hist, u, phi_Q, phi_T, d["x"], d["y"], d["t_hist"], opt_cfg)
    assert np.isclose(cost, expected, rtol=1e-10)

def test_calculate_cost_energy_only_2d(grid_setup):
    """Isolate the control energy term ``J3``.

    A constant control field ``u`` over time and space with zero state yields
    a cost ``0.5 * b3 * ∫∫ u² dx dy dt``.  This test validates that the
    implementation computes this term correctly when the other cost weights
    are set to zero.
    """
    d = grid_setup
    phi_hist = np.zeros(d["shape"])
    u = np.full(d["shape"], 2.0)
    phi_Q = np.zeros_like(phi_hist)
    phi_T = np.zeros((d["Nx"] + 1, d["Ny"] + 1))
    b3 = 0.1
    expected = 0.5 * b3 * (2.0**2) * d["Lx"] * d["Ly"] * d["T"]
    opt_cfg = OptimizationConfig(b1=0.0, b2=0.0, b3=b3, kappa_sparsity=0.0)
    cost = calculate_cost(phi_hist, u, phi_Q, phi_T, d["x"], d["y"], d["t_hist"], opt_cfg)
    assert np.isclose(cost, expected, rtol=1e-10)

def test_calculate_cost_sparsity_only_2d(grid_setup):
    """Isolate the ℓ¹ sparsity term ``J4``.

    With zero state and a constant negative control value, the cost due to
    sparsity is simply ``kappa_sparsity * ∫∫ |u| dx dy dt``.  This test
    constructs a constant control field and checks that the computed cost
    matches this integral exactly.
    """
    d = grid_setup
    phi_hist = np.zeros(d["shape"])
    u = np.full(d["shape"], -3.0)
    phi_Q = np.zeros_like(phi_hist)
    phi_T = np.zeros((d["Nx"] + 1, d["Ny"] + 1))
    kappa_val = 0.01
    expected = kappa_val * 3.0 * d["Lx"] * d["Ly"] * d["T"]
    opt_cfg = OptimizationConfig(b1=0.0, b2=0.0, b3=0.0, kappa_sparsity=kappa_val)
    cost = calculate_cost(phi_hist, u, phi_Q, phi_T, d["x"], d["y"], d["t_hist"], opt_cfg)
    assert np.isclose(cost, expected, rtol=1e-10)
    
    
if __name__ == "__main__":
    # Run pytest programmatically
    import sys
    pytest.main([__file__, "-v","-s", "--tb=short"])