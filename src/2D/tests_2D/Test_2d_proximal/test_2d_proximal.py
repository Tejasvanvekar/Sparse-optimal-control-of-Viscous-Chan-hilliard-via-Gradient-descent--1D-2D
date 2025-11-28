# -*- coding: utf-8 -*-
"""
Proximal (ISTA) tests for the two‑dimensional optimal control solver.

This file contains a set of tests that validate the implementation of the
proximal update (also known as ISTA – Iterative Shrinkage/Thresholding
Algorithm) used to handle the ℓ¹ sparsity term in the control of the
two‐dimensional Cahn–Hilliard system.  In contrast to earlier versions of
the test suite, all solver parameters are read from ``config.ForwardSolverConfig``
and ``OptimizationConfig`` objects, and no module‐level globals from
``Forward2_solver`` are referenced directly.  Synthetic grids are created via
fixtures and the tests themselves do not open any interactive plots.

The tests cover the following properties:

* **Closed form solution:** When the box constraints are inactive and the
  step size ``alpha`` is one, a single ISTA step reduces to a soft
  thresholding operation.  This test checks that the proximal operator
  matches the closed form solution exactly.
* **Soft threshold then clip:** When box constraints are present, the
  proximal operator is equivalent to applying soft thresholding and then
  projecting onto the admissible interval ``[u_min, u_max]``.  This is
  verified for several shapes and bounds.
* **Monotonic objective:** For an objective of the form
  ``0.5||u − y||² + kappa_sparsity * ||u||₁``, repeated ISTA steps with
  ``alpha <= 1/L`` should produce a nonincreasing sequence of objective
  values.  The test runs many iterations and checks that the objective
  decreases (allowing for tiny numerical variations).
* **Fixed point characterisation:** A point obtained by soft thresholding
  ``y`` is a fixed point of the ISTA mapping.  This test asserts that
  starting from such a point and applying the proximal step yields the
  same point.

These tests mirror the one‐dimensional proximal tests but operate on
two–dimensional arrays (or, by virtue of numpy’s broadcasting, any shape).
"""

import numpy as np
import pytest



# Project functions
from cost2_and_function import (
    
    proximal_step,
)

from Forward2_solver import (
    laplacian_matrix_neumann)
    
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




# ============================= ISTA / Proximal tests (2-D) =============================

def _soft_threshold(v, lam):
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)

def _obj_l2_l1(u, y, kappa_prox):
    return 0.5 * np.sum((u - y) ** 2) + kappa_prox * np.sum(np.abs(u))

@pytest.mark.parametrize("shape", [(24, 36), (15, 25)])
def test_ista_one_step_equals_closed_form_2d(shape):
    """Check that a single ISTA step equals soft thresholding when unconstrained.

    For ``alpha = 1`` and no box bounds, the proximal operator

        ``prox_{kappa ||·||₁}(y)``

    has the closed form ``soft_threshold(y, kappa)``.  Starting from a random
    ``u`` and computing ``u − alpha * grad`` where ``grad = u − y`` leads to
    ``u_temp = y``.  The proximal update should therefore equal
    ``soft_threshold(y, kappa)``.  This test uses random matrices of various
    shapes to ensure broadcast correctness.
    """
    """
    One ISTA step with α=1 and no box bounds must equal the closed-form proximal solution:
       prox_{κ‖·‖₁}(y) = soft_threshold(y, κ)
    """
    rng = np.random.default_rng(7)
    y = rng.normal(0.0, 0.7, size=shape)
    kappa_test = 0.12
    alpha = 1.0

    u = rng.normal(0.0, 0.5, size=shape)
    grad = u - y  # gradient of 0.5||u - y||^2
    BIG = 1e12
    opt = OptimizationConfig(b1=0.0, b2=0.0, b3=0.0,
                             kappa_sparsity=kappa_test,
                             u_min=-BIG, u_max=+BIG)
    u_next = proximal_step(u, grad, alpha, opt)
    u_star = _soft_threshold(y, kappa_test)

    assert np.allclose(u_next, u_star, atol=1e-12, rtol=0.0)

@pytest.mark.parametrize("shape,lbound,ubound", [((20, 30), -0.6, 0.8), ((15, 40), -0.5, 0.5)])
def test_ista_one_step_with_box_equals_soft_then_clip_2d(shape, lbound, ubound):
    """Verify that one ISTA step with bounds equals soft thresholding then clipping.

    When box constraints ``[lbound, ubound]`` are active, the proximal
    operator first performs soft thresholding and then projects onto the
    interval.  Starting from ``u = 0`` and ``grad = −y`` with ``alpha = 1``
    again gives ``u_temp = y``.  The test asserts that the computed update
    matches ``clip(soft_threshold(y, kappa), lbound, ubound)``.
    """
    """
    One ISTA step with α=1 and box constraints must equal: clip(soft_threshold(y, κ), [l,u]).
    """
    rng = np.random.default_rng(11)
    y = rng.normal(0.0, 0.9, size=shape)
    kappa_test = 0.2
    alpha = 1.0

    u = np.zeros(shape)
    grad = u - y
    opt = OptimizationConfig(b1=0.0, b2=0.0, b3=0.0,
                             kappa_sparsity=kappa_test,
                             u_min=lbound, u_max=ubound)
    u_next = proximal_step(u, grad, alpha, opt)

    u_star = np.clip(_soft_threshold(y, kappa_test), lbound, ubound)
    assert np.allclose(u_next, u_star, atol=1e-12, rtol=0.0)

def test_ista_objective_monotone_nonincreasing_2d():
    """Ensure the ISTA objective decreases monotonically.

    The objective function is ``0.5||u − y||² + kappa_prox * ||u||₁``.  For
    ``alpha <= 1/L`` with Lipschitz constant ``L=1`` for the quadratic term,
    each ISTA iteration should not increase the objective.  This test runs
    150 iterations from an initial zero control and checks that the
    sequence of objective values is nonincreasing up to tiny numerical
    tolerances.  Intermediate objective values are printed every 10
    iterations for debugging.
    """
    """
    The ISTA objective 0.5||u - y||^2 + κ||u||_1 should be nonincreasing with α <= 1 (Lipschitz=1).
    """
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 1.0, size=(60, 80))
    kappa_prox = 0.1
    alpha = 0.9  # <= 1/L with L=1 for 0.5||u - y||^2
    u = np.zeros_like(y)
    BIG = 1e12
    opt = OptimizationConfig(b1=0.0, b2=0.0, b3=0.0,
                             kappa_sparsity=kappa_prox,
                             u_min=-BIG, u_max=+BIG)

    vals = []
    for _ in range(150):
        vals.append(_obj_l2_l1(u, y, kappa_prox))
        grad = u - y
        u = proximal_step(u, grad, alpha, opt)  # does (u - α∇f) then prox_{κ‖·‖₁} (no clipping effect)

    diffs = np.diff(vals)
    assert np.all(diffs <= 1e-12 + 1e-12*np.abs(vals[:-1]))

@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0])
def test_fixed_point_characterization_2d(alpha):
    """Check that soft thresholding of ``y`` yields a fixed point of the ISTA map.

    A vector ``u_star = soft_threshold(y, kappa)`` satisfies

        ``u_star = prox_{alpha kappa ||·||₁}(u_star − alpha (u_star − y))``.

    This property characterises fixed points of the ISTA iteration.  The test
    generates a random ``y``, computes ``u_star``, and verifies that
    applying the proximal update returns ``u_star`` exactly (within
    numerical tolerance).
    """
    """
    If u* = soft_threshold(y, κ), then u* is a fixed point of the ISTA map:
       u* = prox_{α κ‖·‖₁}( u* - α(u* - y) )
    """
    rng = np.random.default_rng(21)
    y = rng.normal(0.0, 0.8, size=(32, 48))
    kappa_test = 0.15

    u_star = _soft_threshold(y, kappa_test)
    grad = u_star - y
    BIG = 1e12
    opt = OptimizationConfig(b1=0.0, b2=0.0, b3=0.0,
                             kappa_sparsity=kappa_test,
                             u_min=-BIG, u_max=+BIG)
    u_fp = proximal_step(u_star, grad, alpha, opt)

    assert np.allclose(u_fp, u_star, atol=1e-12, rtol=0.0)



if __name__ == "__main__":
    # Run pytest programmatically
    import sys
    pytest.main([__file__, "-v","-s", "--tb=short"])