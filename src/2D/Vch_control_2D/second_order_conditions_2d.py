"""
second_order_conditions_2d.py
=============================

This module provides utilities to test second‑order optimality conditions
and sparsity for the 2D Cahn–Hilliard control problem.  After an optimal
control ``u_star`` has been obtained, it is prudent to verify whether the
cost functional is locally convex along feasible directions and whether the
sparsity condition holds.

Two main checks are implemented:

1. **Finite‑difference approximation of the second derivative** using
   random directions in the critical cone.  The function
   :func:`approximate_second_order_condition_2d` perturbs the optimal
   control along random feasible directions and evaluates the resulting
   second‑order variation of the cost functional.
2. **Sparsity condition verification** via :func:`verify_sparsity_condition`,
   which compares the zero set of the optimal control with the region where
   the filtered adjoint satisfies |r*| ≤ κ.

These diagnostics help ensure that a candidate solution is a local
minimum and complies with necessary optimality conditions.
"""

import numpy as np
from typing import List, Optional

# Project imports
from Forward2_solver import run_main_simulation
from cost2_and_function import calculate_cost
from config import OptimizationConfig, ForwardSolverConfig


def _generate_direction(
    u_star: np.ndarray,
    r_star: np.ndarray,
    u_min: float,
    u_max: float,
    rng: np.random.Generator,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Generate a random direction within the critical cone C(u*).

    The critical cone at the optimal control u* consists of perturbations
    that respect the bound constraints.  Specifically, if u* is at its
    lower bound at a point, only non‑negative perturbations are allowed
    there; if u* is at its upper bound, only non‑positive perturbations
    are allowed.  This helper produces a random perturbation ``v``, then
    enforces these sign conditions and normalizes the vector.

    Parameters
    ----------
    u_star : ndarray
        Optimal control field.
    r_star : ndarray
        Filtered adjoint (unused here but kept for API completeness).
    u_min, u_max : float
        Lower and upper bounds on the control.
    rng : Generator
        NumPy random generator for reproducibility.
    tol : float, optional
        Tolerance used to decide whether u* is at a bound.

    Returns
    -------
    ndarray
        A unit‑norm vector direction obeying critical cone constraints.
    """
    v = rng.standard_normal(size=u_star.shape)

    # Enforce sign constraints at saturation points
    lower_mask = u_star <= (u_min + tol)
    upper_mask = u_star >= (u_max - tol)  # FIX: was (u_max + tol), which rarely triggers

    if np.any(lower_mask):
        v[lower_mask] = np.abs(v[lower_mask])
    if np.any(upper_mask):
        v[upper_mask] = -np.abs(v[upper_mask])

    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:  # Avoid division by zero
        v = np.zeros_like(v)
        v.ravel()[0] = 1.0
        norm_v = 1.0

    return v / norm_v


def _ensure_opt_config(
    b1: Optional[float],
    b2: Optional[float],
    b3: Optional[float],
    kappa_sparsity: Optional[float],
    opt_config: Optional[OptimizationConfig],
) -> OptimizationConfig:
    """
    Return an OptimizationConfig instance from either a provided object or scalars.

    Supports backward compatibility by allowing users to supply either a
    full ``opt_config`` or the individual scalar weights (b1, b2, b3,
    kappa_sparsity).  Exactly one of ``opt_config`` or the tuple of
    scalars must be provided.
    """
    if opt_config is not None:
        return opt_config
    if any(v is None for v in (b1, b2, b3, kappa_sparsity)):
        raise ValueError(
            "Either provide opt_config or all of (b1, b2, b3, kappa_sparsity)."
        )
    return OptimizationConfig(
        b1=float(b1),
        b2=float(b2),
        b3=float(b3),
        kappa_sparsity=float(kappa_sparsity),
    )


def approximate_second_order_condition_2d(
    u_star: np.ndarray,
    r_star: np.ndarray,
    phi_star: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t_hist: np.ndarray,
    # Preferred new API:
    opt_config: Optional[OptimizationConfig] = None,
    # Legacy API (still supported):
    b1: Optional[float] = None,
    b2: Optional[float] = None,
    b3: Optional[float] = None,
    kappa: Optional[float] = None,
    # Bounds and options:
    phi_Q_target: Optional[np.ndarray] = None,
    phi_T_target: Optional[np.ndarray] = None,
    u_min: float = -np.inf,
    u_max: float = np.inf,
    num_directions: int = 10,
    epsilon: float = 1e-4,
    seed: Optional[int] = None,   # FIX: Python 3.8-friendly typing (no '|')
    fwd_config: Optional[ForwardSolverConfig] = None
) -> List[float]:
    """
    Approximate directional second derivatives of the reduced cost.

    This function estimates the second derivative of the reduced cost
    functional J at an optimal control ``u_star`` and corresponding state
    ``phi_star`` along ``num_directions`` random directions in the critical
    cone.  For each direction ``h``, a small perturbation ``u_star + ε h``
    is constructed and a forward simulation is run to obtain the perturbed
    state.  A finite‑difference formula then approximates hᵀ H h.  The
    baseline cost and gradient are computed once to avoid redundant work.

    Parameters
    ----------
    u_star : ndarray
        Optimal control at which to test second‑order conditions.
    r_star : ndarray
        Filtered adjoint associated with the optimal solution.
    phi_star : ndarray
        State history associated with u_star.
    x, y : 1D ndarrays
        Spatial grids used in integration.
    t_hist : 1D ndarray
        Time grid used in integration.
    opt_config : OptimizationConfig, optional
        Full configuration object providing weights.  If None, legacy scalars
        must be supplied.
    b1, b2, b3, kappa : float, optional
        Legacy scalar weights used if opt_config is None.  kappa refers to
        kappa_sparsity.
    phi_Q_target, phi_T_target : ndarrays, optional
        Desired trajectories used in cost computation; defaults to zero.
    u_min, u_max : float, optional
        Control bounds; used to generate perturbations.
    num_directions : int, optional
        Number of random directions to sample.
    epsilon : float, optional
        Magnitude of the perturbation; small positive number.
    seed : int, optional
        Seed for random number generation.
    fwd_config : ForwardSolverConfig, optional
        Forward solver configuration for running perturbed simulations.

    Returns
    -------
    List[float]
        Estimates of the quadratic form hᵀ H h for each sampled direction.
    """
    rng = np.random.default_rng(seed)

    # Build config from legacy scalars if needed
    opt = _ensure_opt_config(b1, b2, b3, kappa, opt_config)

    # Defaults if not provided
    if phi_Q_target is None:
        phi_Q_target = np.zeros_like(phi_star)
    if phi_T_target is None:
        phi_T_target = np.zeros_like(phi_star[-1])

    # Baseline cost and gradient at u_star
    cost_star = calculate_cost(
        phi_star, u_star, phi_Q_target, phi_T_target,
        x, y, t_hist, opt
    )
    grad_star = r_star + opt.b3 * u_star  # smooth part

    directional_second_derivatives: List[float] = []
    print(f"Testing {num_directions} random directions in the critical cone...")

    for i in range(num_directions):
        h = _generate_direction(u_star, r_star, u_min, u_max, rng)
        u_perturbed = u_star + epsilon * h

        # Forward solve with perturbed control
        phi_perturbed, _, _ = run_main_simulation(config=fwd_config,
            store_history=True,
            control_input=u_perturbed,
            verbose=False,
        )

        # Perturbed cost
        cost_perturbed = calculate_cost(
            phi_perturbed, u_perturbed, phi_Q_target, phi_T_target,
            x, y, t_hist, opt
        )

        inner_prod = np.sum(grad_star * h)
        # Finite-difference estimate of h^T H h
        d2 = (cost_perturbed - cost_star - epsilon * inner_prod) / (0.5 * epsilon**2)
        directional_second_derivatives.append(float(d2))
        print(f"  Direction {i+1}/{num_directions}: estimated d²J/dh² ≈ {d2:.6e}")

    return directional_second_derivatives


def verify_sparsity_condition(
    u_optimal: np.ndarray,
    r_optimal: np.ndarray,
    kappa: float,
    tol: float = 1e-6
) -> None:
    """
    Verify the necessary sparsity condition for an optimal control.

    According to optimality conditions for control problems with L1
    regularization, at points where the optimal control u* is zero the
    magnitude of the filtered adjoint r* should not exceed the sparsity
    weight κ.  Conversely, if |r*| ≤ κ then the optimal control should be
    zero there.  This function performs a numerical check of this
    equivalence and prints statistics on how many grid points satisfy or
    violate the condition.

    Parameters
    ----------
    u_optimal : ndarray
        Optimal control field.
    r_optimal : ndarray
        Filtered adjoint field corresponding to the optimal control.
    kappa : float
        Sparsity weight κ (from the L1 regularization term).
    tol : float, optional
        Tolerance below which the control is considered zero.

    Returns
    -------
    None
        Prints diagnostic information to standard output.
    """
    print("\n" + "=" * 60)
    print("VERIFYING SPARSITY CONDITION")
    print("Condition: u*(x,t) = 0  <=>  |r*(x,t)| <= kappa")
    print("=" * 60)

    u_flat = np.ravel(u_optimal)
    r_flat = np.ravel(r_optimal)

    is_u_zero = np.abs(u_flat) < tol
    is_r_small = np.abs(r_flat) <= kappa

    conditions_match = (is_u_zero == is_r_small)
    total_points = u_flat.size
    u_zero_count = int(np.sum(is_u_zero))
    r_small_count = int(np.sum(is_r_small))
    match_count = int(np.sum(conditions_match))
    sparsity_percentage = (u_zero_count / total_points) * 100.0
    match_percentage = (match_count / total_points) * 100.0

    print(f"Sparsity of final control (u* ≈ 0): {sparsity_percentage:.2f}% ({u_zero_count}/{total_points} points)")
    print(f"Region where |r*| <= kappa:          {(r_small_count / total_points) * 100.0:.2f}% ({r_small_count}/{total_points} points)")
    print(f"Percentage of points where the conditions match: {match_percentage:.2f}%")
    if match_percentage > 99.0:
        print("\n\u2713 The sparsity condition is satisfied.")
    else:
        print("\n\u26A0 The sparsity condition is not fully satisfied.")
    print("=" * 60)
