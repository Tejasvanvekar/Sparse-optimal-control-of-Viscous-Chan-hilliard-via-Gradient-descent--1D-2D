"""
Approximate and check the second–order sufficient optimality conditions for
the sparse Cahn–Hilliard control problem.

Theorem 4.8 of Colli et al. (2024) states that if a locally optimal control u⋆
satisfies the first–order necessary conditions and the coercivity condition
 J″(u⋆)[v,v] > 0 for all nonzero v in the critical cone C(u⋆),
then there exists ε>0 and ζ>0 such that the quadratic growth condition
 J(u) ≥ J(u⋆) + ζ‖u−u⋆‖² holds for all admissible u close to u⋆.

This module samples random directions h in the critical cone and approximates
the second directional derivative of the reduced cost functional using the
finite–difference formula

  d2 ≈ (J(u⋆+εh) − J(u⋆) − ε ⟨∇J(u⋆), h⟩) / (½ ε²),

where ∇J(u⋆) = r⋆ + b₃ u⋆ and r⋆ is the adjoint state.  Positive
values of d2 provide numerical evidence that the coercivity condition holds.
Note that this is only a practical check; a rigorous verification would require
solving the bilinearised state and adjoint systems as described in the paper.
"""

from __future__ import annotations
from config import ForwardSolverConfig
import numpy as np
from typing import List, Tuple

from Forward_solver import run_main_simulation
from cost_and_function import calculate_cost


# --- replace _generate_direction with this version ---
def _generate_direction(u_star, r_star, u_min, u_max, kappa, b3, rng, tol=1e-8, tol_s=1e-9):
    import numpy as np
    v = rng.standard_normal(size=u_star.shape)  # rng is a Generator now
    s_star = r_star + b3 * u_star
    lower = u_star <= (u_min + tol)
    upper = u_star >= (u_max - tol)
    at_zero = np.abs(u_star) <= tol
    kink_interior = at_zero & (np.abs(s_star) < (kappa - tol_s))
    kink_plus     = at_zero & (s_star >= (kappa - tol_s))      # v <= 0
    kink_minus    = at_zero & (s_star <= (-kappa + tol_s))     # v >= 0

    if np.any(lower): v[lower] =  np.abs(v[lower])
    if np.any(upper): v[upper] = -np.abs(v[upper])
    if np.any(kink_interior): v[kink_interior] = 0.0
    if np.any(kink_plus):     v[kink_plus]     = -np.abs(v[kink_plus])
    if np.any(kink_minus):    v[kink_minus]    =  np.abs(v[kink_minus])

    nrm = np.linalg.norm(v)
    if nrm == 0:
        idx = np.unravel_index(np.argmax(np.abs(s_star)), s_star.shape)
        v[idx] = 1.0
        nrm = 1.0
    return v / nrm

def _coerce_rng(seed_or_rng=None):
    import numpy as np
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    if seed_or_rng is None:
        return np.random.default_rng()
    # accept ints, strings that can be cast, etc.
    try:
        return np.random.default_rng(int(seed_or_rng))
    except Exception:
        return np.random.default_rng()



def approximate_second_order_condition(fwd_config: ForwardSolverConfig,
    u_star: np.ndarray,
    r_star: np.ndarray,
    phi_star: np.ndarray,
    x: np.ndarray,
    t_hist: np.ndarray,
    b1: float,
    b2: float,
    b3: float,
    kappa: float,
    phi_Q_target: np.ndarray,
    phi_T_target: np.ndarray,
    u_min: float,
    u_max: float,
    num_directions: int = 10,
    epsilon: float = 1e-4,
    seed: int | None = None,         # <-- new
    rng=None
) -> List[float]:
    """Approximate the second derivative of the reduced cost functional.

    This routine evaluates the finite–difference approximation of the
    second–order directional derivative of the reduced cost functional
    ``J`` at ``u_star`` in ``num_directions`` randomly chosen directions
    belonging to the critical cone.  For each direction ``h``, it computes

        d2 ≈ ( J(u_star + ε h) − J(u_star) − ε ⟨∇J(u_star), h⟩ ) / (0.5 ε² ),

    where the gradient of the smooth part of the cost is ``r_star + b3*u_star``.
    A positive value of ``d2`` indicates that the cost is locally convex in
    the direction ``h``, as required by the coercivity condition (4.54).

    Args:
        u_star: The optimal control array.
        r_star: The adjoint variable associated with the optimal control.
        phi_star: Forward state history corresponding to ``u_star``.
        x: Spatial grid from the forward solver.
        t_hist: Temporal grid from the forward solver.
        b1, b2, b3: Weights in the cost functional.
        kappa: Sparsity weight in the L¹ term.
        phi_Q_target: Space–time target for the state.
        phi_T_target: Final‐time target for the state.
        u_min, u_max: Control bounds.
        num_directions: Number of random test directions to evaluate.
        epsilon: Perturbation size for finite differences.
        seed: Optional random seed for reproducibility.

    Returns:
        A list of approximated second–order directional derivatives for each
        test direction.  Positive values provide evidence of coercivity.
    """
    rng = _coerce_rng(rng if rng is not None else seed)
    # Baseline cost and gradient at u_star
    cost_star = calculate_cost(
        phi_star,
        u_star,
        phi_Q_target,
        phi_T_target,
        x,
        t_hist,
        b1,
        b2,
        b3,
        kappa,
        verbose=False,
    )
    # Smooth part gradient ∇J = r_star + b3 * u_star
    grad_star = r_star + b3 * u_star

    directional_second_derivatives: List[float] = []

    for _ in range(num_directions):
        h = _generate_direction(u_star, r_star, u_min, u_max, kappa, b3, rng)

        # Perturb the control
        u_perturbed = u_star + epsilon * h

        # Run forward simulation for the perturbed control
        phi_perturbed, _, _ = run_main_simulation(fwd_config=fwd_config,
            store_history=True,
            control_input=u_perturbed,
            verbose=False,
        )

        # Evaluate the cost at the perturbed control
        cost_perturbed = calculate_cost(
            phi_perturbed,
            u_perturbed,
            phi_Q_target,
            phi_T_target,
            x,
            t_hist,
            b1,
            b2,
            b3,
            kappa,
            verbose=False,
        )

        # Compute inner product ⟨∇J(u_star), h⟩
        inner_prod = np.sum(grad_star * h)

        # Finite–difference approximation of the second derivative
        d2 = (cost_perturbed - cost_star - epsilon * inner_prod) / (0.5 * epsilon**2)
        directional_second_derivatives.append(d2)

    return directional_second_derivatives