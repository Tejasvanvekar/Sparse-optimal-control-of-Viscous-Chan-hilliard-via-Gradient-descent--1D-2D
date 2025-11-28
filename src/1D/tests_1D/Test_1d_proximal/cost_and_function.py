# This module implements the discrete cost functional and its gradient for the
# sparse optimal control of the viscous Cahn–Hilliard system.
#
# The total cost J is the sum of four terms: a space–time tracking cost, a terminal
# cost, a control energy cost, and an L¹ sparsity penalty.  In the continuous
# setting the cost is
#
#   J(ϕ, u) = \frac{b_1}{2}∫∫_Q (ϕ − ϕ_Q)^2 dt dx + \frac{b_2}{2}∫_Ω (ϕ(T) − ϕ_T)^2 dx
#            + \frac{b_3}{2}∫∫_Q u^2 dt dx + κ ∫∫_Q |u| dt dx.
#
# This discretisation follows exactly that structure: the first three terms are
# computed by applying the trapezoidal rule in space and time to the squared
# differences, while the last term uses the trapezoidal rule on the absolute
# value of u.  The weights b1, b2, b3 and kappa correspond to the physical
# parameters in the cost functional.  The gradient of the smooth part with
# respect to u is ∇u J = r + b3 u, where r is the adjoint variable computed by the
# backward solver.  Sparsity enters via a proximal soft–thresholding
# step implemented elsewhere in the optimisation loop.


# --- 1. Import your functions from their respective files ---
from Forward_solver import run_main_simulation

import numpy as np

def calculate_cost(
    phi_hist: np.ndarray,
    u: np.ndarray,
    phi_Q_target: np.ndarray,
    phi_T_target: np.ndarray,
    x: np.ndarray,
    t_hist: np.ndarray,
    b1: float,
    b2: float,
    b3: float,
    kappa: float,
    verbose: bool = True
) -> float:
    """
    Calculates the value of the discrete cost functional.

    Args:
        phi_hist: State history from the forward solver, shape (M+1, N+1).
        u: Control function, same shape as phi_hist.
        phi_Q_target: Space-time target, same shape as phi_hist.
        phi_T_target: Final time target, shape (N+1,).
        x: Spatial grid.
        t_hist: Time grid.
        b1, b2, b3, kappa: Cost functional weights.

    Returns:
        The total cost (a single scalar value).
    """
    # Term 1: Tracking Cost (b1 term)
    error_sq = (phi_hist - phi_Q_target)**2
    # First, integrate over space for each time step
    integral_in_space_b1 = np.trapezoid(error_sq, x=x, axis=1)
    # Then, integrate the result over time
    cost1 = (b1 / 2.0) * np.trapezoid(integral_in_space_b1, x=t_hist)

    # Term 2: Terminal Cost (b2 term)
    final_error_sq = (phi_hist[-1] - phi_T_target)**2
    cost2 = (b2 / 2.0) * np.trapezoid(final_error_sq, x=x)

    # Term 3: Control Energy Cost (b3 term)
    u_sq = u**2
    integral_in_space_b3 = np.trapezoid(u_sq, x=x, axis=1)
    cost3 = (b3 / 2.0) * np.trapezoid(integral_in_space_b3, x=t_hist)

    # Term 4: Sparsity Cost (kappa term)
    u_abs = np.abs(u)
    integral_in_space_kappa = np.trapezoid(u_abs, x=x, axis=1)
    cost4 = kappa * np.trapezoid(integral_in_space_kappa, x=t_hist)

    total_cost = cost1 + cost2 + cost3 + cost4
    
    print(f"  Tracking Cost (J1): {cost1:.6g}")
    print(f"  Terminal Cost (J2): {cost2:.6g}")
    print(f"  Control Energy (J3): {cost3:.6g}")
    print(f"  Sparsity Cost (J4): {cost4:.6g}")
    print(f"-----------------------------")
    print(f"  Total Cost: {total_cost:.6g}")

    return total_cost

def calculate_gradient(r: np.ndarray, u: np.ndarray, b3: float) -> np.ndarray:
    """
    Calculates the gradient of the smooth part of the cost functional.

    Args:
        r: The adjoint state variable 'r' from the backward solver.
        u: The current control function.
        b3: The weight of the control energy term in the cost functional.

    Returns:
        The gradient of the smooth part of the cost, with the same shape as r and u.
    """
    # The formula is derived from the first-order necessary optimality conditions
    grad_smooth = r + b3 * u
    return grad_smooth

# From update_step.py
def perform_gradient_step(
    u_current: np.ndarray, 
    grad_smooth: np.ndarray, 
    alpha: float
) -> np.ndarray:
    """
    Performs one gradient descent step on the smooth part of the cost functional.
    """
    u_temp = u_current - alpha * grad_smooth
    return u_temp

