
"""
Utility functions for the 2D Cahn–Hilliard optimization loop.

This module collects helper functions used by the gradient descent solver
in :mod:`GD2_configured`.  The cost functional for the optimal control
problem comprises several terms: a space–time tracking cost that measures
deviation of the phase field φ from a desired path ``phi_Q_target``;
a terminal cost penalizing the difference at the final time between φ and
``phi_T_target``; a quadratic cost on the control input ``u``; and an L1
sparsity term promoting sparse controls.  The functions here perform
vectorized trapz-based integration of these terms, compute the gradient
with respect to ``u`` (the smooth part), and implement the proximal
gradient step incorporating the sparsity penalty and bound projection.
"""
import numpy as np
from config import OptimizationConfig

def calculate_cost(
    phi_hist: np.ndarray,
    u: np.ndarray,
    phi_Q_target: np.ndarray,
    phi_T_target: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t_hist: np.ndarray,
    opt_config: OptimizationConfig,
) -> float:
    """
    Compute the total cost functional for a given state and control.

    The objective functional J(u) is discretized using the trapezoidal rule
    in both space and time.  It is composed of four contributions:

    1. **Tracking cost (J1)**: measures how closely the simulated φ
       follows the desired space–time trajectory ``phi_Q_target`` over the
       entire simulation.  It is weighted by ``b1``.
    2. **Terminal cost (J2)**: penalizes the difference at the final time
       between φ and ``phi_T_target``, weighted by ``b2``.
    3. **Control energy (J3)**: a quadratic energy ∫ |u|² dt dx dy,
       weighted by ``b3``.
    4. **Sparsity cost (J4)**: an L1 term ∫ |u| dt dx dy promoting sparse
       controls, weighted by ``kappa_sparsity``.

    Parameters
    ----------
    phi_hist : ndarray
        Phase–field history of shape (M, Nx, Ny).
    u : ndarray
        Control history with the same shape as ``phi_hist``.
    phi_Q_target : ndarray
        Target trajectory φ_Q(t,x,y) of the same shape as ``phi_hist``.
    phi_T_target : ndarray
        Target final state at t = T of shape (Nx, Ny).
    x, y : 1D ndarrays
        Spatial coordinates used for integration.
    t_hist : 1D ndarray
        Time grid used for temporal integration.
    opt_config : OptimizationConfig
        Contains weights (b1, b2, b3) and sparsity coefficient.

    Returns
    -------
    float
        The total cost evaluated for the provided state and control.
    """
    # --- Unpack optimization parameters from the config object ---
    # Extract weights and sparsity parameter from the configuration for
    # readability.  These values influence how strongly each term of the
    # functional is penalized.
    b1 = opt_config.b1
    b2 = opt_config.b2
    b3 = opt_config.b3
    kappa_sparsity = opt_config.kappa_sparsity

    # --- Term 1: Tracking Cost (J1) ---
    # Compute φ deviation from the desired trajectory and integrate
    # |φ − φ_Q|² over space (nested trapz) and time.  This yields a scalar
    # value measuring the tracking performance.
    error_sq = (phi_hist - phi_Q_target)**2
    # Spatially integrate at each time step: first along y-axis, then x-axis
    integral_in_space_b1 = np.trapz(np.trapz(error_sq, y, axis=-1), x, axis=-1)
    # Temporally integrate the result
    cost1 = (b1 / 2.0) * np.trapz(integral_in_space_b1, x=t_hist)

    # --- Term 2: Terminal Cost (J2) ---
    # Penalize the discrepancy at the final time.  Only the last slice of
    # phi_hist is used here, so we integrate over space only.
    final_error_sq = (phi_hist[-1] - phi_T_target)**2
    cost2 = (b2 / 2.0) * np.trapz(np.trapz(final_error_sq, y, axis=-1), x, axis=-1)

    # --- Term 3: Control Energy Cost (J3) ---
    # Quadratic penalization of the control input.  A large ``b3`` will
    # discourage high control amplitudes, while a smaller value allows more
    # aggressive actuation.  Integrate |u|² over space and time.
    u_sq = u**2
    integral_in_space_b3 = np.trapz(np.trapz(u_sq, y, axis=-1), x, axis=-1)
    cost3 = (b3 / 2.0) * np.trapz(integral_in_space_b3, x=t_hist)

    # --- Term 4: Sparsity Cost (J4) ---
    # L1 sparsity promotes controls that are zero over large regions.  This
    # term uses |u| instead of |u|².  The coefficient ``kappa_sparsity``
    # tunes how strongly sparsity is enforced.
    u_abs = np.abs(u)
    integral_in_space_kappa = np.trapz(np.trapz(u_abs, y, axis=-1), x, axis=-1)
    cost4 = kappa_sparsity * np.trapz(integral_in_space_kappa, x=t_hist)

    total_cost = cost1 + cost2 + cost3 + cost4

    # --- Print a summary of the cost components ---
    # For transparency, print each component of the cost.  Users can tune
    # the weights based on the relative magnitudes of these terms.
    print(f"  Tracking Cost (J1): {cost1:.6g}")
    print(f"  Terminal Cost (J2): {cost2:.6g}")
    print(f"  Control Energy (J3): {cost3:.6g}")
    print(f"  Sparsity Cost (J4): {cost4:.6g}")
    print(f"  -----------------------------")
    print(f"  Total Cost:         {total_cost:.6g}")

    return total_cost


def calculate_gradient(r: np.ndarray, u: np.ndarray, opt_config: OptimizationConfig) -> np.ndarray:
    """
    Compute the gradient of the differentiable part of the cost functional.

    The smooth portion of the objective is ½ b3 ∫ |u|² plus the implicit
    coupling to the state via the adjoint ``r`` (already computed by the
    backward solver).  Differentiating with respect to u yields

        ∂J/∂u = r + b3 * u,

    where ``r`` is the filtered adjoint variable.  This function performs
    elementwise addition on the input arrays.

    Parameters
    ----------
    r : ndarray
        Filtered adjoint variable of shape (M, Nx, Ny).
    u : ndarray
        Current control guess of the same shape.
    opt_config : OptimizationConfig
        Contains the control weight b3.

    Returns
    -------
    ndarray
        The gradient with respect to u.
    """
    return r + opt_config.b3 * u


def proximal_step(
    u_current: np.ndarray,
    grad_smooth: np.ndarray,
    alpha: float,
    opt_config: OptimizationConfig,
) -> np.ndarray:
    """
    Perform a proximal gradient step on the control variable.

    Given the current control ``u_current`` and the gradient of the smooth
    part ``grad_smooth``, this function executes one iteration of the
    proximal gradient method:

    1. **Gradient descent**: take a step of size ``alpha`` along the
       negative gradient of the smooth part to get an intermediate control.
    2. **Soft thresholding**: apply elementwise shrinkage to the
       intermediate control to impose the L1 sparsity penalty.  The threshold
       value is ``alpha * kappa_sparsity``.
    3. **Projection**: clip the resulting control to lie within the
       admissible box [``u_min``, ``u_max``].

    Parameters
    ----------
    u_current : ndarray
        Current iterate of the control.
    grad_smooth : ndarray
        Gradient of the smooth part of the cost with respect to u.
    alpha : float
        Step size (learning rate) used in the gradient update.
    opt_config : OptimizationConfig
        Provides ``kappa_sparsity`` and the box constraints (``u_min``, ``u_max``).

    Returns
    -------
    ndarray
        The updated control after applying the proximal gradient step.
    """
    # 1. Standard gradient descent step on the smooth part
    u_intermediate = u_current - alpha * grad_smooth
    
    # 2. Soft thresholding for the L1 (sparsity) term
    threshold = alpha * opt_config.kappa_sparsity
    u_thresholded = np.sign(u_intermediate) * np.maximum(np.abs(u_intermediate) - threshold, 0)
    
    # 3. Project the result onto the admissible control bounds [u_min, u_max]
    u_projected = np.clip(u_thresholded, opt_config.u_min, opt_config.u_max)
    
    return u_projected


if __name__ == "__main__":
    print("--- Testing 2D cost and gradient functions ---")
    from config import OptimizationConfig

    # --- Setup test parameters and data ---
    M, Nx, Ny = 10, 16, 16
    t_hist = np.linspace(0.0, 1.0, M)
    x = np.linspace(0.0, 2.0, Nx)
    y = np.linspace(0.0, 2.0, Ny)
    
    # Create some dummy data
    phi_hist = np.sin(np.pi * x[None, :, None]) * np.sin(np.pi * y[None, None, :]) * np.cos(t_hist[:, None, None])
    u = np.random.randn(M, Nx, Ny) * 0.5
    phi_Q_target = np.zeros_like(phi_hist)
    phi_T_target = np.zeros((Nx, Ny))

    # Use the config object
    test_opt_config = OptimizationConfig(b1=5.0, b2=10.0, b3=0.01, kappa_sparsity=1e-4)

    print("\n--- Calculating Cost ---")
    total_cost = calculate_cost(
        phi_hist, u, phi_Q_target, phi_T_target, x, y, t_hist, test_opt_config
    )

    print(f"\nExample Total Cost: {total_cost:.6f}")

    print("\n--- Testing Gradient and Proximal Step ---")
    r_test = np.random.randn(M, Nx, Ny)
    grad = calculate_gradient(r_test, u, test_opt_config)
    u_updated = proximal_step(u, grad, alpha=0.1, opt_config=test_opt_config)
    
    print("Gradient and update steps computed successfully.")
    assert u_updated.shape == u.shape
    assert np.all(u_updated >= test_opt_config.u_min)
    assert np.all(u_updated <= test_opt_config.u_max)
    print("✅ Sanity checks passed.")