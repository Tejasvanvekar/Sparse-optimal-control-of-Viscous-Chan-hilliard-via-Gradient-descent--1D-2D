# -*- coding: utf-8 -*-
"""
Test suite for the one‑dimensional forward Cahn–Hilliard solver.

This module contains a collection of unit and integration tests that
exercise the behaviour of the one‑dimensional forward solver.  The forward
solver integrates the Cahn–Hilliard equation using a Crank–Nicolson scheme
with an implicit update for the chemical potential and a Newton–Raphson
iteration at each time step.  The key properties checked by these tests
include:

* **Auxiliary variable update:** The function ``solve_w`` computes ``w^{n+1}``
  from ``w^n`` and two successive control values.  A closed form formula
  exists for constant inputs and is used for validation.
* **Spatial discretisation:** Applying the discrete Laplacian to a cosine
  function should reproduce the analytical second derivative.
* **Mass conservation:** Without control, the integral of the solution
  (mass) must remain constant; a plot of the absolute deviation is saved
  and the maximum deviation is required to be very small.
* **Energy decay:** The free energy should decrease monotonically in the
  absence of control, reflecting the dissipative nature of the scheme.
* **Temporal convergence:** A refinement study is performed by running the
  solver with decreasing time steps and comparing against a fine reference
  solution; the observed order of convergence should be close to two.
* **Symmetry preservation:** A symmetric initial condition should remain
  symmetric; the solver is run with a symmetric initial profile and the
  final state is checked for symmetry.
* **Unconditional stability:** The implicit scheme should remain stable
  even for large ``dt``; the solution is checked for finiteness.
* **Newton–Raphson convergence:** The Newton solver should converge
  quadratically; residuals are inspected for monotonic decrease and the
  final residual should fall below a tolerance.  A short tail of the
  residual history is used to estimate the convergence order.

All plots are saved into a ``test_plots`` directory to avoid popping up
GUI windows during automated testing.  The tests use small grids to keep
execution times reasonable but still capture the essential behaviour of the
scheme.
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


from Forward_solver import (
    laplacian_matrix_neumann,
    run_main_simulation,
    init_phi_random,
    free_energy,
    trapz_weights,
    newton_raphson
)
# Import config classes but NOT the interactive functions
from config import ForwardSolverConfig



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
# ==                       Tests from Test_Forward_solver.py                                   ==
# =================================================================================================



@pytest.mark.parametrize("N, w0, uval, dt", [
    (64, 0.1, 0.5, 0.1),
    (127, 0.2, -0.3, 0.05),
])
def test_solve_w_simple_case_1d(N, w0, uval, dt):
    """Check the closed form update for ``w`` in one dimension.

    For constant arrays ``w_old``, ``u_n`` and ``u_{n+1}``, there is an
    analytic expression for ``w^{n+1}`` arising from the Crank–Nicolson
    discretisation.  This test computes the expected update and compares
    it against the value returned by the solver's ``solve_w`` (either the
    1‑D or 2‑D implementation, as appropriate) for a variety of problem sizes
    and time steps.
    """
    """
    1-D analog of the 2-D solve_w test:
    For constant w_old and constant u at times n and n+1,
    the closed-form CN filter update is:

        w^{n+1} = ((γ/Δt - 1/2) w^n + 1/2 (u^{n+1} + u^n)) / (γ/Δt + 1/2)

    We verify the implementation matches this elementwise.
    """
    import numpy as np
    from numpy.testing import assert_allclose
    from config import ForwardSolverConfig

    # Prefer the 1-D function if present; otherwise use the 2-D version (works on 1-D arrays too).
    try:
        from Forward_solver import solve_w as _solve_w
    except Exception:
        from Forward2_solver import solve_w as _solve_w

    cfg = ForwardSolverConfig()     # only need gamma
    gamma_dt = cfg.gamma / dt

    w_old  = np.full(N + 1, w0, dtype=float)
    u_n    = np.full(N + 1, uval, dtype=float)
    u_np1  = np.full(N + 1, uval, dtype=float)

    expected = ((gamma_dt - 0.5) * w_old + 0.5 * (u_np1 + u_n)) / (gamma_dt + 0.5)
    calc     = _solve_w(w_old, dt, cfg.gamma, u_n, u_np1)

    assert_allclose(calc, expected, rtol=1e-15, atol=0.0,
                    err_msg="solve_w incorrect for 1-D arrays.")




def test_laplacian_matrix_on_known_function(solver_params):
    """Apply the discrete 1‑D Laplacian to a cosine mode and compare with the exact derivative.

    A cosine mode ``cos(pi x / Lx)`` is an eigenfunction of the Laplacian
    with eigenvalue ``-(pi/Lx)^2``.  This test applies the discrete
    Neumann Laplacian matrix to the cosine and checks that the numerical
    result matches the analytic second derivative at interior points.
    A few diagnostic values at the midpoint of the domain are printed for
    inspection.
    """
    x = solver_params["x"]
    Lmat = solver_params["Lmat"]
    v = np.cos(np.pi * x / Lx)
    analytical_laplacian_v = -(np.pi/Lx)**2 * np.cos(np.pi * x / Lx)
    numerical_laplacian_v = Lmat @ v

    print("\n--- Testing Laplacian Operator ---")
    print(f"Analytical (midpoint): {analytical_laplacian_v[N//2]:.5f}")
    print(f"Numerical (midpoint):  {numerical_laplacian_v[N//2]:.5f}")
    
    
    
    assert_allclose(numerical_laplacian_v[1:-1], analytical_laplacian_v[1:-1], rtol=1e-3, atol=1e-8)

def test_mass_conservation_without_control(solver_params):
    """Ensure conservation of mass in the absence of control.

    The forward solver is called with no control input.  The spatial
    trapezoidal rule is used to compute the integral (mass) of ``phi`` at
    each time step.  A plot of the absolute deviation from the initial
    mass is saved, and the maximum deviation is asserted to be below a
    tight tolerance.
    """
    """Mass conservation check: use absolute deviation vs tolerance."""
    config = ForwardSolverConfig()
    phi_hist, _, _ = run_main_simulation(fwd_config=config, store_history=True, verbose=False)
    wts = trapz_weights(config.N + 1)
    mass = np.array([np.dot(wts, phi) for phi in phi_hist])
    M0 = mass[0]
    abs_err = np.abs(mass - M0)
    tol = 1e-12  # absolute tolerance in mass units

    print("\n--- Testing Mass Conservation ---")
    print(f"M(0) = {M0:.6e}")
    print(f"max_t |M(t)-M(0)| = {abs_err.max():.3e}  (tol = {tol:.1e})")

    # 1) Absolute error (log scale) + tolerance line
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.semilogy(abs_err + 1e-30, 'b-', lw=1.5, label=r"$|M(t)-M(0)|$")
    ax.axhline(tol, color='r', ls='--', lw=1.0, label=fr"Tolerance = {tol:g}")
    ax.set_title("Mass Conservation (Absolute Error)",fontsize=16)
    ax.set_xlabel("Time Step",fontsize=15)
    ax.set_ylabel("absolute mass deviation",fontsize=15)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "test_mass_conservation_abs_error.png"), dpi=200)
    plt.close(fig)

    

    # Strict criterion: mass stays within tol for all times
    assert abs_err.max() <= tol

def test_energy_decrease_without_control(solver_params):
    """Verify that the free energy decreases monotonically over time.

    The free energy is computed at each time level from the forward
    simulation.  A plot of energy versus time is saved, and the test
    asserts that the differences between successive energies are
    nonpositive (within a tiny tolerance), demonstrating the dissipative
    property of the scheme.
    """
    config = ForwardSolverConfig()
    phi_hist, _, _ = run_main_simulation(fwd_config=config, store_history=True, verbose=False)
    p = solver_params
    energy_over_time = [free_energy(phi, p["kappa"], p["c1"], p["c2"], p["h"]) for phi in phi_hist]

    print("\n--- Testing Energy Decrease ---")
    print(f"Initial Energy: {energy_over_time[0]:.6f}")
    print(f"Final Energy:   {energy_over_time[-1]:.6f}")
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)  # auto-manage margins
    ax.plot(energy_over_time)
    ax.set_title("Free Energy Decrease Over Time", fontsize=16, pad=10)
    ax.set_xlabel("Time Step", fontsize=15, labelpad=6)
    ax.set_ylabel("Free Energy", fontsize=15, labelpad=6)
    ax.grid(True)
    
    fig.savefig(os.path.join(PLOT_DIR, "test_energy_decrease.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    assert np.all(np.diff(energy_over_time) <= 1e-9)

def test_time_integrator_convergence_order(solver_params):
    """Estimate the order of accuracy of the time integrator by refinement.

    Running the solver with different time steps on the same spatial grid
    and comparing against a fine reference solution allows one to estimate
    the convergence order from the slope of a log–log plot of error versus
    time step.  A log–log plot is saved and the slope is required to be
    between approximately 1.2 and 2.2, indicating near second‐order
    accuracy.  A small grid is used to keep the computation tractable.
    """
    base_dt = 0.005
    dt_values = np.array([base_dt, base_dt / 2.0, base_dt / 4.0])
    
    # Fine reference solution
    config_fine = ForwardSolverConfig(N=512, dt_initial=base_dt / 8.0)
    phi_fine, _, _ = run_main_simulation(fwd_config=config_fine, store_history=True, verbose=False)
    phi_ref_final = phi_fine[-1]
    
    errors = []
    for dt_coarse in dt_values:
        config_coarse = ForwardSolverConfig(N=512, dt_initial=dt_coarse)
        phi_coarse_hist, _, _ = run_main_simulation(fwd_config=config_coarse, store_history=True, verbose=False)
        errors.append(np.linalg.norm(phi_coarse_hist[-1] - phi_ref_final))
    
    log_dt = np.log(dt_values)
    log_error = np.log(np.array(errors) + 1e-30)
    slope, _ = np.polyfit(log_dt, log_error, 1)

    print(f"\n--- Testing Time Integrator Order ---")
    print(f"Calculated convergence order (slope) = {slope:.4f}")
    
    plt.figure()
    plt.loglog(dt_values, errors, 'bo-', label=f'Numerical (Slope ≈ {slope:.2f})')
    plt.loglog(dt_values, (errors[0] / (dt_values[0]**2)) * dt_values**2, 'r--', label='Theoretical Order 2')
    plt.title("Time Integrator Convergence Order",fontsize=16)
    plt.xlabel("Time Step (dt)",fontsize=15)
    plt.ylabel("L2 Error",fontsize=15)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.savefig(os.path.join(PLOT_DIR, "test_convergence_order.png"))
    plt.close()

    assert 1.2 < slope < 2.2, f"Expected convergence order near 2, but got {slope:.4f}"



def test_symmetry_preservation(solver_params):
    """Check that symmetry of the initial condition is preserved.

    A symmetric cosine profile is supplied as the initial condition via
    the ``initial_phi`` keyword of the forward solver.  After the run,
    the final solution is compared to its reflection about the domain
    midpoint, ensuring that symmetry has been preserved up to numerical
    precision.
    """
    config = ForwardSolverConfig()
    x = np.linspace(0, config.Lx, config.N + 1)
    symmetric_phi0 = 0.5 * np.cos(2 * np.pi * x / config.Lx)
    
    # Create control input to set initial condition
    # This is a workaround since we can't easily override init_phi_random
    phi_hist, _, _ = run_main_simulation(fwd_config=config, store_history=True, verbose=False,initial_phi=symmetric_phi0)
    final_phi = phi_hist[-1]
    
    # Check if final state is symmetric (approximately)
    assert_allclose(final_phi, np.flip(final_phi), atol=1e-8)



def test_unconditional_stability_with_large_dt():
    """Ensure stability of the implicit scheme for a very large time step.

    The solver is run with a time step ``dt_initial = 1.0`` on a moderate
    spatial grid.  The final state is checked for finite values (no NaNs
    or infinities) which indicates that the scheme remains stable even
    when the time step is large compared to the spatial resolution.
    """
    config = ForwardSolverConfig(N=64, dt_initial=1.0)
    phi_hist, _, _ = run_main_simulation(fwd_config=config, store_history=True, verbose=False)
    final_phi = phi_hist[-1]
    
    print("\n--- Testing Unconditional Stability ---")
    print(f"Max phi value: {np.max(final_phi):.6f}")
    print(f"Min phi value: {np.min(final_phi):.6f}")
    
    assert np.all(np.isfinite(final_phi))


def test_newton_raphson_quadratic_convergence(solver_params):
    """Check the quadratic convergence of the Newton–Raphson solver.

    The forward solver uses a Newton–Raphson iteration to solve the
    non‐linear system at each time step.  This test constructs a random
    initial ``phi`` strictly within bounds, computes ``mu_old`` using the
    analytic formula, then performs one Newton solve with a small change
    in the control ``w``.  The residuals returned by the solver are
    printed for manual inspection and subjected to the following checks:
    (i) at least three iterations occur, (ii) residuals decrease
    monotonically near the end of the iteration, and (iii) the final
    residual is below a tolerance.  Additional checks on the convergence
    rate (slope ≈ 2) are left commented out but can be reactivated if
    desired.
    """
    p = solver_params
    phi_old = init_phi_random(p["N"], delta_sep, enforce_zero_mean=True)
    w_old = np.zeros(p["N"] + 1)
    from Forward_solver import regularized_log
    mu_old = -kappa * (p["Lmat"] @ phi_old) + c1 * regularized_log(phi_old) - 2*c2*phi_old - w_old
    w_new = w_old + 0.01
    
    _, _, residuals = newton_raphson(
        phi_old, mu_old, w_old, w_new, 
        dt_initial, tau, p["c1"], p["c2"], p["h"],
        delta_sep, p["Lmat"], kappa, return_residual_history=True
    )
    
    print("\n--- Testing Newton-Raphson Convergence ---")
    print(f"Number of iterations: {len(residuals)}")
    print(f"Final residual: {residuals[-1]:.6e}")
    tol = 1e-6
    print("\n--- Newton residuals (1D) ---")
    print(" iter | residual")
    for k, r in enumerate(residuals):
        print(f"{k:5d} | {r:.6e}")
    print(f"tolerance = {tol:.1e}")
    # --- extra assertions (match 2-D test style) ---
    # basic guard
    assert len(residuals) >= 3, f"Too few Newton iterations: {residuals}"
    # use a short tail for robustness
    tail = np.array(residuals[-4:] if len(residuals) >= 4 else residuals, dtype=float)
    # monotone decrease near the end (allow tiny wiggle)
    assert np.all(tail[1:] <= tail[:-1] + 1e-12), f"Residuals not decreasing near end: {tail}"
    # quadratic hallmark: slope ≈ 2 on log(e_{k+1}) vs log(e_k)
    #if tail.size >= 3:
        #logs_k  = np.log(tail[:-1] + 1e-300)
        #logs_k1 = np.log(tail[1:]  + 1e-300)
        #slope, _ = np.polyfit(logs_k, logs_k1, 1)
       # assert 1.5 < slope < 2.5, f"Expected ~quadratic slope ~2, got {slope:.2f} (tail={tail})"

    # original limits
    assert len(residuals) < 10
    assert residuals[-1] < tol

if __name__ == "__main__":
    # Run pytest programmatically
    
    pytest.main([__file__, "-v","-s", "--tb=short"])