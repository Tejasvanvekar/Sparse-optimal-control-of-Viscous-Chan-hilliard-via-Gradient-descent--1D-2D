# -*- coding: utf-8 -*-
"""
Test suite for the 2‑D backward (adjoint) solver used in the Cahn–Hilliard
optimal control problem.

This module defines a collection of unit tests that verify the correctness of
the adjoint evolution equations in two spatial dimensions.  The adjoint
variables `(p, q, r)` satisfy a coupled Crank–Nicolson time discretisation
derived from the optimality system for a gradient‐based optimal control
algorithm.  These tests construct small synthetic forward histories and
numerically integrate the adjoint equations backwards in time using
``backward2_solver.run_backward``.  Various properties are verified:

* **Terminal conditions:** At the final time, the adjoint variables must
  satisfy linear relations involving the state variable and target data.
* **Discrete adjoint step:** Each time step solves a linear system derived
  from the Crank–Nicolson discretisation; residuals of this equation are
  checked against machine‐precision tolerances.
* **Spatial relation:** The auxiliary variable `q` is defined as `q = -L p`
  where `L` is the Laplacian operator with Neumann boundary conditions.
* **Consistency of the r–equation:** The variable `r` satisfies its own
  Crank–Nicolson update involving `q`, and the residuals of this update are
  asserted.
* **Integration test:** A forward simulation is run to generate a history
  ``phi_hist``.  A short window of the last time steps is passed to the
  backward solver and the resulting adjoint variables are used to rebuild
  the discrete operators `A` and `B`; the correct ordering of these
  operators leads to much smaller residuals than a deliberately swapped
  ordering.

Fixtures construct small grids and provide consistent solver parameters.  All
residuals are normalised by the norms of the left– and right–hand sides to
avoid spurious failures on near‐zero vectors.  Automatic tolerance
estimation functions are used to scale the relative error thresholds based
on the condition numbers of the linear systems under test.
"""
import numpy as np
import pytest
# Project functions

from backward2_solver import run_backward, fpp_log
from Forward2_solver import (
    laplacian_matrix_neumann
    )
    
# Configured input (like 1-D tests)
from config import ForwardSolverConfig
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

# ========= backward tests =========

def _make_synth_forward_2d(Nx_b=6, Ny_b=5, M_b=5, Lx_b=1.0, Ly_b=1.0, T_b=0.2, A_b=0.2):
    """
    Construct a small synthetic forward solution for testing.

    Parameters
    ----------
    Nx_b, Ny_b : int
        Number of interior intervals (so the grid has ``Nx_b+1`` and ``Ny_b+1`` points
        in the x and y directions).  Small values keep the linear solves cheap.
    M_b : int
        Number of time steps minus one; the returned history has ``M_b+1`` time
        levels.
    Lx_b, Ly_b : float
        Domain lengths in x and y.
    T_b : float
        Final time for the synthetic evolution.
    A_b : float
        Amplitude of the spatial sine mode.

    Returns
    -------
    phi_hist : ndarray, shape (M_b+1, Nx_b+1, Ny_b+1)
        Synthetic state history, smoothly varying in time and separable in space.
    x, y : ndarray
        One–dimensional coordinate arrays for the grid.
    t : ndarray
        Time stamps corresponding to each slice of ``phi_hist``.

    Notes
    -----
    The spatial structure is given by ``sin(pi x / Lx_b) * sin(pi y / Ly_b)``.  The
    temporal modulation is a cosine oscillation around a constant amplitude.  This
    function is used by the fixtures to generate forward trajectories for adjoint
    testing.
    """
    # Spatial grid and time discretisation
    x = np.linspace(0.0, Lx_b, Nx_b + 1)
    y = np.linspace(0.0, Ly_b, Ny_b + 1)
    t = np.linspace(0.0, T_b, M_b + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    # Spatial mode and time‐dependent amplitude
    spatial = np.sin(np.pi * X / Lx_b) * np.sin(np.pi * Y / Ly_b)
    phi_hist = np.zeros((M_b + 1, Nx_b + 1, Ny_b + 1))
    for n in range(M_b + 1):
        phi_hist[n] = A_b * spatial * (1.0 + 0.2 * np.cos(2.0 * np.pi * t[n] / T_b))
    return phi_hist, x, y, t

@pytest.fixture(scope="module")
def small_problem_2d():
    Nx_b, Ny_b, M_b = 6, 5, 5
    phi_hist, x, y, t_hist = _make_synth_forward_2d(Nx_b, Ny_b, M_b)
    hx = x[1] - x[0]; hy = y[1] - y[0]
    L = laplacian_matrix_neumann(Nx_b, Ny_b, hx, hy)
    L2 = L @ L
    Nloc = (Nx_b + 1) * (Ny_b + 1)
    I = np.eye(Nloc)
    return dict(Nx=Nx_b, Ny=Ny_b, M=M_b,
                phi_hist=phi_hist, x=x, y=y, t_hist=t_hist,
                hx=hx, hy=hy, L=L, L2=L2, I=I)

def test_terminal_conditions_hold_relative_2d(small_problem_2d, cfg: ForwardSolverConfig):
    """Validate that the adjoint variables satisfy the terminal conditions.

    At the final time, ``p``, ``q`` and ``r`` must obey simple linear
    relationships derived from the optimality system.  This test constructs
    a synthetic forward solution and checks that the residuals of these
    conditions are below automatically estimated tolerances.  Specifically:

    * ``A_T p(T) = b2 (phi(T) - phi_T)`` where ``A_T = I - tau L``.
    * ``q(T) + L p(T) = 0``.
    * The auxiliary variable ``r`` vanishes at the final time.
    """
    P = small_problem_2d
    phi, x, y, t = P["phi_hist"], P["x"], P["y"], P["t_hist"]
    L, I = P["L"], P["I"]
    b1, b2 = 1.3, 0.7
    phi_Q = np.zeros_like(phi)
    phi_T = np.zeros((P["Nx"] + 1, P["Ny"] + 1))
    # Pass the forward solver configuration into run_backward as required by the new API.
    p, q, r = run_backward(phi, x, y, t, cfg, b1, b2, phi_Q, phi_T)
    pT = p[-1].ravel(); qT = q[-1].ravel(); rT = r[-1]
    rhs_T = b2 * (phi[-1].ravel() - phi_T.ravel())
    A_T = np.eye(I.shape[0]) - cfg.tau * L
    res_p = A_T @ pT - rhs_T
    tol_p, _ = auto_tol_from_cond(A_T, base=2e3)
    rel_p = rel_residual(res_p, A_T @ pT, rhs_T)
    res_q = qT + (L @ pT)
    tol_q = 5e2 * EPS
    rel_q = rel_residual(res_q, qT, L @ pT)
    assert rel_p < tol_p
    assert rel_q < tol_q
    assert np.linalg.norm(rT) < 1e-12

def test_p_step_discrete_equation_is_satisfied_relative_2d(small_problem_2d, cfg: ForwardSolverConfig):
    """Check the discrete Crank–Nicolson update for ``p``.

    For each interior time step ``n``, the adjoint variable ``p`` must satisfy a
    linear system ``A(phi_n) p_n = B(phi_{n+1}) p_{n+1} + source`` where the
    matrices ``A`` and ``B`` depend on the second derivative of the potential
    ``fpp_log`` evaluated at the forward states.  This test forms the matrices
    explicitly using dense arrays and verifies that the relative residuals of
    the linear equation are below tolerance for every step.  The tolerance is
    scaled by the condition number of ``A`` to avoid false positives.
    """
    P = small_problem_2d
    phi, x, y, t = P["phi_hist"], P["x"], P["y"], P["t_hist"]
    L, L2, I = P["L"], P["L2"], P["I"]
    b1, b2 = 1.3, 0.7
    phi_Q = np.zeros_like(phi)
    phi_T = np.zeros((P["Nx"] + 1, P["Ny"] + 1))
    # Use the new API: supply the ForwardSolverConfig object before b1 and b2.
    p, q, r = run_backward(phi, x, y, t, cfg, b1, b2, phi_Q, phi_T)
    Nloc = (P["Nx"] + 1) * (P["Ny"] + 1)
    for n in range(len(t) - 1):
        dt_n = t[n+1] - t[n]
        if dt_n <= 0: continue
        phi_n = phi[n].reshape(Nloc)
        phi_np1 = phi[n+1].reshape(Nloc)
        # fpp_log now requires c1 and c2 as parameters
        fpp_n = fpp_log(phi_n, cfg.c1, cfg.c2)
        fpp_np1 = fpp_log(phi_np1, cfg.c1, cfg.c2)
        A = I - cfg.tau * L + 0.5 * dt_n * L2 - 0.5 * dt_n * (np.diag(fpp_n) @ L.toarray())
        B = I - cfg.tau * L - 0.5 * dt_n * L2 + 0.5 * dt_n * (np.diag(fpp_np1) @ L.toarray())
        src = 0.5 * dt_n * b1 * ((phi[n].reshape(Nloc) - phi_Q[n].reshape(Nloc)) +
                                 (phi[n+1].reshape(Nloc) - phi_Q[n+1].reshape(Nloc)))
        left = A @ p[n].reshape(Nloc)
        right = B @ p[n+1].reshape(Nloc) + src
        res = left - right
        tol, _ = auto_tol_from_cond(A, base=2e3)
        rel = rel_residual(res, left, right)
        assert rel < tol

def test_q_equals_minus_Lp_all_steps_relative_2d(small_problem_2d):
    """Ensure the definition ``q = -L p`` holds at every time step.

    The adjoint variable ``q`` is analytically defined as the negative of the
    Laplacian applied to ``p``.  After calling ``run_backward`` we loop over
    all time levels and compute the relative residual of the vector equation
    ``q_n + L p_n = 0``.  All residuals are required to be below a small
    absolute threshold.
    """
    P = small_problem_2d
    phi, x, y, t = P["phi_hist"], P["x"], P["y"], P["t_hist"]
    L = P["L"]
    b1, b2 = 0.9, 0.4
    # Supply the default ForwardSolverConfig instance to run_backward
    # Use the default ForwardSolverConfig since cfg is not passed explicitly in this test
    cfg_local = ForwardSolverConfig()
    p, q, r = run_backward(phi, x, y, t, cfg_local, b1, b2, np.zeros_like(phi), np.zeros((P["Nx"] + 1, P["Ny"] + 1)))
    residuals = []
    for n in range(len(t)):
        p_n = p[n].reshape(-1)
        q_n = q[n].reshape(-1)
        res = q_n + L @ p_n
        residuals.append(rel_residual(res, q_n, L @ p_n))
    assert np.all(np.array(residuals) < 1e-9)

def test_r_crank_nicolson_equation_is_satisfied_relative_2d(small_problem_2d, cfg: ForwardSolverConfig):
    """Verify the Crank–Nicolson update for the auxiliary variable ``r``.

    The variable ``r`` satisfies an equation analogous to the primary adjoint
    variable but without the source term.  We compute the residual of this
    update for every time step and assert that it is essentially zero.  This
    guards against sign errors or incorrect averaging in the implementation.
    """
    P = small_problem_2d
    phi, x, y, t = P["phi_hist"], P["x"], P["y"], P["t_hist"]
    b1, b2 = 1.0, 0.6
    # Supply the ForwardSolverConfig to run_backward as per the new API
    p, q, r = run_backward(phi, x, y, t, cfg, b1, b2, np.zeros_like(phi), np.zeros((P["Nx"] + 1, P["Ny"] + 1)))
    residuals = []
    for n in range(len(t) - 1):
        dt_n = t[n+1] - t[n]
        if dt_n <= 0: continue
        r_n = r[n].reshape(-1)
        r_np1 = r[n+1].reshape(-1)
        q_n = q[n].reshape(-1)
        q_np1 = q[n+1].reshape(-1)
        left = -cfg.gamma * (r_np1 - r_n) / dt_n
        mid  = 0.5 * ((r_np1 - q_np1) + (r_n - q_n))
        residuals.append(rel_residual(left + mid, left, mid))
    assert np.all(np.array(residuals) < 1e-9)
    
def test_backward_reads_last10_reverse_real_forward_2d():
    """
    2D analog of the 1D 'backward_reads_last10_reverse_real_forward':
    - Run a forward simulation
    - Take the last 10 frames (φ_n)
    - Run the 2D adjoint backward to get p_n
    - Rebuild A(φ_n) and B(φ_{n+1}) and verify the discrete CN adjoint step:
         A(φ_n) p_n  =  B(φ_{n+1}) p_{n+1} + 0.5*dt*b1 * [(φ_n - φ_Qn) + (φ_{n+1} - φ_Q{n+1})]
      while the 'swapped' ordering (A at n+1, B at n) yields a much larger residual.
    """
    import numpy as np
    import scipy.sparse as sps
    from numpy.linalg import norm
    from config import ForwardSolverConfig
    from Forward2_solver import laplacian_matrix_neumann
    from backward2_solver import run_backward, fpp_log
    from Forward2_solver import trapz_weights  # not strictly needed, just for completeness

    # --- Small, fast 2D forward run ---
    cfg = ForwardSolverConfig(Nx=32, Ny=32, Lx=1.0, Ly=1.0, T=0.10, dt_initial=1e-3,
                              tau=0.05, gamma=10.0, c1=0.75, c2=1.0, kappa=0.03**2)

    from Forward2_solver import run_main_simulation
    phi_hist, (x, y), t_hist = run_main_simulation(cfg, store_history=True, verbose=False)

    # Use the last 10 frames (or fewer if short run)
    K = min(10, len(t_hist))
    phi10 = phi_hist[-K:]
    t10   = t_hist[-K:]

    Nx, Ny = phi10.shape[1] - 1, phi10.shape[2] - 1
    hx, hy = x[1] - x[0], y[1] - y[0]
    Nloc   = (Nx + 1) * (Ny + 1)

    # --- Run the backward (adjoint) solve on these K frames ---
    b1, b2 = 1.0, 0.7
    phi_Q  = np.zeros_like(phi10)
    phi_T  = np.zeros_like(phi10[-1])
    p, q, r = run_backward(phi10, x, y, t10, cfg, b1, b2, phi_Q, phi_T)  # shapes: (K, Nx+1, Ny+1)

    # --- Build spatial operators ---
    L  = laplacian_matrix_neumann(Nx, Ny, hx, hy).tocsr()
    L2 = (L @ L).tocsr()
    I  = sps.eye(Nloc, format="csr")

    def vec(A2d):  # flatten (Nx+1, Ny+1) -> (Nloc,) for convenience
        """Reshape a 2‑D field into a 1‑D vector with row–major ordering."""
        return A2d.reshape(Nloc)

    def A_mat(phi_2d, dt):
        """
        Assemble the left‐hand matrix for the adjoint Crank–Nicolson step.

        For a given forward state ``phi_2d`` and time step ``dt``, the matrix
        ``A(phi_n)`` is defined as

            A = I − τ L + 0.5 dt L² − 0.5 dt (diag(f''(phi_n)) L)

        where ``fpp_log`` computes the second derivative of the logarithmic
        potential at each grid point.  All operators are represented in
        flattened CSR sparse format.
        """
        fpp = fpp_log(phi_2d, cfg.c1, cfg.c2)  # second derivative f''
        D   = sps.diags(vec(fpp), 0, shape=(Nloc, Nloc), format="csr")
        return (I - cfg.tau * L + 0.5 * dt * L2 - 0.5 * dt * (D @ L)).tocsr()

    def B_mat(phi_2d, dt):
        """
        Assemble the right‐hand matrix for the adjoint Crank–Nicolson step.

        This matrix uses the forward state at time ``n+1`` and has the form

            B = I − τ L − 0.5 dt L² + 0.5 dt (diag(f''(phi_{n+1})) L).
        """
        fpp = fpp_log(phi_2d, cfg.c1, cfg.c2)
        D   = sps.diags(vec(fpp), 0, shape=(Nloc, Nloc), format="csr")
        return (I - cfg.tau * L - 0.5 * dt * L2 + 0.5 * dt * (D @ L)).tocsr()

    def rel_residual(res, left, right):
        """Compute a relative residual using the norms of the operands.

        To avoid division by zero, a small epsilon is added to the denominator.
        """
        num = norm(res)
        den = norm(left) + norm(right) + 1e-30
        return num / den

    # --- Check every backward step on this window ---
    print("\n--- Testing 2D Backward Solver Direction ---")
    for i in range(K - 1):
        dt_i = float(t10[i + 1] - t10[i])
        # Correct Crank–Nicolson form: A_i p_i = B_{i+1} p_{i+1} + source
        Ai   = A_mat(phi10[i],   dt_i)
        Bip1 = B_mat(phi10[i+1], dt_i)

        # Source term comes from the tracking component of the cost
        src = 0.5 * dt_i * b1 * (vec(phi10[i] - phi_Q[i]) + vec(phi10[i+1] - phi_Q[i+1]))

        left_corr  = Ai @ vec(p[i])
        right_corr = Bip1 @ vec(p[i+1]) + src
        rel_corr   = rel_residual(left_corr - right_corr, left_corr, right_corr)

        # Swapped ordering intentionally misplaces A and B; this should worsen the residual.
        Aip1 = A_mat(phi10[i+1], dt_i)
        Bi   = B_mat(phi10[i],   dt_i)
        left_swap  = Aip1 @ vec(p[i])
        right_swap = Bi   @ vec(p[i+1]) + src
        rel_swap   = rel_residual(left_swap - right_swap, left_swap, right_swap)

        print(f"  Step {i}: correct residual = {rel_corr:.3e}, swapped = {rel_swap:.3e}")
        # The correct ordering should yield very small residuals; a ratio of >100
        # between the swapped and correct residuals demonstrates the sensitivity
        # to the proper operator ordering in the adjoint equation.
        assert rel_corr < 5e-7, f"Adjoint CN step residual too large at step {i}: {rel_corr:.2e}"
        assert (rel_swap + 1e-30) / (rel_corr + 1e-30) > 1e2, \
            f"Swapped/Correct residual ratio too small at step {i}"

    
if __name__ == "__main__":
    # Run pytest programmatically
    import sys
    pytest.main([__file__, "-v","-s", "--tb=short"])