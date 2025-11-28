"""
backward2_solver.py
====================

This module contains the numerical routines to solve the adjoint system
associated with a 2D Cahn–Hilliard optimal control problem.  In the context
of gradient-based optimization, the adjoint variables (``p``, ``q`` and ``r``)
provide sensitivities of the cost functional with respect to the state
variables and the control.  Solving the adjoint system backward in time is
essential for computing gradients efficiently.

The implementation discretizes space on a uniform grid with Neumann (no-flux)
boundary conditions and uses a Crank–Nicolson (CN) discretization in time.
Second derivatives of the logarithmic potential are safely regularized to
avoid numerical blow‑ups near |φ| ≈ 1.  The solver accepts a precomputed
forward history ``phi_hist`` and optionally target trajectories and terminal
targets.
"""

from typing import Optional, Tuple
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

# Optional numba acceleration (safe fallback if not installed)
try:
    import numba
    njit = numba.njit
except Exception:
    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco

# --- Import from your project ---
from config import ForwardSolverConfig
from Forward2_solver import laplacian_matrix_neumann


@njit(cache=True)
def fpp_log(phi: np.ndarray, c1: float, c2: float, eps: float = 1e-8) -> np.ndarray:
    """Compute the second derivative f''(φ) of the double‑well potential.

    The underlying free energy contains a convex–concave splitting with a
    logarithmic term.  Taking the second derivative yields

        f''(φ) = 2 * c1 / (1 - φ^2)  -  2 * c2,

    which diverges as φ → ±1.  To prevent division by zero and preserve
    numerical stability, the input ``phi`` is clipped to lie within
    (−1 + eps, 1 − eps) before applying the formula.  A small positive
    ``eps`` is used as a safety margin.

    Parameters
    ----------
    phi : ndarray
        Array of phase–field values.
    c1, c2 : float
        Coefficients controlling the convex and concave parts of the free
        energy.
    eps : float, optional
        Safety tolerance to avoid evaluating the derivative too close to
        |φ| = 1.  Defaults to 1e‑8.

    Returns
    -------
    ndarray
        The elementwise second derivative values.
    """
    # Clip φ into the open interval (−1 + eps, 1 − eps) to avoid singularities
    ph = np.clip(phi, -1.0 + eps, 1.0 - eps)
    return 2.0 * c1 / (1.0 - ph * ph) - 2.0 * c2


def run_backward(
    phi_hist: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t_hist: np.ndarray,
    config: ForwardSolverConfig,
    b1: float,
    b2: float,
    phi_Q: Optional[np.ndarray] = None,
    phi_T_target: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the adjoint equations backward in time.

    Given the forward state history ``phi_hist`` and a simulation grid, this
    routine solves for three adjoint variables:

    - ``p``: the primary adjoint associated with the phase–field equation;
    - ``q``: an auxiliary variable equal to ``−L p``; this is computed to
      avoid repeated sparse matrix multiplications during the time loop;
    - ``r``: a filtered version of ``q`` solving ``γ r_t + r = q`` via
      Crank–Nicolson (backward) time stepping.  The filtered variable arises
      from the control regularization in the cost functional.

    The adjoint system is linear in ``p`` and depends on the second
    derivative of the free energy, which is evaluated by :func:`fpp_log`.
    Time integration proceeds backward from t = T to t = 0, aligning with
    the usual adjoint formulation.  When the step size ``dt_n`` is very
    small (≲1e‑14), the solver simply copies the adjoint state from the
    subsequent time level to avoid numerical issues.

    Parameters
    ----------
    phi_hist : ndarray, shape (M+1, Nx+1, Ny+1)
        History of the phase–field φ from the forward simulation.
    x, y : 1D ndarrays
        Spatial grids in x and y directions.
    t_hist : 1D ndarray
        Time discretization points corresponding to the rows of ``phi_hist``.
    config : ForwardSolverConfig
        Configuration with parameters (τ, γ, c₁, c₂, κ) and grid sizes.
    b1, b2 : float
        Weights of the space–time tracking cost and terminal cost in the
        optimization functional.
    phi_Q, phi_T_target : ndarray or None, optional
        Desired trajectories (space–time) and terminal state.  If ``None``,
        zero trajectories are assumed.

    Returns
    -------
    (p, q, r) : tuple of ndarrays
        Arrays of shape (M+1, Nx+1, Ny+1) containing the adjoint variables.
    """
    # --- Unpack parameters ---
    # Convert config attributes to Python floats to avoid dtype surprises and for
    # consistency when passing into numba functions.  These quantities
    # represent physical parameters of the Cahn–Hilliard model.
    tau = float(config.tau)
    gamma = float(config.gamma)
    c1 = float(config.c1)
    c2 = float(config.c2)
    kappa = float(config.kappa)

    # --- Basic shape checks ---
    # Perform sanity checks on input array dimensions to catch user errors
    # early.  ``phi_hist`` should be a 3D array with time as the first
    # dimension; ``x`` and ``y`` should be 1D arrays of node coordinates.
    assert phi_hist.ndim == 3, "phi_hist must be (M+1, Nx+1, Ny+1)"
    M_plus1, Nx1, Ny1 = phi_hist.shape
    assert x.ndim == 1 and y.ndim == 1, "x and y must be 1D arrays"
    assert x.size >= 2 and y.size >= 2, "x and y must have at least 2 points"
    assert t_hist.ndim == 1 and t_hist.shape[0] == M_plus1, "t_hist must align with phi_hist"

    # --- Grid & operators ---
    # Compute grid spacings and assemble the discrete Laplacian operator L
    # (with Neumann boundary conditions).  ``L2`` is the biharmonic operator
    # used in the adjoint equation.  ``I`` is the identity matrix.  All
    # matrices are stored in CSR format for efficient arithmetic and solves.
    Nx = Nx1 - 1
    Ny = Ny1 - 1
    hx = float(x[1] - x[0])
    hy = float(y[1] - y[0])
    Nloc = (Nx + 1) * (Ny + 1)

    L = laplacian_matrix_neumann(Nx, Ny, hx, hy).tocsr()
    L2 = (L @ L).tocsr()
    I = sps.eye(Nloc, format="csr")

    # --- Flattened history/targets ---
    # Reshape 3D arrays to 2D (time × space) to facilitate matrix operations.
    # If the user did not specify targets, assume zero trajectories and
    # terminal state.
    phi_hist_f = phi_hist.reshape(M_plus1, Nloc)
    phi_Q_f = np.zeros_like(phi_hist_f) if phi_Q is None else phi_Q.reshape(M_plus1, Nloc)
    phi_T_f = np.zeros(Nloc) if phi_T_target is None else phi_T_target.reshape(Nloc)

    # --- Allocate adjoints (flattened) ---
    # Preallocate arrays for the adjoint variables in flattened form.  These
    # arrays store (p, q, r) at each time level; we reshape them back to
    # (M+1, Nx+1, Ny+1) at the end of the function.
    p = np.zeros((M_plus1, Nloc), dtype=np.float64)
    q = np.zeros_like(p)
    r = np.zeros_like(p)

    # --- Terminal condition at t = T ---
    # Solve for ``p`` at the final time step T using the terminal cost.  The
    # adjoint equation for ``p`` is linear and of the form (I − τL) p = rhs.
    # Once ``p`` is obtained, compute q = −L p.  The filtered variable r at
    # t=T is initialized to zero (homogeneous terminal condition).
    rhs_T = b2 * (phi_hist_f[-1] - phi_T_f)
    A_T = (I - tau * L).tocsc()
    p[-1] = spsolve(A_T, rhs_T)
    q[-1] = -(L @ p[-1])
    r[-1] = 0.0

    # --- Adjoint system matrices (Crank–Nicolson) ---
    # Define helper functions that assemble the Crank–Nicolson matrices for
    # the adjoint update.  ``A_adjoint`` corresponds to the implicit part of
    # the CN discretization and involves f''(φ_n); ``B_adjoint`` corresponds
    # to the explicit part involving f''(φ_{n+1}).  Both return sparse CSR
    # matrices of size (Nloc × Nloc).
    def A_adjoint(phi_n: np.ndarray, dt_n: float) -> sps.csr_matrix:
        fpp_vals = fpp_log(phi_n, c1, c2)
        D = sps.diags(fpp_vals, 0, shape=(Nloc, Nloc), format="csr")
        return (I - tau * L + 0.5 * dt_n *  L2 - 0.5 * dt_n * (D @ L)).tocsr()

    def B_adjoint(phi_np1: np.ndarray, dt_n: float) -> sps.csr_matrix:
        fpp_vals = fpp_log(phi_np1, c1, c2)
        D = sps.diags(fpp_vals, 0, shape=(Nloc, Nloc), format="csr")
        return (I - tau * L - 0.5 * dt_n *  L2 + 0.5 * dt_n * (D @ L)).tocsr()

    # --- Backward march ---
    # Main backward time loop: iterate from n = M−1 down to 0.  For each
    # interval [t_n, t_{n+1}] with length dt_n, we compute the trapezoidal
    # approximation of the source term (due to tracking cost), assemble and
    # solve the linear system for ``p[n]``, compute ``q[n]``, and update
    # ``r[n]`` via a CN step.  Very small time steps are skipped by simply
    # copying the adjoint state from the next time level.
    for n in range(M_plus1 - 2, -1, -1):
        dt_n = float(t_hist[n + 1] - t_hist[n])
        if dt_n <= 1e-14:
            p[n], q[n], r[n] = p[n + 1], q[n + 1], r[n + 1]
            continue

        # Trapezoidal source from running cost: this term corresponds to
        # the derivative of the cost ∫ b1/2 |φ − φ_Q|^2 dt with respect
        # to φ.  Using the trapezoidal rule we average the source at
        # t_n and t_{n+1}.
        src = 0.5 * dt_n * b1 * (
            (phi_hist_f[n] - phi_Q_f[n]) + (phi_hist_f[n + 1] - phi_Q_f[n + 1])
        )

        rhs = (B_adjoint(phi_hist_f[n + 1], dt_n) @ p[n + 1]) + src
        A = A_adjoint(phi_hist_f[n], dt_n).tocsc()
        try:
            p[n] = spsolve(A, rhs)
        except Exception:
            p[n] = spsolve(A + 1e-10 * sps.eye(Nloc, format="csc"), rhs)

        q[n] = -(L @ p[n])

        # Backward Crank–Nicolson update for the filtered adjoint ``r``:
        # the ODE γ r_t + r = q is integrated backward in time using
        # CN, yielding a simple recursion for r[n] in terms of r[n+1] and
        # the values of q at t_n and t_{n+1}.
        denom = (gamma + 0.5 * dt_n)
        gamma_factor_back = (gamma - 0.5 * dt_n) / denom
        gamma_factor_src = (0.5 * dt_n) / denom
        r[n] = gamma_factor_back * r[n + 1] + gamma_factor_src * (q[n] + q[n + 1])

    # Reshape back
    shape3d = (M_plus1, Nx + 1, Ny + 1)
    return p.reshape(shape3d), q.reshape(shape3d), r.reshape(shape3d)
