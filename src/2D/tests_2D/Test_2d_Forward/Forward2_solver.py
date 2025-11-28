"""
Forward2_solver.py
===================

This module implements a forward solver for the two‑dimensional
Cahn–Hilliard equation with inertial relaxation and control forcing.  The
solver supports Neumann boundary conditions on a rectangular domain, employs
a Crank–Nicolson time discretization for both φ and the chemical potential
μ, and maintains mass conservation via correction steps.  It also includes
utilities for initialization, evaluating stability, and computing free
energy.

Key features of the solver:
  - Sparse matrix assembly of the Laplacian using Kronecker products.
  - Regularization of logarithmic terms to avoid singularities near |φ|=1.
  - Newton–Raphson iteration with backtracking line search and step size
    ceiling to enforce the physical bound |φ| < 1.
  - Optional storage of the full time history for use by the adjoint
    solver and optimization routines.
  - Mass conservation enforced at each time step by redistributing
    deviations of the weighted integral of φ.

The solver functions here are imported by the gradient descent driver in
``GD2_configured.py`` and by the adjoint solver.  Standalone execution
provides an interactive interface to run a forward simulation and display
the final state.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # kept for compatibility (not required)
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

# --- Optional Numba acceleration (safe fallback if unavailable) ---
try:
    import numba
    njit = numba.njit
except Exception:
    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco

from config import ForwardSolverConfig, load_params, get_user_input_for_config

# --- Constants ---
DEBUG = True
COMPUTE_ENERGY = False  # Set to True for energy checks
ENERGY_EVERY_N_STEPS = 100

# === CORE FUNCTIONS ===
def instability_report(c1, c2, kappa, tau, Lx, Nmodes=12):
    """
    Analyze the linear stability of the homogeneous solution.

    Computes the dispersion relation λ(k) for perturbations e^{ikx} on a
    one‑dimensional slice of the 2D domain (assuming separable modes),
    returning the eigenvalues for the first ``Nmodes`` wavenumbers.  A
    positive λ indicates an unstable mode.  The function prints a brief
    summary including the maximum eigenvalue and the number of unstable
    modes.

    Parameters
    ----------
    c1, c2, kappa, tau : float
        Physical parameters of the Cahn–Hilliard model.
    Lx : float
        Length of the domain in the x‑direction.
    Nmodes : int
        Number of wavenumber modes to evaluate (default 12).

    Returns
    -------
    ndarray
        Eigenvalues λ(k) for k = 1, …, Nmodes.
    """
    a = 2 * (c1 - c2)
    ks = np.pi * np.arange(1, Nmodes + 1) / Lx
    q = ks**2
    lam = (-kappa * q**2 - a * q) / (1 + tau * q)
    print(f"a={a:.3g},  max λ={lam.max():.3g} at mode n={lam.argmax()+1},  unstable modes={(lam>0).sum()}")
    return lam


@njit(cache=True)
def regularized_log(phi: np.ndarray, delta_sep: float) -> np.ndarray:
    """Compute a safely regularized logarithm.

    The Flory–Huggins bulk free energy uses terms of the form

        ln((1+φ)/(1−φ)),

    which become singular as φ approaches ±1.  This helper clips the
    argument φ to remain inside (−1 + ε, 1 − ε) where ε is chosen as
    ``max(1e-8, 0.5*delta_sep)``.  The returned value is the natural
    logarithm of the ratio (1+φ)/(1−φ).
    """
    eps = max(1e-8, 0.5 * delta_sep)
    # FIX: upper bound must be 1 - eps (not 1 + eps)
    phi_s = np.clip(phi, -1.0 + eps, 1.0 - eps)
    return np.log((1.0 + phi_s) / (1.0 - phi_s))


def laplacian_matrix_neumann_1d(N: int, h: float) -> sps.csr_matrix:
    """
    Construct a sparse 1D Laplacian matrix with Neumann boundary conditions.

    The matrix corresponds to the second derivative operator discretized
    using second‑order centered finite differences.  To enforce Neumann
    conditions, the off‑diagonals at the boundaries are doubled, which
    effectively mirrors the stencil outside the domain.  The returned
    matrix has shape (N+1, N+1).
    """
    a = 1.0 / (h * h)
    main = -2.0 * a * np.ones(N + 1)
    off = 1.0 * a * np.ones(N)
    L = sps.diags([off, main, off], offsets=[-1, 0, 1], shape=(N + 1, N + 1), format="lil")
    # Neumann BC via mirrored stencil
    L[0, 1] = 2.0 * a
    L[N, N - 1] = 2.0 * a
    return L.tocsr()


def laplacian_matrix_neumann(Nx: int, Ny: int, hx: float, hy: float) -> sps.csr_matrix:
    """Assemble the 2D Laplacian with Neumann boundary conditions.

    Uses Kronecker products of 1D Laplacian matrices in x and y to
    efficiently build a sparse matrix representing Δ under Neumann
    boundary conditions in both directions.  The resulting operator acts
    on flattened (Nx+1)×(Ny+1) fields.
    """
    Lx_op = laplacian_matrix_neumann_1d(Nx, hx)
    Ly_op = laplacian_matrix_neumann_1d(Ny, hy)
    Ix = sps.eye(Nx + 1, format="csr")
    Iy = sps.eye(Ny + 1, format="csr")
    return sps.kron(Iy, Lx_op) + sps.kron(Ly_op, Ix)


def apply_laplacian(L: sps.csr_matrix, v: np.ndarray, Nx: int, Ny: int) -> np.ndarray:
    """
    Apply the precomputed Laplacian matrix to a 2D field.

    The input field ``v`` is flattened, multiplied by the sparse matrix ``L``,
    and reshaped back to its original (Nx+1, Ny+1) shape.  This helper
    centralizes the flattening and reshaping logic used throughout the solver.
    """
    if v.ndim != 2 or v.shape != (Nx + 1, Ny + 1):
        raise ValueError(f"Input field must have shape ({Nx+1}, {Ny+1})")
    v_flat = v.ravel(order="C")
    out = L @ v_flat
    return out.reshape((Nx + 1, Ny + 1), order="C")


def initialize_mu(phi: np.ndarray, w: np.ndarray, c1: float, c2: float, kappa: float,
                  L: sps.csr_matrix, Nx: int, Ny: int, delta_sep: float) -> np.ndarray:
    """
    Compute an initial guess for the chemical potential μ.

    Given an initial phase field ``phi`` and auxiliary variable ``w``, the
    chemical potential μ is defined as μ = -κΔφ + f'(φ) - w, where f'(φ)
    derives from the Flory–Huggins free energy.  This function evaluates
    these terms using the precomputed Laplacian matrix.
    """
    lap_phi = apply_laplacian(L, phi, Nx, Ny)
    f_prime = c1 * regularized_log(phi, delta_sep) - 2.0 * c2 * phi
    return -kappa * lap_phi + f_prime - w


@njit(cache=True)
def solve_w(w_old: np.ndarray, dt: float, gamma: float, u_n: np.ndarray, u_np1: np.ndarray) -> np.ndarray:
    """
    Solve for the auxiliary variable ``w`` using Crank–Nicolson.

    The auxiliary variable satisfies the linear ODE γ w_t + w = u, which is
    discretized over one time step ``dt`` via Crank–Nicolson.  The scheme
    solves for w_{n+1} in closed form given w_n and control inputs u_n,
    u_{n+1}.
    """
    gamma_dt = gamma / dt
    return ((gamma_dt - 0.5) * w_old + 0.5 * (u_np1 + u_n)) / (gamma_dt + 0.5)


def solve_mu_residual(phi_new: np.ndarray, phi_old: np.ndarray,
                      mu_new: np.ndarray, mu_old: np.ndarray,
                      dt: float, L: sps.csr_matrix, Nx: int, Ny: int) -> np.ndarray:
    """
    Residual of the μ equation in the coupled Newton system.

    Computes the discretized residual of (φ_t) − Δμ = 0 using
    Crank–Nicolson.  A zero residual implies the time discretization is
    satisfied at the current guess.  Used within the Newton iteration.
    """
    lap_new = apply_laplacian(L, mu_new, Nx, Ny)
    lap_old = apply_laplacian(L, mu_old, Nx, Ny)
    return (phi_new - phi_old) / dt - 0.5 * (lap_new + lap_old)


def solve_phi_residual(phi_new: np.ndarray, phi_old: np.ndarray,
                       mu_new: np.ndarray, mu_old: np.ndarray,
                       w_new: np.ndarray, w_old: np.ndarray,
                       dt: float, tau: float, c1: float, c2: float, kappa: float,
                       L: sps.csr_matrix, Nx: int, Ny: int, delta_sep: float) -> np.ndarray:
    """
    Residual of the φ equation in the coupled Newton system.

    Implements the convex–concave splitting scheme for φ and evaluates
    (τ (φ_{n+1} − φ_n)/dt) − (κ/2)(Δφ_{n+1} + Δφ_n) + f_cvx(φ_{n+1}) + f_ccv(φ_n)
    − (μ_{n+1} + μ_n)/2 − (w_{n+1} + w_n)/2.  Returning a tensor of zeros
    indicates that the nonlinear φ equation is satisfied at the current
    iterate.
    """
    lap_new = apply_laplacian(L, phi_new, Nx, Ny)
    lap_old = apply_laplacian(L, phi_old, Nx, Ny)

    f_cvx = c1 * regularized_log(phi_new, delta_sep)   # implicit convex part
    f_ccv = -2.0 * c2 * phi_old                        # explicit concave part
    mu_avg = 0.5 * (mu_new + mu_old)
    w_avg = 0.5 * (w_new + w_old)

    return (tau * (phi_new - phi_old) / dt) - 0.5 * kappa * (lap_new + lap_old) + (f_cvx + f_ccv) - mu_avg - w_avg


def assemble_jacobian(phi_new: np.ndarray, dt: float, tau: float, c1: float,
                      kappa: float, L: sps.csr_matrix, delta_sep: float) -> sps.csr_matrix:
    """
    Assemble the Jacobian matrix of the coupled (φ, μ) system.

    The Newton method requires the derivative of the residual with respect
    to the unknowns (φ_new, μ_new).  This function builds a 2×2 block
    sparse matrix corresponding to the derivative of the residual vector
    [R_phi; R_mu].  Diagonal additions ensure invertibility when φ is
    near ±1 by clipping φ² in the denominator.
    """
    phi_flat = phi_new.ravel()
    Nloc = phi_flat.size
    t = tau / dt
    s = 1.0 / dt

    # K_phi_phi block
    Kpp = (-0.5 * kappa) * L.tocsr()
    # Safety clip so denominator (1 - phi^2) stays >= delta_sep^2
    phi_sq = np.clip(phi_flat ** 2, 0.0, 1.0 - delta_sep ** 2)
    diag_add = t + 2.0 * c1 / (1.0 - phi_sq)
    Kpp.setdiag(Kpp.diagonal() + diag_add)

    # Other blocks
    I = sps.eye(Nloc, format="csr")
    Kpm = -0.5 * I           # d R_phi / d mu_new
    Kmp = s * I              # d R_mu  / d phi_new
    Kmm = -0.5 * L.tocsr()   # d R_mu  / d mu_new

    return sps.bmat([[Kpp, Kpm], [Kmp, Kmm]], format="csr")


def free_energy(phi, kappa, c1, c2, hx, hy, w=None, eps=None):
    """
    Compute the discrete free energy of a phase field.

    The Cahn–Hilliard free energy functional is discretized on a uniform
    grid (Ny+1 by Nx+1) as follows:

        E = ∫Ω [ (κ/2)|∇φ|² + c1[(1+φ)ln(1+φ) + (1−φ)ln(1−φ)] − c2 φ² ] dΩ
            − ∫Ω w φ dΩ,

    where ∇φ is approximated by forward differences.  Integration is
    performed using 2D trapezoidal weights ``wts_2d``.  The optional
    coupling term −∫ w φ is included when ``w`` is provided.

    Parameters
    ----------
    phi : ndarray
        Phase field array of shape (Ny+1, Nx+1).
    kappa, c1, c2 : float
        Physical parameters of the energy functional.
    hx, hy : float
        Grid spacings in x and y directions.
    w : ndarray, optional
        Coupling field; if provided, subtracts ∫ w φ from the energy.
    eps : float, optional
        Regularization for the logarithmic terms; default 1e‑8.

    Returns
    -------
    float
        The discrete free energy value.
    """
    if eps is None:
        eps = 1e-8

    phi2d = np.asarray(phi)                 # shape (Ny+1, Nx+1)
    Ny, Nx = phi2d.shape[0] - 1, phi2d.shape[1] - 1

    # trapezoidal weights
    wts_y = trapz_weights(Ny + 1)
    wts_x = trapz_weights(Nx + 1)
    wts_2d = np.outer(wts_y, wts_x)

    # gradient term: forward differences
    dphi_y = np.diff(phi2d, axis=0)         # (Ny,   Nx+1)
    dphi_x = np.diff(phi2d, axis=1)         # (Ny+1, Nx)
    E_grad_y = (kappa / (2.0 * hy)) * np.sum(dphi_y**2) * hx
    E_grad_x = (kappa / (2.0 * hx)) * np.sum(dphi_x**2) * hy
    E_grad = E_grad_x + E_grad_y

    # bulk term with safe logs
    phi_s = np.clip(phi2d, -1.0 + eps, 1.0 - eps)
    psi = c1 * ((1.0 + phi_s) * np.log(1.0 + phi_s) + (1.0 - phi_s) * np.log(1.0 - phi_s)) \
          - c2 * (phi_s**2)
    E_bulk = hx * hy * np.sum(wts_2d * psi)

    E = E_grad + E_bulk

    # optional coupling −∫ w φ
    if w is not None:
        w2d = np.asarray(w)
        E -= hx * hy * np.sum(wts_2d * w2d * phi2d)

    return E



def newton_raphson(phi_old: np.ndarray, mu_old: np.ndarray, w_old: np.ndarray, w_new: np.ndarray,
                   dt: float, tau: float, c1: float, c2: float, kappa: float, delta_sep: float,
                   L: sps.csr_matrix, Nx: int, Ny: int, hx: float, hy: float,return_residual_history=False) -> (np.ndarray, np.ndarray):
    """
    Solve the coupled nonlinear system for (φ_{n+1}, μ_{n+1}) via Newton–Raphson.

    At each time step of the implicit discretization, we must solve the
    nonlinear equations for φ and μ simultaneously.  This function
    implements a Newton–Raphson iteration with several safeguards:

    - **Step ceiling**: ensures φ stays within the admissible bounds
      (−1 + delta_sep, 1 − delta_sep) by limiting the update along dφ.
    - **Armijo backtracking**: reduces the Newton step size α until the
      residual norm decreases sufficiently.  The best trial step is saved
      in case no step satisfies the Armijo condition.
    - **Biharmonic regularization**: small diagonal shifts are added to the
      Jacobian in case it is near singular.

    Optional return of the residual norm history can aid in diagnosing
    convergence behavior.

    Returns
    -------
    (phi_new, mu_new [, history]) : tuple
        Updated fields, and optionally the norm of the residual at each
        Newton iteration if ``return_residual_history`` is True.
    """
    phi_new = phi_old.copy()
    mu_new  = initialize_mu(phi_old, w_new, c1, c2, kappa, L, Nx, Ny, delta_sep).copy()

    tol, max_iter = 1e-6, 500
    Nloc = phi_old.size
    hist = []
    for k in range(max_iter):
        res_phi = solve_phi_residual(phi_new, phi_old, mu_new, mu_old, w_new, w_old, dt, tau, c1, c2, kappa, L, Nx, Ny, delta_sep)
        res_mu = solve_mu_residual(phi_new, phi_old, mu_new, mu_old, dt, L, Nx, Ny)
        R = np.concatenate([res_phi.ravel(), res_mu.ravel()])
        norm_R = np.linalg.norm(R)
        hist.append(norm_R)
        
        #print(norm_R)
        if norm_R < tol:
            return (phi_new, mu_new, hist) if return_residual_history else (phi_new, mu_new)

        J = assemble_jacobian(phi_new, dt, tau, c1, kappa, L, delta_sep)

        try:
            delta = spsolve(J.tocsc(), -R)
        except Exception:
            delta = spsolve((J + 1e-10 * sps.eye(J.shape[0], format="csr")).tocsc(), -R)

        dphi, dmu = delta[:Nloc], delta[Nloc:]

        # Step ceiling to keep phi strictly within (-1 + delta_sep, 1 - delta_sep)
        alpha = 1.0
        phi_new_flat = phi_new.ravel(order="C")

        with np.errstate(divide="ignore", invalid="ignore"):
            pos_mask = dphi > 0
            neg_mask = dphi < 0
            alpha_max = 2.0
            if np.any(pos_mask):
                alpha_max = min(alpha_max, 0.9 * np.min((1.0 - delta_sep - phi_new_flat[pos_mask]) / dphi[pos_mask]))
            if np.any(neg_mask):
                alpha_max = min(alpha_max, 0.9 * np.min((-1.0 + delta_sep - phi_new_flat[neg_mask]) / dphi[neg_mask]))

        if not np.isfinite(alpha_max) or alpha_max <= 0:
            alpha_max = 1
        alpha = min(1.0, alpha_max)

        # Armijo backtracking on residual norm
        eta = 1e-4
        best_norm = np.inf
        best_phi, best_mu = phi_new, mu_new
        accepted = False
        for _ in range(12):
            phi_t = phi_new + alpha * dphi.reshape(phi_new.shape)
            mu_t  = mu_new  + alpha * dmu.reshape(mu_new.shape)

            res_phi_t = solve_phi_residual(phi_t, phi_old, mu_t, mu_old,
                                           w_new, w_old, dt, tau, c1, c2, kappa,
                                           L, Nx, Ny, delta_sep)
            res_mu_t  = solve_mu_residual(phi_t, phi_old, mu_t, mu_old,
                                          dt, L, Nx, Ny)
            R_t = np.concatenate([res_phi_t.ravel(), res_mu_t.ravel()])
            norm_R_t = np.linalg.norm(R_t)

            # track best trial in case none pass Armijo
            if norm_R_t < best_norm:
                best_norm = norm_R_t
                best_phi, best_mu = phi_t, mu_t

            if norm_R_t <= (1.0 - eta * alpha) * norm_R:
                phi_new, mu_new = phi_t, mu_t
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            # fallback to best found step if backtracking never accepted
            if best_norm < norm_R:
                phi_new, mu_new = best_phi, best_mu
            # else leave (phi_new, mu_new) unchanged and continue; another Newton
            # sweep (with a fresh Jacobian) often succeeds.
            
    return (phi_new, mu_new, hist) if return_residual_history else (phi_new, mu_new)  # Max iterations reached


@njit(cache=True)
def trapz_weights(n_nodes: int) -> np.ndarray:
    """
    Compute the weights for the 1D trapezoidal rule.

    Returns a vector of ones except for the first and last elements,
    which are ½.  Multiplying a function vector by these weights and
    summing approximates the integral via the trapezoidal rule.
    """
    w = np.ones(n_nodes)
    w[0], w[-1] = 0.5, 0.5
    return w


def init_phi_random(Nx: int, Ny: int, delta_sep: float, amp: float = 0.5, seed: int = 42, enforce_zero_mean: bool = True) -> np.ndarray:
    """
    Initialize the phase field φ with random noise and optional zero mean.

    A random Gaussian field of amplitude ``amp`` is generated on the
    (Nx+1)×(Ny+1) grid.  Optionally, its weighted integral is projected
    to zero using trapezoidal weights to enforce mass conservation.  The
    resulting field is clipped to stay within (−1 + delta_sep, 1 − delta_sep).
    An iterative interior correction can be applied to fine‑tune the
    zero‑mean condition without altering boundary values.
    """
    rng = np.random.default_rng(seed)
    phi0 = amp * rng.standard_normal((Nx + 1, Ny + 1))

   
    wts_x = trapz_weights(Nx + 1)
    wts_y = trapz_weights(Ny + 1)
    wts = np.outer(wts_x, wts_y)
    Wtot = np.sum(wts)
    if enforce_zero_mean:
        m0 = np.sum(wts * phi0) / Wtot
        phi0 -= m0
    lo, hi = -1.0 + delta_sep, 1.0 - delta_sep
    phi0 = np.clip(phi0, lo, hi)
    if enforce_zero_mean:
        # 3) Mass-preserving interior correction (avoid changing saturated points)
        margin = 5e-3  # keep some distance from the bounds
        for _ in range(8):
            M = np.sum(wts * phi0)  # weighted mass error
            if abs(M) <= 1e-14 * Wtot:
                break
            interior = np.abs(phi0) < (hi - margin)
            Wint = float(np.sum(wts[interior]))
            if Wint <= 0:
                # fallback: tiny uniform correction then reclip
                phi0 -= (M / Wtot)
                phi0 = np.clip(phi0, lo, hi)
                break
            # subtract only on interior nodes; stays inside because of the margin
            phi0[interior] -= (M / Wint)
        # no final clip here (preserve zero-mean we just enforced)

    return phi0


def run_main_simulation(config: ForwardSolverConfig, store_history: bool = False,
                        control_input: np.ndarray = None, verbose: bool = True):
    """
    Time‑march the Cahn–Hilliard system with optional control forcing.

    Given a configuration ``config`` and an optional control history
    ``control_input``, this function advances the 2D Cahn–Hilliard system
    from t=0 to t=T using variable time steps.  The inertial relaxation
    parameter ``tau`` introduces first‑order dynamics in φ, and a filtered
    control enters through the auxiliary variable w.  The chemical
    potential μ is implicitly coupled to φ.  Mass conservation is enforced
    after each step.  When ``store_history`` is True, the full φ history
    and the time grid are returned for use in optimization and visualization.
    Otherwise the function displays a final ``imshow`` plot of φ at t=T.
    """
    # --- Unpack parameters from config object ---
    Nx, Ny = int(config.Nx), int(config.Ny)
    Lx, Ly = float(config.Lx), float(config.Ly)
    T, dt = float(config.T), float(config.dt_initial)
    tau, gamma = float(config.tau), float(config.gamma)
    c1, c2, kappa = float(config.c1), float(config.c2), float(config.kappa)
    delta_sep = 1e-2  # Numerical safeguard tolerance

    # --- Setup grid and initial conditions ---
    hx, hy = Lx / Nx, Ly / Ny
    x = np.linspace(0.0, Lx, Nx + 1)
    y = np.linspace(0.0, Ly, Ny + 1)

    phi = init_phi_random(Nx, Ny, delta_sep, amp=0.1, seed=42)
    w = np.zeros_like(phi)
    Lmat = laplacian_matrix_neumann(Nx, Ny, hx, hy)
    mu = initialize_mu(phi, w, c1, c2, kappa, Lmat, Nx, Ny, delta_sep)

    # Validate control input shape if provided
    if control_input is not None:
        if control_input.ndim != 3 or control_input.shape[1:] != phi.shape:
            raise ValueError(f"control_input must have shape (M, {Nx+1}, {Ny+1})")

    # --- Mass Conservation Setup ---
    wts_x = trapz_weights(Nx + 1)
    wts_y = trapz_weights(Ny + 1)
    wts = np.outer(wts_x, wts_y)
    wts_h = hx * hy * wts
    initial_mass = np.sum(wts_h * phi)

    # --- History Management (Flexible Lists) ---
    if store_history:
        phi_hist_list = [phi.copy()]
        t_hist_list = [0.0]

    current_time, step = 0.0, 0
    time_tol = 1e-10

    while current_time < T - time_tol:
        dt_step = min(dt, T - current_time)

        if control_input is not None and step < control_input.shape[0] - 1:
            u_n, u_np1 = control_input[step], control_input[step + 1]
        else:
            u_n, u_np1 = np.zeros_like(phi), np.zeros_like(phi)

        w_new = solve_w(w, dt_step, gamma, u_n, u_np1)
        
        if COMPUTE_ENERGY and (step % ENERGY_EVERY_N_STEPS == 0):
            E_prev = free_energy(phi, kappa, c1, c2, hx, hy, w=None, eps=0.5*delta_sep)

        phi_new, mu_new = newton_raphson(
            phi, mu, w, w_new, dt_step, tau, c1, c2, kappa, delta_sep, Lmat, Nx, Ny, hx, hy
        )
        
        if COMPUTE_ENERGY and (step % ENERGY_EVERY_N_STEPS == 0):
            E_now = free_energy(phi_new, kappa, c1, c2, hx, hy, w=None, eps=0.5*delta_sep)
            print(f"ΔE = {E_now - E_prev:.3e}")  # should be ≤ 0 up to roundoff
        phi = np.clip(phi_new, -1.0 + delta_sep, 1.0 - delta_sep)

        # --- Mass Conservation Correction ---
        current_mass = np.sum(wts_h * phi)
        mass_error = current_mass - initial_mass
        if abs(mass_error) > 1e-16:
    # leave a small margin from bounds so a tiny shift stays inside
            margin = 5e-3
            interior = np.abs(phi) < (1.0 - delta_sep - margin)
            Wint = float(np.sum(wts_h[interior]))
            if Wint > 0.0:
                phi[interior] -= mass_error / Wint
            else:
                # fallback: uniform correction then re-clip (should be rare)
                phi -= mass_error / (Lx * Ly)
                phi = np.clip(phi, -1.0 + delta_sep, 1.0 - delta_sep)

        mu, w = mu_new, w_new
        current_time += dt_step
        step += 1

        if store_history:
            phi_hist_list.append(phi.copy())
            t_hist_list.append(min(current_time, T))

        if verbose and (step % 10 == 0 or current_time >= T):
            print(f"Step {step:5d} | t={current_time:.4e} | ||phi||_inf={np.max(np.abs(phi)):.5f}")

    if verbose:
        print("Simulation complete.")

    if store_history:
        phi_hist = np.array(phi_hist_list)
        t_hist = np.array(t_hist_list)
        return phi_hist, (x, y), t_hist
    else:
        # Quick visualization of final state (orientation consistent with other plots)
        plt.figure(figsize=(6, 5))
        extent = [x[0], x[-1], y[0], y[-1]]
        plt.imshow(phi.T, origin='lower', extent=extent, vmin=-1.0, vmax=1.0, cmap='RdBu_r', interpolation='bilinear')
        plt.title(f"Final Profile of φ at t={T}", fontsize=16)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.colorbar(label="φ")
        plt.tight_layout()
        plt.show()
        return None


if __name__ == "__main__":
    params = load_params()
    fwd_cfg = get_user_input_for_config(
        ForwardSolverConfig, "Forward Solver Parameters", params.forward_solver
    )
    run_main_simulation(config=fwd_cfg, store_history=False, verbose=True)
