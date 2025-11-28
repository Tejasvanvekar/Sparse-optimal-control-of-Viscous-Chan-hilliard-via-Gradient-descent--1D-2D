import numpy as np
import matplotlib.pyplot as plt
from config import ForwardSolverConfig

"""
Forward solver for the viscous Cahn–Hilliard system with logarithmic potential and
linear control.  The state system consists of a conserved order parameter ϕ and a
chemical potential μ governed by the PDEs

  ∂t ϕ − Δ μ = 0,
  τ ∂t ϕ − Δ ϕ + f′(ϕ) = μ + w,
  γ ∂t w + w = u,

subject to homogeneous Neumann boundary conditions and prescribed initial data.  Here f is the
logarithmic double–well potential with derivative

  f′(ϕ) = c₁ log((1 + ϕ)/(1 − ϕ)) − 2 c₂ ϕ,

which becomes singular as |ϕ| → 1.  To preserve the physical range (−1, 1), the
code clips ϕ away from ±1 using a small separation buffer ``delta_sep``.

Spatial derivatives are discretised on a uniform grid of N+1 points using second-
order central finite differences with Neumann ghost–point reflection.  Time is
discretised by a Crank–Nicolson scheme combined with a convex/concave splitting
of the potential: the convex logarithmic part of f′ is treated implicitly while the
concave quadratic part is handled explicitly.  This semi–implicit method
allows large time steps while maintaining stability.  The control channel w obeys
a one–step relaxation equation γ ∂t w + w = u, which is integrated by a
Crank–Nicolson step.

The solver assembles the discrete Laplacian matrix, evaluates the nonlinear
chemical potential μ = −κ Δϕ + f′(ϕ) − w, and applies a Newton–Raphson
method with line search to solve the implicit equations for (ϕ, μ) at each
time step.  A final projection enforces mass conservation by subtracting any
accumulated mass error.  See [Colli et al., 2024] for a detailed description of
the underlying continuous problem and its well-posedness.
"""
# === PARAMETERS ===


# Safeguards / tolerances
delta_sep = 1e-2         # keep |phi| <= 1 - delta_sep

DEBUG = True
COMPUTE_ENERGY = True  # Only compute energy when needed

# === CORE FUNCTIONS ===
def instability_report(c1, c2, kappa, tau, Lx, Nmodes=12):
    import numpy as np
    a = 2*(c1 - c2)  # curvature at φ≈0 for your f'
    ks = np.pi * np.arange(1, Nmodes+1) / Lx
    q  = ks**2
    lam = (-kappa*q**2 - a*q) / (1 + tau*q)   # growth rates λ(k)
    print(f"a={a:.3g},  max λ={lam.max():.3g} at mode n={lam.argmax()+1},  unstable modes={(lam>0).sum()}")
    return lam

def regularized_log(phi, eps=None):
    
    if eps is None:
        eps = max(1e-8, 0.5*delta_sep)
    phi_s = np.clip(phi, -1+eps, 1-eps)
    return np.log((1 + phi_s)/(1 - phi_s))

def laplacian_matrix_neumann(N, h):
   
    a = 1.0 / (h*h)
    L = np.zeros((N+1, N+1))
    # Build using vectorization
    diag_indices = np.arange(1, N)
    L[diag_indices, diag_indices-1] = a
    L[diag_indices, diag_indices] = -2*a
    L[diag_indices, diag_indices+1] = a
    # boundaries: (Lv)_0 = 2/h^2 (v_1 - v_0), (Lv)_N = 2/h^2 (v_{N-1} - v_N)
    L[0, 0], L[0, 1]   = -2*a,  2*a
    L[N, N-1], L[N, N] =  2*a, -2*a
    return L

def apply_laplacian(L, v):
   
    return L @ v

def initialize_mu( phi, w, c1, c2, L, kappa):
  
    lap = apply_laplacian(L, phi)
    f_prime = c1 * regularized_log(phi) - 2.0*c2*phi
    return -kappa * lap + f_prime - w

def solve_w(w_old, dt, gamma, u_n, u_np1):
 
    gamma_dt = gamma/dt
    return ((gamma_dt - 0.5)*w_old + 0.5*(u_np1 + u_n)) / (gamma_dt + 0.5)

def solve_mu_residual(phi_new, phi_old, mu_new, mu_old, dt, L):

    lap_new = apply_laplacian(L, mu_new)
    lap_old = apply_laplacian(L, mu_old)
    return (phi_new - phi_old) / dt - 0.5*(lap_new + lap_old)

def solve_phi_residual(phi_new, phi_old, mu_new, mu_old, w_new, w_old, dt, tau, c1, c2, L, kappa ):
 
    lap_new = apply_laplacian(L, phi_new)
    lap_old = apply_laplacian(L, phi_old)
    
    f_cvx = c1 * regularized_log(phi_new)   # implicit convex
    f_ccv = -2.0 * c2 * phi_old            # explicit concave
    mu_avg = 0.5*(mu_new + mu_old)
    w_avg  = 0.5*(w_new  + w_old)

    return (tau*(phi_new - phi_old)/dt) - 0.5 * kappa * (lap_new + lap_old) + (f_cvx + f_ccv) - mu_avg - w_avg

def assemble_jacobian(phi_new, dt, tau, c1, L,kappa):
  
    Nloc = len(phi_new) - 1
    size = 2*(Nloc+1)
    J = np.zeros((size, size))
    t = tau/dt
    s = 1.0/dt

    # K_phi_phi
    Kpp = -0.5 * kappa * L.copy()
    # Vectorized diagonal computation
    phi_sq = phi_new**2
    Diag = 2.0*c1 / (1.0 - phi_sq)      # safe since line search enforces |phi|<1
    np.fill_diagonal(Kpp, np.diag(Kpp) + t + Diag)

    # K_phi_mu, K_mu_phi, K_mu_mu
    I = np.eye(Nloc+1)
    Kpm = -0.5 * I
    Kmp =  s   * I
    Kmm = -0.5 * L

    # pack using slicing
    J[:Nloc+1, :Nloc+1]       = Kpp
    J[:Nloc+1, Nloc+1:]       = Kpm
    J[Nloc+1:, :Nloc+1]       = Kmp
    J[Nloc+1:, Nloc+1:]       = Kmm
    return J

def newton_raphson(phi_old, mu_old, w_old, w_new, dt, tau, c1, c2,h,  delta_sep, L,kappa, return_residual_history=False):
 
    phi_new = phi_old.copy()
    mu_new  = mu_old.copy()
    tol = 1e-6
    max_iter = 50
    Nloc = len(phi_old) - 1

    # trapezoidal weights for conservation check (w^T L = 0)
    wts = np.ones(Nloc + 1)
    wts[0]  = 0.5
    wts[-1] = 0.5
    wts_h = h * wts  # Precompute weighted factor
    
    if return_residual_history:
        residual_norms = []
        
    for k in range(max_iter):
        # Residuals (compute once per iteration)
        res_phi = solve_phi_residual(phi_new, phi_old, mu_new, mu_old, w_new, w_old, dt, tau, c1, c2, L,kappa)
        res_mu  = solve_mu_residual(phi_new, phi_old, mu_new, mu_old, dt, L)
        R = np.concatenate([res_phi, res_mu])
        norm_R = np.linalg.norm(R)
        
        if return_residual_history:
            residual_norms.append(norm_R)

        if DEBUG and k % 10 == 0:  # Reduce frequency of debug checks
            # weighted mass balance (discrete integral with trapz weights)
            mass_defect = np.dot(wts_h, res_mu)
            if not np.isfinite(mass_defect):
              raise RuntimeError("Non-finite mass_defect; check φ bounds/log regularization.")
            #if abs(mass_defect) > 1e-12:
                #print(f"[warn] weighted mass defect = {mass_defect:.3e}")

        if norm_R < tol:
            if return_residual_history:
                return phi_new, mu_new, residual_norms
            else:
                return phi_new, mu_new # converged  # converged

        # Analytic Jacobian
        J = assemble_jacobian(phi_new, dt, tau, c1, L, kappa)

        # Solve for Newton step with regularization if needed
        try:
            delta = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            delta = np.linalg.solve(J + 1e-10*np.eye(J.shape[0]), -R)

        dphi = delta[:Nloc+1]
        dmu  = delta[Nloc+1:]

        # Step ceiling to keep φ in (-1+delta_sep, 1-delta_sep)
        # Vectorized computation
        with np.errstate(divide='ignore', invalid='ignore'):
            pos_mask = dphi > 0
            neg_mask = dphi < 0
            
            if np.any(pos_mask):
                alpha_pos = np.min((1 - delta_sep - phi_new[pos_mask]) / dphi[pos_mask])
            else:
                alpha_pos = np.inf
                
            if np.any(neg_mask):
                alpha_neg = np.min((-1 + delta_sep - phi_new[neg_mask]) / dphi[neg_mask])
            else:
                alpha_neg = np.inf
                
            alpha_max = min(alpha_pos, alpha_neg)
            
        if not np.isfinite(alpha_max) or alpha_max <= 0:
            alpha_max = 1.0
        alpha = min(1.0, 0.9 * alpha_max)

        # Armijo backtracking
        eta = 1e-3
        for _ in range(12):
            phi_t = phi_new + alpha * dphi
            mu_t  = mu_new  + alpha * dmu
            if np.all(np.abs(phi_t) < 1 - delta_sep):
                Rphi_t = solve_phi_residual(phi_t, phi_old, mu_t, mu_old, w_new, w_old, dt, tau, c1, c2, L,kappa)
                Rmu_t  = solve_mu_residual(phi_t, phi_old, mu_t, mu_old, dt, L)
                Rt = np.concatenate([Rphi_t, Rmu_t])
                if np.linalg.norm(Rt) <= (1 - eta * alpha) * norm_R:
                    phi_new, mu_new = phi_t, mu_t
                    break
            alpha *= 0.5
        else:
            # line search failed: return last iterate (partially improved state)
            return phi_new, mu_new

    # Not converged within max_iter: return last iterate
    if return_residual_history:
       return phi_new, mu_new, residual_norms
    else:
       return phi_new, mu_new

def trapz_weights(n_nodes: int) -> np.ndarray:
    w = np.ones(n_nodes)
    w[0] = 0.5
    w[-1] = 0.5
    return w

def free_energy(phi, kappa, c1, c2, h, w=None, eps=None):
    
    wts = trapz_weights(len(phi))
    # gradient part: ∫ (kappa/2) |phi_x|^2 dx  ≈  (kappa/(2h)) Σ (Δphi)^2
    dphi = np.diff(phi)  # More efficient than phi[1:] - phi[:-1]
    E_grad = (kappa/(2.0*h)) * np.sum(dphi**2)

    # bulk part: ψ(φ) with safe logs
    if eps is None: 
        eps = 1e-8
    phi_s = np.clip(phi, -1+eps, 1-eps)
    psi = c1*((1+phi_s)*np.log(1+phi_s) + (1-phi_s)*np.log(1-phi_s)) - c2*(phi_s**2)
    E_bulk = h * np.dot(wts, psi)

    E = E_grad + E_bulk

    # optional external coupling: -∫ w φ dx
    if w is not None:
        E -= h * np.dot(wts, w*phi)
    return E

def init_phi_random(N, delta_sep, amp=0.1, seed=42, enforce_zero_mean=True):
   
    rng = np.random.default_rng(seed)
    phi0 = amp * rng.standard_normal(N + 1)

    # optional: enforce trapezoidal zero mean (mass-neutral start)
    if enforce_zero_mean:
        wts = trapz_weights(N + 1)
        m = np.dot(wts, phi0) / wts.sum()
        phi0 -= m

    # safety: stay within the log domain
    phi0 = np.clip(phi0, -1 + delta_sep, 1 - delta_sep)
    return phi0


def source_u(t, x):
   
    return np.zeros_like(x)

# In Forward_solver.py

def run_main_simulation(fwd_config: ForwardSolverConfig| None = None ,store_history=False, control_input=None, verbose=True,initial_phi: np.ndarray | None = None ):
    """
    Run the forward simulation of the viscous Cahn–Hilliard system.
    
    """
    if fwd_config is None:
        fwd_config = ForwardSolverConfig()
        
    dt = fwd_config.dt_initial
    N = fwd_config.N
    Lx = fwd_config.Lx
    T = fwd_config.T
    h = Lx / N
    dt = fwd_config.dt_initial
    tau= fwd_config.tau
    c1 = fwd_config.c1
    c2 = fwd_config.c2
    gamma = fwd_config.gamma
    kappa = fwd_config.kappa
    # Make sure to use the correct kappa from the config
    
    
    x = np.linspace(0, Lx, N + 1)
    if initial_phi is not None and initial_phi.shape == (N + 1,):
        phi = initial_phi.copy()
        if verbose:
            print("Using provided initial condition for phi.")
    else:
        if verbose and initial_phi is not None:
             print(f"[Warning] Provided initial_phi has incorrect shape. Expected ({N+1},), got {initial_phi.shape}. Defaulting to random.")
        phi = init_phi_random(N, delta_sep, amp=0.01, seed=42, enforce_zero_mean=True)
    # --- END MODIFICATION ---
    #phi = init_phi_random(N, delta_sep, amp=0.01, seed=42, enforce_zero_mean=True)
    w = np.zeros(N + 1)
    Lmat = laplacian_matrix_neumann(N, h)
    wts = trapz_weights(N + 1)
    wts_h = h * wts
    initial_mass = np.dot(wts_h, phi)
    mu = initialize_mu(phi, w, c1, c2, Lmat,kappa)

    current_time = 0.0
    step = 0
    # Always keep time history
    t_hist_list = [current_time]
    # Keep phi history only if requested
    phi_hist_list = [phi.copy()] if store_history else []

    # --- FIX: Use a list to dynamically store history ---
    t_hist_list.append(min(current_time, T))
    if store_history:
        phi_hist_list.append(phi.copy())
        

    zero_source = np.zeros(N + 1)
    time_tol = 1e-10
    
    while current_time < T - time_tol:
        dt_step = min(dt, T - current_time)
        if current_time + dt_step > T:
            dt_step = T - current_time

        if control_input is not None:
            if step < control_input.shape[0] - 1:
                u_n = control_input[step, :]
                u_np1 = control_input[step + 1, :]
            else:
                u_n = control_input[step, :]
                u_np1 = control_input[step, :]
        else:
            u_n = zero_source
            u_np1 = zero_source
        w_new = solve_w(w, dt_step, gamma, u_n, u_np1)
        
        phi_new, mu_new = newton_raphson(phi, mu, w, w_new, dt_step, tau, c1, c2,h, delta_sep, Lmat, kappa)
        
        phi = np.clip(phi_new, -1 + delta_sep, 1 - delta_sep)
        mu = mu_new
        w = w_new
        current_mass = np.dot(wts_h, phi)
        mass_error = current_mass - initial_mass
        phi -= mass_error / Lx
        current_time += dt_step
        step += 1

        # --- FIX: Append current state to the list ---
        if store_history:
            phi_hist_list.append(phi.copy())
            t_hist_list.append(min(current_time, T))
            

        if verbose and (step % 100 == 0 or current_time >= T):
            print(f"Step {step:5d} | t={current_time:.4e} | ||phi||_inf={np.max(np.abs(phi)):.5f}")

    if verbose:
        print("Simulation complete.")

    t_hist = np.array(t_hist_list)

    if store_history:
        phi_hist = np.array(phi_hist_list)
        return phi_hist, x, t_hist
    else:
        # keep your plot if you like
        plt.figure(figsize=(10, 6))
        plt.plot(x, phi, label=f'Final state at t={T}')
        plt.title("Final Profile of φ", fontsize=16)
        plt.xlabel("x", fontsize=15)
        plt.ylabel("φ(x, T)", fontsize=15)
        plt.grid(True)
        plt.legend()
        plt.show()
        return phi.copy(), x, t_hist
        

if __name__ == '__main__':
    from config import ForwardSolverConfig
    print("Running a simple test simulation with default parameters...")
    default_config = ForwardSolverConfig()
    # When executed directly, run the simulation without storing the history
    
    run_main_simulation(default_config, store_history=False, verbose=True)