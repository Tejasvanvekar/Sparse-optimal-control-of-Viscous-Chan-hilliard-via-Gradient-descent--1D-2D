
"""
Backward solver computing the adjoint variables p, q and r for the viscous
Cahn–Hilliard optimal control problem.

The adjoint system arises from differentiating the Lagrangian with respect
to the state variables.  For the continuous problem it reads:

  −∂t p − τ ∂t q − Δ q + f″(ϕ) q = b₁ (ϕ − ϕ_Q),
  −Δ p − q = 0,
  −γ ∂t r + r = q,

with Neumann boundary conditions ∂n p = ∂n q = 0 and terminal conditions
p(T) + τ q(T) = b₂ (ϕ(T) − ϕ_Ω) and r(T) = 0.  The discrete solver
mirrors the Crank–Nicolson discretisation used in the forward solver:
at each time step we solve a linear system for p using matrices A_adjoint and
B_adjoint that depend on the forward state ϕ, then compute q = −Δ p and
update r by a backward Crank–Nicolson step.  The second derivative of the
logarithmic potential is f″(ϕ) = 2 c₁/(1 − ϕ²) − 2 c₂.  The resulting
adjoint variables are returned with the same shape as the forward solution and
provide the gradient ∇u J = r + b₃ u needed for optimisation.
"""

# In backward_solver_fixed.py
import numpy as np
from typing import Tuple, Optional
from Forward_solver import laplacian_matrix_neumann
from config import ForwardSolverConfig
_cfg = ForwardSolverConfig()


c1, c2, tau, gamma = _cfg.c1, _cfg.c2, _cfg.tau, _cfg.gamma
kappa = _cfg.kappa


def fpp_log(phi: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Evaluate the second derivative of the logarithmic potential f″(ϕ) for the
    Cahn–Hilliard system with a small clipping to avoid hitting the singular
    points ±1.  For the potential f_log(ϕ) = c₁[(1+ϕ)log(1+ϕ)+(1−ϕ)log(1−ϕ)] − c₂ ϕ²,
    the second derivative is f″(ϕ) = 2 c₁/(1 − ϕ²) − 2 c₂.  The
    function clips ϕ into (−1,1) with a tolerance ``eps`` before evaluating the
    formula to prevent numerical overflow.
    """
    ph = np.clip(phi, -1 + eps, 1 - eps)
    return 2.0 * c1 / (1.0 - ph**2) - 2.0 * c2

def run_backward(
    phi_hist: np.ndarray,
    x: np.ndarray,
    t_hist: np.ndarray,
    b1: float,
    b2: float,
    phi_Q: Optional[np.ndarray] = None,
    phi_T_target: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the discretised adjoint system backward in time.

    Given the forward state history ``phi_hist`` and grids ``x``, ``t_hist``, this
    routine computes the adjoint variables p, q and r that satisfy the
    discretised version of the continuous adjoint PDEs.  The
    right–hand sides involve the running tracking term b₁(ϕ−ϕ_Q) and the
    terminal tracking term b₂(ϕ(T)−ϕ_T_target).  Matrices ``A_adjoint`` and
    ``B_adjoint`` are assembled per time step to implement the Crank–Nicolson
    discretisation of the operator −∂t p − τ ∂t q + Δ² p − f″(ϕ) Δ p.  After
    solving for p and q, the control–channel adjoint r is updated via a
    backward Crank–Nicolson step that mirrors the discretisation of γ ∂t w + w = u
    in the forward problem.  The resulting arrays p, q and r can be used to
    compute the gradient ∇u J = r + b₃ u.
    """
    M_plus1, N_plus1 = phi_hist.shape
    
    # --- Setup Targets ---
    if phi_Q is None:
        phi_Q = np.zeros_like(phi_hist)
    if phi_T_target is None:
        phi_T_target = np.zeros(N_plus1)
    
    # --- Grid and Operators ---
    h = x[1] - x[0]
    dts = np.diff(t_hist)
    L = laplacian_matrix_neumann(N_plus1 - 1, h)
    L2 = L @ L
    I = np.eye(N_plus1)
    
    # --- Allocate Adjoints ---
    p = np.zeros_like(phi_hist)
    q = np.zeros_like(phi_hist)
    r = np.zeros_like(phi_hist)
    
    # --- Terminal Conditions at t=T ---
    rhs_T = b2 * (phi_hist[-1] - phi_T_target)
    p[-1] = np.linalg.solve(I - tau * L, rhs_T)
    q[-1] = -(L @ p[-1])
    r[-1] = np.zeros(N_plus1)
    
    # --- Adjoint Operators ---
    def A_adjoint(phi_n, dt_n):
        fpp_diag = np.diag(fpp_log(phi_n))
        return I - tau*L + 0.5*dt_n*L2 - 0.5*dt_n*(fpp_diag @ L)
    
    def B_adjoint(phi_np1, dt_n):
        fpp_diag = np.diag(fpp_log(phi_np1))
        return I - tau*L - 0.5*dt_n*L2 + 0.5*dt_n*(fpp_diag @ L)

    # --- Backward Time March ---
    for n in range(M_plus1 - 2, -1, -1):
        dt_n = t_hist[n+1] - t_hist[n]
        if dt_n <= 0: continue

        src = 0.5 * dt_n * b1 * ((phi_hist[n] - phi_Q[n]) + (phi_hist[n+1] - phi_Q[n+1]))
        rhs = B_adjoint(phi_hist[n+1], dt_n) @ p[n+1] + src
        
        try:
            p[n] = np.linalg.solve(A_adjoint(phi_hist[n], dt_n), rhs)
        except np.linalg.LinAlgError:
            p[n] = np.linalg.solve(A_adjoint(phi_hist[n], dt_n) + 1e-10*I, rhs)
        
        q[n] = -(L @ p[n])
        
        gamma_factor_back = (gamma - 0.5*dt_n) / (gamma + 0.5*dt_n)
        gamma_factor_source = (dt_n * 0.5) / (gamma + 0.5*dt_n)
        r[n] = gamma_factor_back * r[n+1] + gamma_factor_source * (q[n] + q[n+1])

    return p, q, r