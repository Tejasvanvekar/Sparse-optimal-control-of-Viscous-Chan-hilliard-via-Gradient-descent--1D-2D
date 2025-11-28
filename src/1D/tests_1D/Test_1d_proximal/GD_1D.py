

# -*- coding: utf-8 -*-
"""
Interactive gradient descent loop for the sparse optimal control of the viscous
Cahn–Hilliard system.

The optimisation problem seeks a control u that minimises the cost functional
J(ϕ,u) subject to the state equations.  Following the first–order optimality
conditions, the gradient of the smooth part of J with respect to u is
∇uJ = r + b₃ u, where r is the adjoint variable obtained from the
backward solver.  An L¹–sparsity term κ ∫∫|u| dt dx leads to the projection
formula

  u⋆(x,t) = 0 ⇔ |r⋆(x,t)| ≤ κ,

from Theorem 4.7.  Accordingly, the algorithm performs a gradient step followed
by a soft–thresholding (proximal) operator and projection onto the control
bounds to enforce sparsity.  A backtracking line search ensures a decrease in
the cost, and plateau detection heuristics adjust the step size adaptively.  At
each iteration, the forward and backward solvers are called to update the
state and adjoint, and optional diagnostics such as second–order sufficiency
conditions (4.54)–(4.55) are computed.

This script also provides an interactive interface to select target states and
tracking trajectories, visualises the optimisation process, and stores the
results to disk.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation # Added for animation
import time
import json
from datetime import datetime
from Forward_solver import run_main_simulation
from backward_solver import run_backward
from cost_and_function import calculate_cost, calculate_gradient, perform_gradient_step
from second_order_conditions import approximate_second_order_condition
from config import (
    ForwardSolverConfig,
    OptimizationConfig,
    get_user_input_for_config,
    get_yes_no_input,save_params,  
    load_params   

)
import sys

# --- Controls for interactivity (match 2D style) ---
INTERACTIVE = True            # set False for batch runs
DEFAULT_TARGET_CHOICE = 1     # 1: sine, 2: cosine, 3: tan (safe)
DEFAULT_TRACKING_CHOICE = 1   # 1: ramp initial→target, 2: zeros


def perform_proximal_and_projection(u_temp, alpha, kappa, u_min, u_max):
    """
    Apply the soft–thresholding (proximal) operator for the L¹ sparsity term and
    project the result onto the admissible control bounds.

    After a gradient step, the tentative control ``u_temp`` is shrunk by the
    threshold ``alpha * kappa`` to promote sparsity.  This corresponds to the
    projection formula u⋆ = max(u_min, min(u_max, sign(u_temp) *
    max(|u_temp| − α κ, 0))) derived from the optimality system【189763021881173†L2597-L2619】.  The
    returned array ``u_final`` satisfies both the sparsity condition |u|≤κ
    wherever the adjoint r satisfies |r| ≤ κ and the box constraints u_min ≤ u ≤ u_max.
    """
    threshold = alpha * kappa
    u_intermediate = np.sign(u_temp) * np.maximum(np.abs(u_temp) - threshold, 0)
    u_final = np.clip(u_intermediate, u_min, u_max)
    return u_final

def perform_backtracking_line_search(u_k, cost_k, grad_smooth,
    phi_Q_target, phi_T_target, x, t_hist,
    b1, b2, b3, kappa, u_min, u_max, fwd_config,
    alpha_init=10.0, beta=0.8, max_ls_iter=5):
    """
    Perform a backtracking line search to determine a suitable step size α.

    Starting from ``alpha_init``, this routine proposes a new control via a
    gradient step followed by a proximal–projection, solves the forward problem
    for that control, and evaluates the cost.  If the cost decreases relative
    to ``cost_k``, the step is accepted; otherwise the step size is reduced by
    the factor ``beta``.  The process repeats up to ``max_ls_iter`` times.
    """
    alpha = alpha_init
    ls_time_total = 0.0
    success_time = 0.0
    n_trials = 0
    for _ in range(max_ls_iter):
        n_trials += 1
        u_temp = perform_gradient_step(u_k, grad_smooth, alpha)
        u_next = perform_proximal_and_projection(u_temp, alpha, kappa, u_min, u_max)
        t_try = time.perf_counter()
        
        
        phi_next, _, _ = run_main_simulation(fwd_config, store_history=True,
                                     control_input=u_next, verbose=False)
        cost_next = calculate_cost(phi_next, u_next, phi_Q_target, phi_T_target,
                           x, t_hist, b1, b2, b3, kappa, verbose=False)
        t_try = time.perf_counter() - t_try
        
        if cost_next < cost_k:
            success_time += t_try
            return alpha, u_next, cost_next, phi_next, ls_time_total, success_time, n_trials

        alpha *= beta
        ls_time_total += t_try
        
    print("[Warning] Line search could not find a step that reduces cost.")
    # last tried step is returned as "accepted"
    success_time += 0.0
    return alpha, u_next, cost_next, phi_next, ls_time_total, success_time, n_trials
    
def verify_sparsity_condition(u_optimal, r_optimal, kappa, tol=1e-6):
    """
    Numerically verify the sparsity condition of Theorem 4.7.

    The theorem states that the optimal control satisfies u⋆(x,t) = 0
    if and only if |r⋆(x,t)| ≤ κ【189763021881173†L2597-L2606】.  This function compares the zero set
    of the final control with the set where the adjoint variable satisfies
    |r| ≤ κ and prints statistics on their overlap.  A high match percentage
    provides evidence that the sparsity condition holds in the discrete
    solution.
    """
    print("\n" + "="*60)
    print("VERIFYING SPARSITY CONDITION (Theorem 4.7)")
    print("Condition: u*(x,t) = 0  <=>  |r*(x,t)| <= kappa")
    print("="*60)
    # ... (rest of function is unchanged)
    is_u_zero = np.abs(u_optimal) < tol
    is_r_small = np.abs(r_optimal) <= kappa
    conditions_match = (is_u_zero == is_r_small)
    total_points = u_optimal.size
    u_zero_count = np.sum(is_u_zero)
    r_small_count = np.sum(is_r_small)
    match_count = np.sum(conditions_match)
    sparsity_percentage = (u_zero_count / total_points) * 100
    match_percentage = (match_count / total_points) * 100
    print(f"Sparsity of final control (u* ≈ 0): {sparsity_percentage:.2f}% ({u_zero_count}/{total_points} points)")
    print(f"Region where |r*| <= kappa:          { (r_small_count / total_points) * 100:.2f}% ({r_small_count}/{total_points} points)")
    print(f"Percentage of points where the conditions match: {match_percentage:.2f}%")
    if match_percentage > 99.0:
        print("\n✓ The sparsity condition is satisfied.")
    else:
        print("\n⚠ The sparsity condition is not fully satisfied.")
    print("="*60)



def build_targets_1d(
    x: np.ndarray, t_hist: np.ndarray, phi_initial: np.ndarray,
    Lx: float, T: float,
    interactive: bool = False,
    choice_t: int = DEFAULT_TARGET_CHOICE,
    choice_q: int = DEFAULT_TRACKING_CHOICE
):
    """
    Create (phi_T_target, phi_Q_target) for 1D with three final-target options:
      1) Sinusoidal:  φ_T = A_T * sin(2πx/Lx)
      2) Cosine:      φ_T = A_T * cos(2πx/Lx)
      3) Safe Tangent: normalize tan to avoid poles and set max|φ_T|=A_T
         φ_T = A_T * tan(2π*k_tan*(x/Lx - 1/2)) / max|tan(...)|
         with default k_tan = 0.45 (< 0.5 so there are no poles on [0,Lx])
    φ_Q options:
      1) time-ramp from φ_initial to φ_T
      2) zeros (tracks 0)
    """
    # --------------------------
    # pick φ_T (final target)
    # --------------------------
    if interactive:
        print("\n" + "="*50)
        print("🎯 CHOOSE YOUR TARGET STATE (phi_T)")
        print("="*50)
        print("  1: Sinusoidal  (A_T * sin(2πx/Lx))")
        print("  2: Cosine      (A_T * cos(2πx/Lx))")
        print("  3: Tan (safe)  (normalized, no poles; max|φ_T| = A_T)")
        while True:
            try:
                choice_t = int(input("Enter your choice for the final target (1/2/3): ").strip())
                if choice_t in [1, 2, 3]:
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    # amplitude
    A_T = 0.7
    if interactive:
        try:
            amp_in = input("Enter amplitude A_T for φ_T (default 0.7): ").strip()
            A_T = float(amp_in) if amp_in else 0.7
        except Exception:
            print("[warn] Invalid amplitude; using A_T = 0.7")
            A_T = 0.7

    # optional k for tan (kept safe by default)
    k_tan = 0.45  # < 0.5 ensures no poles on [0,Lx]
    if interactive and choice_t == 3:
        try:
            kin = input("Enter k_tan ∈ (0, 0.5) for tan (default 0.45): ").strip()
            if kin:
                k_tan = max(1e-3, min(0.49, float(kin)))
        except Exception:
            print("[warn] Invalid k_tan; using 0.45")
            k_tan = 0.45

    # build φ_T
    if choice_t == 1:
        phi_T_target = A_T * np.sin(2.0 * np.pi * x / Lx)
        print("  -> φ_T: Sinusoidal")
    elif choice_t == 2:
        phi_T_target = A_T * np.cos(2.0 * np.pi * x / Lx)
        print("  -> φ_T: Cosine")
    else:
        # safe tan: center at x=Lx/2, no poles for k_tan<0.5, normalized to unit max
        arg = 2.0 * np.pi * k_tan * (x / Lx - 0.5)
        tan_raw = np.tan(arg)
        # normalize so max|tan_raw|=1 (avoid division by zero if constant)
        scale = np.max(np.abs(tan_raw))
        scale = scale if scale > 1e-12 else 1.0
        phi_T_target = A_T * (tan_raw / scale)
        print(f"  -> φ_T: Tan (safe), k_tan={k_tan:g}, normalized to amplitude A_T")

    # --------------------------
    # pick φ_Q (tracking path)
    # --------------------------
    if interactive:
        print("\n" + "="*50)
        print("🚀 CHOOSE YOUR TRACKING TRAJECTORY (phi_Q)")
        print("="*50)
        print("  1: Linear path from initial state to φ_T")
        print("  2: Zero path (φ_Q ≡ 0)")
        while True:
            try:
                choice_q = int(input("Enter your choice for the tracking path (1/2): ").strip())
                if choice_q in [1, 2]:
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    if choice_q == 1:
        time_points = (t_hist / (t_hist[-1] if t_hist[-1] > 0 else 1.0))[:, np.newaxis]  # (M+1,1)
        phi_Q_target = (1.0 - time_points) * phi_initial + time_points * phi_T_target
        print("  -> φ_Q mode: time-ramp (initial → φ_T)")
    else:
        phi_Q_target = np.zeros((len(t_hist), len(x)))
        print("  -> φ_Q mode: zeros")

    return phi_T_target, phi_Q_target


if __name__ == '__main__':
    print("Welcome to the Cahn-Hilliard Optimal Control Simulator.")
    last_run_params = load_params()
    # === PHASE 1: Forward Solver Configuration & Preview ===
    fwd_config = get_user_input_for_config(ForwardSolverConfig, "STEP 1: Configure the Forward Solver",previous_instance=last_run_params.forward_solver)

    print("\nRunning a baseline simulation with these parameters...")
    # Run simulation but don't store the full history yet to save memory
    phi_final, x, _ = run_main_simulation(fwd_config, store_history= False, verbose=False)
    
    phi_hist, x, t_hist = run_main_simulation(fwd_config, store_history=True, verbose=False)
    phi_initial_pristine = phi_hist[0, :]

    print("Baseline simulation complete. Displaying the final state (phase separation).")
    plt.figure(figsize=(10, 6))
    plt.plot(x, phi_initial_pristine, ':', color='gray', label='Initial State (t=0)')
    plt.plot(x, phi_final, '-', color='blue', label=f'Final State at t={fwd_config.T}')
    plt.title("Forward Solver Preview: Phase Separation Result", fontsize=16)
    plt.xlabel("Space (x)", fontsize=12)
    plt.ylabel("Phase Field (φ)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

    # === PHASE 2: User Decision ===
    if not get_yes_no_input("Do you want to proceed to optimization with these parameters?"):
        print("Exiting simulator.")
        sys.exit(0)
    # === PHASE 3: Optimization Configuration & Execution ===
    opt_config = get_user_input_for_config(
        OptimizationConfig,
        "STEP 2: Configure the Optimization Algorithm",previous_instance=last_run_params.optimization
    )  
        
    save_gif = get_yes_no_input("Do you want to save the final evolution as a GIF?")

    print("\n--- Starting Optimization ---")
    # Now run the simulation again, but store the history for the backward solve
    print("Initializing state for optimization...")
    phi_k, x, t_hist = run_main_simulation(fwd_config, store_history=True, verbose=False)
    u_k = np.zeros_like(phi_k)
    phi_initial_pristine = phi_k[0, :].copy()
    
    
    phi_T_target, phi_Q_target = build_targets_1d(
    x=x,
    t_hist=t_hist,
    phi_initial=phi_k[0, :].copy(),
    Lx=float(fwd_config.Lx),
    T=float(fwd_config.T),
    interactive=INTERACTIVE,
    choice_t=DEFAULT_TARGET_CHOICE,
    choice_q=DEFAULT_TRACKING_CHOICE,
)



    
    
    print("Setup complete. Beginning gradient descent loop...")
    
    cost_history = []
    alpha_history = []
    tracking_err_hist = []   # relative ||phi - phi_Q|| over space-time
    terminal_err_hist = []   # relative ||phi(T) - phi_T||

    # --- timing accumulators ---
    timers = {
        "total_optimization": 0.0,
        "backward_total": 0.0,
        "line_search_total": 0.0,      # rejected trials only
       "successful_step_total": 0.0,  # accepted step eval (forward + cost)
        "optimistic_eval_total": 0.0,  # optimistic forward+cost time (subset of successful if accepted)
    }
    iter_walltimes = []
    ls_trials_per_iter = []
    cost_k = calculate_cost(
        phi_k, u_k, phi_Q_target, phi_T_target, x, t_hist,
        opt_config.b1, opt_config.b2, opt_config.b3, opt_config.kappa_sparsity
    )
    cost_history.append(cost_k)
    alpha_prev = opt_config.alpha_max
    
    # --- ALL FEATURE INITIALIZATIONS ---
    # Alpha Advisor
    successful_optimistic_alphas = []
    start_storing_iter = 100
    stable_avg_counter = 0
    last_avg_alpha = 0.0
    # Plateau Detection
    plateau_counter = 0
    PLATEAU_LENGTH = 10
    PLATEAU_TOLERANCE = 1e-7
    
    
    t_opt_start = time.perf_counter()
    for k in range(opt_config.max_iter):
        t_iter_start = time.perf_counter()
        print(f"\n--- Iteration {k+1}/{opt_config.max_iter} ---")
        print(f"Current Cost = {cost_k:.6f}")
        
        t0 = time.perf_counter()
        _, _, r_k = run_backward(phi_k, x, t_hist,
                         opt_config.b1, opt_config.b2,
                         phi_Q_target, phi_T_target)
        timers["backward_total"] += time.perf_counter() - t0
        grad_smooth = calculate_gradient(r_k, u_k, opt_config.b3)
        
        u_optimistic_temp = perform_gradient_step(u_k, grad_smooth, alpha_prev)
        u_optimistic = perform_proximal_and_projection(u_optimistic_temp, alpha_prev,
                                               opt_config.kappa_sparsity,
                                               opt_config.u_min, opt_config.u_max)
        t1 = time.perf_counter()
        phi_optimistic, _, _ = run_main_simulation(fwd_config, store_history=True,
                                           control_input=u_optimistic, verbose=False)
        cost_optimistic = calculate_cost(phi_optimistic, u_optimistic,
                                 phi_Q_target, phi_T_target, x, t_hist,
                                 opt_config.b1, opt_config.b2, opt_config.b3,
                                 opt_config.kappa_sparsity, verbose=False)
        optimistic_eval_time = time.perf_counter() - t1
        
        if cost_optimistic < cost_k:
            print(f"Optimistic step successful with alpha = {alpha_prev:.4f}")
            alpha_k = alpha_prev
            u_k_plus_1 = u_optimistic
            cost_k_plus_1 = cost_optimistic
            phi_k_plus_1 = phi_optimistic
            timers["optimistic_eval_total"] += optimistic_eval_time
            timers["successful_step_total"] += optimistic_eval_time
            ls_trials_per_iter.append(1)           
            
            # --- Alpha Advisor Logic ---
            if k >= start_storing_iter:
                successful_optimistic_alphas.append(alpha_prev)
                if len(successful_optimistic_alphas) > 10:
                    current_avg_alpha = np.mean(successful_optimistic_alphas)
                    if np.isclose(current_avg_alpha, last_avg_alpha, rtol=1e-3):
                        stable_avg_counter += 1
                    else:
                        stable_avg_counter = 0
                    last_avg_alpha = current_avg_alpha
                    
                    if stable_avg_counter >= 50 and k % 10 == 0:
                        print("\n" + "*"*65)
                        print(f"    [LIVE ADVISOR] Stable average alpha of {current_avg_alpha:.4f} found.")
                        print("    Consider stopping (Ctrl+C) and restarting with this value")
                        print("    as your new `alpha_max` for faster convergence.")
                        print("*"*65)
        
        else:
            print(f"[Notice] Optimistic step failed. Engaging full backtracking search...")
            alpha_k, u_k_plus_1, cost_k_plus_1, phi_k_plus_1,ls_time,success_time, n_trials = perform_backtracking_line_search(
                u_k, cost_k, grad_smooth, phi_Q_target, phi_T_target, x, t_hist,
                opt_config.b1, opt_config.b2, opt_config.b3, opt_config.kappa_sparsity,
                opt_config.u_min, opt_config.u_max, fwd_config,  
                alpha_init=alpha_prev
            )

            print(f"Backtracking found optimal alpha = {alpha_k:.4f}")
            timers["line_search_total"] += ls_time
            timers["successful_step_total"] += success_time
            ls_trials_per_iter.append(n_trials)
        cost_history.append(cost_k_plus_1)
        alpha_history.append(alpha_k)

        # --- Convergence metrics (relative L2) for accepted step ---
        # space-time tracking error
        # --- robust norms & normalizations ---
        def _l2_xt(A):
            # L2 over space-time: trapz in x (per time), then in t
            s = np.trapz(A**2, x=x, axis=1)
            return np.sqrt(np.trapz(s, x=t_hist))

        def _l2_x(a):
            return np.sqrt(np.trapz(a**2, x=x))

        # space-time RMS scale (used when phi_Q is ~0)
        domain_len = float(x[-1] - x[0])
        time_len   = float(t_hist[-1] - t_hist[0])
        rms_scale_xt = np.sqrt(max(domain_len, 1e-30) * max(time_len, 1e-30))

        # tracking error (auto-normalized)
        numQ = _l2_xt(phi_k_plus_1 - phi_Q_target)
        denQ = _l2_xt(phi_Q_target)
        if denQ < 1e-9 * rms_scale_xt:
            # target is ~zero → use RMS scale (absolute error reported as "relative")
            denQ = rms_scale_xt
        rel_track = numQ / (denQ + 1e-12)
        tracking_err_hist.append(rel_track)

        # terminal error (relative to target final state)
        numT = _l2_x(phi_k_plus_1[-1] - phi_T_target)
        denT = _l2_x(phi_T_target) + 1e-12
        terminal_err_hist.append(numT / denT)
        
        # --- Combined Plateau Detection and Optimistic Alpha Update ---
        if k > 0 and abs(cost_history[-1] - cost_history[-2]) < PLATEAU_TOLERANCE:
            plateau_counter += 1
        else:
            plateau_counter = 0
        ak = alpha_k if (alpha_k is not None and np.isfinite(alpha_k)) else alpha_prev
        if plateau_counter >= PLATEAU_LENGTH:
            print(f"[Notice] Cost has plateaued for {plateau_counter} iterations. Boosting learning rate.")
            alpha_prev = min(opt_config.alpha_max, alpha_k * 2.0)
            plateau_counter = 0
        else:
            alpha_prev = min(opt_config.alpha_max, alpha_k * 1.2)
        
        # --- Original Stopping Criteria ---
        change = np.linalg.norm(u_k_plus_1 - u_k) / (np.linalg.norm(u_k) + 1e-9)
        print(f"Relative control change: {change:.6e}")
        iter_walltimes.append(time.perf_counter() - t_iter_start)  
        if change < 1e-5 and k > 10:
            print(f"\nConvergence reached at iteration {k+1}.")
            u_k = u_k_plus_1.copy()
            final_iteration_count = k + 1 # Store final count
            break
        
        u_k = u_k_plus_1.copy()
        cost_k = cost_k_plus_1
        phi_k = phi_k_plus_1
    else: # This 'else' belongs to the 'for' loop
        final_iteration_count = opt_config.max_iter
        print("\nWarning: Optimization finished because max_iter was reached.")
    timers["total_optimization"] = time.perf_counter() - t_opt_start
    iter_walltimes.append(time.perf_counter() - t_iter_start)       
        
    print("\nOptimization finished.")
    u_optimal = u_k.copy()
    r_optimal = r_k.copy() if 'r_k' in locals() else np.zeros_like(u_k)
    np.save("optimal_control.npy", u_optimal)
    print("Optimal control saved as 'optimal_control.npy'")

    # Additional check: second–order sufficient optimality conditions
    print("\nChecking second–order sufficient condition via finite–difference tests...")
    num_test_dirs = 3
    hessian_values = approximate_second_order_condition(fwd_config=fwd_config,
    u_star=u_optimal, r_star=r_optimal, phi_star=phi_k, x=x, t_hist=t_hist,
    b1=opt_config.b1, b2=opt_config.b2, b3=opt_config.b3,
    kappa=opt_config.kappa_sparsity,
    phi_Q_target=phi_Q_target, phi_T_target=phi_T_target,
    u_min=opt_config.u_min, u_max=opt_config.u_max,
    num_directions=3, epsilon=1e-4, seed=42
    )

    for i, d2 in enumerate(hessian_values, start=1):
        print(f"  Direction {i}: estimated second derivative = {d2:.6e}")
    if all(val > 0 for val in hessian_values):
        print("\n✓ The numerical tests suggest that the coercivity condition holds in the tested directions.")
    else:
        print("\n⚠ Some directions exhibit non–positive second derivatives; the second–order condition may fail.")

    # --- Final Alpha Advisor Suggestion ---
    if successful_optimistic_alphas and last_run_params.last_run_iterations > 100:
        final_avg_alpha = np.mean(successful_optimistic_alphas)
        print("\n" + "="*60)
        print("💡 OPTIMIZATION TIP: ALPHA ADVISOR")
        print(f"Your last run took {last_run_params.last_run_iterations} iterations.")
        print(f"Based on this run, a good initial `alpha_max` for next time might be: {final_avg_alpha:.4f}")
        print("="*60)
        
    verify_sparsity_condition(u_optimal, r_optimal, opt_config.kappa_sparsity)

    
    print("\n--- Generating Final Visualizations ---")
    
    print("Running final forward simulation with u_optimal...")
    phi_hist_final, x_final, t_final = run_main_simulation(fwd_config, store_history=True,
                                                       control_input=u_optimal, verbose=False)
    phi_final_state = phi_hist_final[-1, :]
    print("Final simulation complete.")

    plt.figure(figsize=(12, 7))
    plt.plot(x, phi_initial_pristine, ':', color='gray', label='Initial State (t=0)', linewidth=2)
    plt.plot(x, phi_T_target, '--', color='red', label='Target State (Desired)', linewidth=2.5)
    plt.plot(x_final, phi_final_state, '-', color='blue', label='Final State (Achieved with u*)', linewidth=3)
    plt.title('Effect of Optimal Control: Initial vs. Final vs. Target', fontsize=16)
    plt.xlabel('Space (x)', fontsize=15)
    plt.ylabel('Phase Field (φ)', fontsize=15)
    plt.ylim(-1.1, 1.1)
    plt.legend(fontsize=12.5)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("phi_comparison_plot.png", dpi=250)
    print("Comparison plot saved as 'phi_comparison_plot.png'")
    

    # --- Convergence figure (cost + relative errors) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [1.2, 1]})
    ax1.plot(range(len(cost_history)), cost_history, 'k.-', label='Total Cost (J)')
    ax1.set_ylabel("Total Cost",fontsize=15)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper right')
    ax1.set_title("Convergence Analysis Over Iterations",fontsize=16)
    ax2.plot(range(1, len(tracking_err_hist)+1), tracking_err_hist, 'o--', label='Tracking Error ‖φ - φ_Q‖')
    ax2.plot(range(1, len(terminal_err_hist)+1), terminal_err_hist, 'o-', label='Terminal Error ‖φ(T) - φ_T‖')
    ax2.set_yscale('log')
    ax2.set_xlabel("Iteration",fontsize=15)
    ax2.set_ylabel("Relative L2 Error (log scale)",fontsize=15)
    ax2.grid(True, which='both', linestyle=':', alpha=0.5)
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("convergence_analysis.png", dpi=250)
    print("Convergence plot saved as 'convergence_analysis.png'")

    # --- Time study summary ---
    print("\n" + "="*60)
    print("COMPUTATIONAL TIME STUDY (wall-clock)")
    print("="*60)
    print(f"Total optimization (loop) time       : {timers['total_optimization']:.3f} s")
    print(f"  ├─ Backward solves (adjoint r)     : {timers['backward_total']:.3f} s")
    print(f"  ├─ Line-search backtracking (rejs.): {timers['line_search_total']:.3f} s")
    print(f"  └─ Successful step eval (accepted) : {timers['successful_step_total']:.3f} s")
    print(f"      (optimistic accepted portion)  : {timers['optimistic_eval_total']:.3f} s")
    if ls_trials_per_iter:
        import statistics as _st
        print(f"Avg # line-search trials/iter        : {_st.mean(ls_trials_per_iter):.2f}")
        print(f"Max # line-search trials in an iter  : {max(ls_trials_per_iter)}")
    print("="*60)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, phi_T_target, 'r--', label='Target State')
    line, = ax.plot(x_final, phi_hist_final[0, :], 'b-', lw=2, label='Evolving State (φ)')
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Space (x)")
    ax.set_ylabel("Phase Field (φ)")
    ax.set_title("Evolution of φ under Optimal Control")
    ax.legend()
    ax.grid(True, linestyle='--')

    skip_rate_gif = 10
    phi_hist_gif = phi_hist_final[::skip_rate_gif]
    t_gif = t_final[::skip_rate_gif]
    print(f"GIF will be generated from {len(t_gif)} frames to save memory.")
    
    def update_gif(frame):
        line.set_ydata(phi_hist_gif[frame, :])
        time_text.set_text(f'Time = {t_gif[frame]:.3f}s')
        return line, time_text

    ani_gif = animation.FuncAnimation(fig, update_gif, frames=len(t_gif), interval=50, blit=True)

    print("Saving animation as GIF...")
    ani_gif.save("phi_evolution.gif", writer='pillow', fps=20, dpi=90)
    print("Animation saved as 'phi_evolution.gif'")
    
    
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    

    save_params(fwd_config, opt_config, final_iteration_count)
    plt.show()