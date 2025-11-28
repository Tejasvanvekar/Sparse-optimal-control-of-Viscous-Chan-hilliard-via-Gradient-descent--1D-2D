
"""
GD2_configured.py
==================

This script orchestrates a full 2D optimal control workflow for the
Cahn–Hilliard equation using gradient descent.  Starting from an initial
state, it runs an uncontrolled forward simulation, constructs target
trajectories and terminal targets, then iteratively updates the control
field ``u`` to minimize a cost functional consisting of tracking,
terminal, control energy and sparsity terms.

Major components include:
  - **Interactive parameter selection** using Pydantic models and optional
    prompts.
  - **Forward simulation** via :func:`run_main_simulation` to evaluate the
    state under a given control.
  - **Backward adjoint solve** via :func:`run_backward` to compute
    sensitivities of the cost with respect to the state.
  - **Gradient computation and proximal update** for the control, with
    optional backtracking line search.
  - **Monitoring and termination criteria**, including plateau detection
    and relative control change.
  - **Visualization and analysis** of the optimal solution and convergence
    behavior.

The script is designed to run either interactively (with prompts for
parameters and choices) or non‑interactively by setting the
``INTERACTIVE`` flag.  Running as a standalone program will launch the
full optimization loop.  Intermediate and final results are saved as
images for post‑processing.
"""
import sys
import time
import warnings
from typing import Tuple

import numpy as np

from second_order_conditions_2d import approximate_second_order_condition_2d, verify_sparsity_condition
from visualization_3d import (
    generate_all_3d_plots, plot_convergence_history, save_parameter_text_image,
    animate_time_evolution, create_comparison_panel, create_1d_slice_comparison,
    format_time_hms,_show_forward_final_imshow
)
from Forward2_solver import run_main_simulation
from backward2_solver import run_backward
from cost2_and_function import calculate_cost, calculate_gradient, proximal_step
from config import (
    ForwardSolverConfig, OptimizationConfig, load_params, save_params, get_yes_no_input,
    get_user_input_for_config
)

# --- Controls for interactivity (set to False for headless/batch runs) ---
INTERACTIVE = True  # change to True if you want prompts
DEFAULT_TARGET_CHOICE = 1  # 1: Sinusoid, 2: Centered Circle
DEFAULT_TRACKING_CHOICE = 1  # 1: ramp initial→target, 2: zeros


def _dump_config_json(cfg) -> str:
    """Return a JSON-like string for cfg, working on both Pydantic v1 and v2."""
    try:
        return cfg.model_dump_json(indent=2)  # pydantic v2
    except Exception:
        try:
            return cfg.json(indent=2)  # pydantic v1
        except Exception:
            return repr(cfg)


def perform_backtracking_line_search_2D(
    u_k: np.ndarray, cost_k: float, grad_smooth: np.ndarray,
    phi_Q_target: np.ndarray, phi_T_target: np.ndarray,
    x: np.ndarray, y: np.ndarray,
    fwd_config: ForwardSolverConfig, opt_config: OptimizationConfig,
    alpha_init: float = 1.0, beta: float = 0.8, max_ls_iter: int = 10
) -> Tuple[float, np.ndarray, float, np.ndarray, np.ndarray,float, int]:
    """
    Perform a backtracking line search for the 2D optimal control problem.

    Given a current control ``u_k`` and cost ``cost_k``, together with the
    gradient of the smooth part, this function tries to find a step size
    α such that the updated control reduces the cost functional.  The
    algorithm iteratively decreases α by a factor ``beta`` until a lower
    cost is achieved or ``max_ls_iter`` attempts have been made.  At each
    trial α, the updated control is computed via :func:`proximal_step`, a
    forward simulation is run to obtain the new state, and the cost is
    re‑evaluated.

    Parameters
    ----------
    u_k : ndarray
        Current control guess.
    cost_k : float
        Cost associated with ``u_k``.
    grad_smooth : ndarray
        Gradient of the smooth part of the cost functional.
    phi_Q_target, phi_T_target : ndarrays
        Desired trajectories and terminal state.
    x, y : 1D ndarrays
        Spatial coordinates.
    fwd_config : ForwardSolverConfig
        Forward solver configuration for running simulations.
    opt_config : OptimizationConfig
        Contains step size bounds and sparsity coefficient.
    alpha_init : float, optional
        Initial trial step size.  Default is 1.0.
    beta : float, optional
        Reduction factor for α on each unsuccessful trial.  Default is 0.8.
    max_ls_iter : int, optional
        Maximum number of backtracking attempts.

    Returns
    -------
    Tuple of (alpha, u_next, cost_next, phi_next, t_hist_next, elapsed_time, attempts)
        Contains the step size used, the updated control, the cost at the
        updated control, the state history, the time grid, total time spent
        during line search, and the number of attempts made.
    """
    
    alpha = alpha_init
    u_next = u_k
    phi_next = None
    t_hist_next = None
    cost_next = cost_k
    ls_start = time.perf_counter()
    attempts = 0
    for i in range(max_ls_iter):
        attempts += 1
        u_next = proximal_step(u_k, grad_smooth, alpha, opt_config)
        t_sim_start = time.perf_counter()
        phi_next, _, t_hist_next = run_main_simulation(
            config=fwd_config, store_history=True, control_input=u_next, verbose=False
        )
        cost_next = calculate_cost(
            phi_next, u_next, phi_Q_target, phi_T_target,
            x, y, t_hist_next, opt_config
        )
        if cost_next < cost_k:
            print(f"   ✓ Backtracking found a good step (α = {alpha:.4f}) after {i+1} attempts.")
            ls_elapsed = time.perf_counter() - ls_start
            return alpha, u_next, cost_next, phi_next, t_hist_next, ls_elapsed, attempts
        alpha *= beta
    print("[Warning] Line search could not find a step that reduces cost. Returning last try.")
    ls_elapsed = time.perf_counter() - ls_start
    return alpha, u_next, cost_next, phi_next, t_hist_next, ls_elapsed, attempts


def build_targets(
    x: np.ndarray, y: np.ndarray, t_hist: np.ndarray, phi_initial: np.ndarray,
    Lx: float, Ly: float, T: float,
    interactive: bool = False, choice_t: int = 1, choice_q: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the terminal and tracking target trajectories for optimization.

    Depending on user choice (interactive or via ``choice_t``/``choice_q``),
    this helper returns a final target field φ_T and a space–time tracking
    trajectory φ_Q.  Two options are provided for each: sinusoidal pattern
    or centered circle for φ_T, and linear interpolation from the initial
    state to φ_T or a zero trajectory for φ_Q.

    Parameters
    ----------
    x, y : 1D ndarrays
        Spatial coordinates.
    t_hist : 1D ndarray
        Time grid from the forward simulation.
    phi_initial : ndarray
        Initial state of φ used as the starting point for linear paths.
    Lx, Ly, T : float
        Domain lengths and final time used to scale patterns.
    interactive : bool, optional
        If True, prompt the user for choices.  Otherwise ``choice_t`` and
        ``choice_q`` determine the targets.
    choice_t, choice_q : int, optional
        Preselected choices for the final target and tracking trajectory.

    Returns
    -------
    Tuple (phi_T_target, phi_Q_target)
        The final target field and the tracking trajectory array.
    """
    xx, yy = np.meshgrid(x, y, indexing='ij')
    # --- Choose phi_T ---
    if interactive :
        print("\n" + "="*50)
        print("🎯 CHOOSE YOUR TARGET STATE (phi_T)")
        print("="*50)
        print("  1: Sinusoidal Pattern\n  2: Centered Circle")
        while True:
            try:
                choice_t = int(input("Enter your choice for the final target (1 or 2): ").strip())
                if choice_t in [1, 2]: break
                else: print("Invalid choice. Please enter 1 or 2.")
            except ValueError: print("Invalid input. Please enter a number.")
    if choice_t == 1:
        print("  -> φ_T: Sinusoidal Pattern.")
        phi_T_target = 0.7 * np.sin(2 * np.pi * xx / Lx) * np.cos(np.pi * yy / Ly)
    else:
        print("  -> φ_T: Centered Circle.")
        radius_sq = (Lx / 3.5)**2
        phi_T_target = -np.ones_like(xx)
        mask = (xx - Lx / 2)**2 + (yy - Ly / 2)**2 < radius_sq
        phi_T_target[mask] = 1.0

    # --- Choose φ_Q path ---
    if interactive :
        print("\n" + "="*50)
        print("🚀 CHOOSE YOUR TRACKING TRAJECTORY (phi_Q)")
        print("="*50)
        print("  1: Linear path from initial state to final target\n  2: Zero target (force φ→0)")
        while True:
            try:
                choice_q = int(input("Enter your choice for the tracking path (1 or 2): ").strip())
                if choice_q in [1, 2]: break
                else: print("Invalid choice. Please enter 1 or 2.")
            except ValueError: print("Invalid input. Please enter a number.")

    if choice_q == 1:
        time_points = (t_hist / T)[:, np.newaxis, np.newaxis]
        phi_Q_target = (1 - time_points) * phi_initial + time_points * phi_T_target
        print("  -> φ_Q mode: time-ramp (initial → φ_T)")
    else:
        phi_Q_target = np.zeros((len(t_hist), len(x), len(y)))
        print("  -> φ_Q mode: zeros")

    return phi_T_target, phi_Q_target


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    print("="*60 + "\n    2D GRADIENT DESCENT OPTIMIZATION \n" + "="*60)

    # --- Load configs (optionally interactive) ---
    all_params = load_params()
    if INTERACTIVE  and get_yes_no_input("Do you want to modify the simulation parameters?"):
        fwd_config = get_user_input_for_config(ForwardSolverConfig, "Forward Solver Parameters", all_params.forward_solver)
        opt_config = get_user_input_for_config(OptimizationConfig, "Optimization Parameters", all_params.optimization)
    else:
        fwd_config, opt_config = all_params.forward_solver, all_params.optimization

    b1, b2, b3, kappa_sparsity = opt_config.b1, opt_config.b2, opt_config.b3, opt_config.kappa_sparsity
    alpha_max, MAX_ITER = opt_config.alpha_max, opt_config.max_iter
    Nx, Ny, Lx, Ly, T = fwd_config.Nx, fwd_config.Ny, fwd_config.Lx, fwd_config.Ly, fwd_config.T

    print("\n--- Using the following parameters ---\nForward Solver Config:")
    print(_dump_config_json(fwd_config))
    print("Optimization Config:")
    print(_dump_config_json(opt_config))

    # --- Forward preview immediately after entering forward parameters ---
    print("\n🔧 Running forward simulation with chosen parameters...")
    start_time = time.time()
    phi_k, (x, y), t_hist_k = run_main_simulation(
        config=fwd_config, store_history=True, control_input=None, verbose=False
    )
    phi_initial_pristine, u_k = phi_k[0].copy(), np.zeros_like(phi_k)
    print("✓ Forward simulation complete")
    # Show the imshow-style preview from the forward solver (no saving)
    T_preview = float(t_hist_k[-1]) if len(t_hist_k) > 0 else fwd_config.T
    print("\n📷 Showing forward preview (final φ)...")
    _show_forward_final_imshow(phi_k[-1], x, y, T_preview)
    if INTERACTIVE:
        if not get_yes_no_input("Proceed to optimization with these parameters?"):
            print("\n🛑 Optimization cancelled by user after forward preview.")
            sys.exit(0)
 

    # --- Build targets (non-interactive by default) ---
    phi_T_target, phi_Q_target = build_targets(
        x, y, t_hist_k, phi_initial_pristine, Lx, Ly, T,
        interactive=INTERACTIVE, choice_t=DEFAULT_TARGET_CHOICE, choice_q=DEFAULT_TRACKING_CHOICE
    )

    # --- Optimization loop ---
    cost_history, terminal_error_history, tracking_error_history = [], [], []
    terminal_error_history = [] 
    t_backward_total = 0.0
    t_gradprox_total = 0.0
    t_optimistic_fwd_total = 0.0
    t_optimistic_cost_total = 0.0
    t_backtracking_total = 0.0
    n_backtracking_calls = 0
    n_backtracking_attempts = 0
    t_successful_steps_total = 0.0  # wall time spent on optimistic steps that were accepted
    successful_optimistic_alphas = []
    plateau_counter, PLATEAU_LENGTH, PLATEAU_TOLERANCE = 0, 5, 1e-5

    print("🚀 Starting optimization loop...\n" + "-" * 50)
    cost_k = calculate_cost(phi_k, u_k, phi_Q_target, phi_T_target, x, y, t_hist_k, opt_config)
    cost_history.append(cost_k)
    alpha_prev = alpha_max

    for k in range(MAX_ITER):
        iter_start = time.perf_counter()
        print(f"\n📍 Iteration {k+1}/{MAX_ITER} | Current Cost = {cost_k:.6f}")
        tb0 = time.perf_counter()
        _, _, r_k = run_backward(phi_k, x, y, t_hist_k, fwd_config, b1, b2, phi_Q_target, phi_T_target)
        t_backward_total += time.perf_counter() - tb0
        
        
        tg0 = time.perf_counter()
        grad_smooth = calculate_gradient(r_k, u_k, opt_config)
        u_optimistic = proximal_step(u_k, grad_smooth, alpha_prev, opt_config)
        t_gradprox_total += time.perf_counter() - tg0
        
        tof0 = time.perf_counter()
        phi_optimistic, _, t_hist_optimistic = run_main_simulation(config=fwd_config, store_history=True, control_input=u_optimistic, verbose=False)
        t_optimistic_fwd_total += time.perf_counter() - tof0
        toc0 = time.perf_counter()
        cost_optimistic = calculate_cost(phi_optimistic, u_optimistic, phi_Q_target, phi_T_target, x, y, t_hist_optimistic, opt_config)
        t_optimistic_cost_total += time.perf_counter() - toc0

        step_was_successful = False
        if cost_optimistic < cost_k:
            print(f"   ✓ Optimistic step successful (α = {alpha_prev:.4f})")
            step_was_successful = True
            alpha_k, u_k_plus_1, cost_k_plus_1, phi_k_plus_1, t_hist_k_plus_1 = alpha_prev, u_optimistic, cost_optimistic, phi_optimistic, t_hist_optimistic
        else:
            print(f"   ⚠ Optimistic step failed. Backtracking...")
            alpha_k, u_k_plus_1, cost_k_plus_1, phi_k_plus_1, t_hist_k_plus_1 , ls_elapsed, attempts = perform_backtracking_line_search_2D(
                    u_k, cost_k, grad_smooth, phi_Q_target, phi_T_target, x, y,
                    fwd_config, opt_config, alpha_init=alpha_prev * 0.8
                )
            t_backtracking_total += ls_elapsed
            n_backtracking_calls += 1
            n_backtracking_attempts += attempts
            
        if step_was_successful: 
            successful_optimistic_alphas.append(alpha_k)

        cost_history.append(cost_k_plus_1)            
        # ---------- Robust space–time norms with auto-normalization ----------
        # L2 over space (Ω): trapz in y then x (arrays are (Nx, Ny))
        def _l2_xy(Axy: np.ndarray) -> float:
            s_y = np.trapz(Axy**2, x=y, axis=1)     # integrate over y (axis=1)
            s_xy = np.trapz(s_y, x=x)               # then over x
            return float(np.sqrt(max(s_xy, 0.0)))

        # L2 over space–time: trapz over space at each t, then trapz over t
        def _l2_xyt(A: np.ndarray) -> float:
            # A shape: (Nt, Nx, Ny)
            space_int = np.array([_l2_xy(A[t])**2 for t in range(A.shape[0])])
            st = np.trapz(space_int, x=t_hist_k)
            return float(np.sqrt(max(st, 0.0)))

        # RMS scale √(|Ω|·T) used when ‖φ_Q‖≈0 (so “relative” makes no sense)
        area = float((x[-1]-x[0]) * (y[-1]-y[0]))
        time_len = float(t_hist_k[-1] - t_hist_k[0])
        rms_scale_xt = float(np.sqrt(max(area, 1e-30) * max(time_len, 1e-30)))

        # Tracking error: auto-normalize
        numQ = _l2_xyt(phi_k_plus_1 - phi_Q_target)
        denQ = _l2_xyt(phi_Q_target)
        if denQ < 1e-9 * rms_scale_xt:
            denQ = rms_scale_xt  # fall back to RMS when target trajectory is ~0
        tracking_error_history.append(numQ / (denQ + 1e-12))

        # Terminal error: relative to final target
        numT = _l2_xy(phi_k_plus_1[-1] - phi_T_target)
        denT = _l2_xy(phi_T_target) + 1e-12
        terminal_error_history.append(numT / denT)

        if k > 0 and abs(cost_history[-1] - cost_history[-2]) < PLATEAU_TOLERANCE:
            plateau_counter += 1
        else:
            plateau_counter = 0
        if plateau_counter >= PLATEAU_LENGTH:
            print(f"   [Notice] Cost has plateaued for {plateau_counter} iterations. Boosting step size.")
            alpha_prev, plateau_counter = min(alpha_max, alpha_k * 1.5), 0
        else:
            alpha_prev = min(alpha_max, alpha_k * 1.2)

        change = np.linalg.norm(u_k_plus_1 - u_k) / (np.linalg.norm(u_k) + 1e-9)
        iter_time = time.perf_counter() - iter_start
        print(f"   Relative control change: {change:.6e}\n   Iteration time: {iter_time:.2f}s")
        if change < 1e-5 and k > 20:
            print(f"\n🎉 Convergence reached at iteration {k+1}!")
            u_k, phi_k = u_k_plus_1, phi_k_plus_1
            break
        u_k, cost_k, phi_k, t_hist_k = u_k_plus_1, cost_k_plus_1, phi_k_plus_1, t_hist_k_plus_1

    # --- Final Analysis and Visualization ---
    print("\n" + "="*50 + "\n📊 GENERATING VISUALIZATIONS\n" + "="*50)
    u_optimal, phi_optimal_hist = u_k, phi_k
    phi_hist_natural, _, _ = run_main_simulation(config=fwd_config, store_history=True, control_input=None, verbose=False)
    phi_natural_final, phi_controlled_final = phi_hist_natural[-1], phi_optimal_hist[-1]
    total_runtime = format_time_hms(time.time() - start_time)
    param_string = (f"--- ⏱️ Optimization & Timing ---\nTotal Runtime: {total_runtime} (H:M:S)\n"
                    f"Completed Iterations: {len(cost_history)-1}\nFinal Cost: {cost_history[-1]:.5f}\n"
                    f"Cost Reduction: {100*(1 - cost_history[-1]/cost_history[0]):.2f}%\n\n"
                    f"--- Key Parameters ---\nGrid: {Nx}x{Ny} | Time: {T}s\n"
                    f"Cost Weights (b1, b2, b3): {b1}, {b2}, {b3}\nSparsity (κ_sparse): {kappa_sparsity:.1e}\n")
    print(param_string)
    create_comparison_panel(phi_initial_pristine, phi_natural_final, phi_controlled_final, phi_T_target, x, y)
    animate_time_evolution(phi_optimal_hist, x, y, t_hist_k)
    create_1d_slice_comparison(phi_initial_pristine, phi_controlled_final, phi_T_target, x)
    save_parameter_text_image(param_string, filename="simulation_parameters.png")
    plot_convergence_history(cost_history, terminal_error_history, tracking_error_history)
    generate_all_3d_plots(phi_initial_pristine, phi_natural_final, phi_controlled_final, phi_T_target, x, y)
        # ---------- TIME STUDY SUMMARY ----------
    print("\n" + "="*50 + "\n⏱️  TIME STUDY SUMMARY\n" + "="*50)
    def _fmt_sec(s): 
        try:
            return f"{s:.3f}s"
        except Exception:
            return str(s)
    print(f"Total backward-solve time:        {_fmt_sec(t_backward_total)}")
    print(f"Total optimistic forward time:    {_fmt_sec(t_optimistic_fwd_total)}")
    print(f"Total optimistic cost time:       {_fmt_sec(t_optimistic_cost_total)}")
    print(f"Total backtracking time:          {_fmt_sec(t_backtracking_total)}  "
          f"(calls={n_backtracking_calls}, attempts={n_backtracking_attempts})")
    print(f"Total time in accepted steps:     {_fmt_sec(t_successful_steps_total)}")
    print("="*50)

    if successful_optimistic_alphas:
        final_avg_alpha = np.mean(successful_optimistic_alphas)
        print("\n" + "=" * 60 + "\n💡 OPTIMIZATION TIP: ALPHA ADVISOR\n" +
              f"Based on {len(successful_optimistic_alphas)} successful optimistic steps,\n" +
              f"a good initial `alpha_max` for your next run is: {final_avg_alpha:.4f}\n" +
              "Using this value may lead to faster convergence.\n" + "=" * 60)

    try:
        print("\nRunning final backward solve for analysis...")
        _, _, r_optimal = run_backward(phi_optimal_hist, x, y, t_hist_k, fwd_config, b1, b2, phi_Q_target, phi_T_target)
        print("\n--- Checking Second-Order Sufficient Condition (Coercivity) ---")
        hessian_values = approximate_second_order_condition_2d(
            u_star=u_optimal, r_star=r_optimal, phi_star=phi_optimal_hist, x=x, y=y, t_hist=t_hist_k,
            b1=b1, b2=b2, b3=b3, kappa=kappa_sparsity, phi_Q_target=phi_Q_target, phi_T_target=phi_T_target,
            u_min=opt_config.u_min, u_max=opt_config.u_max, num_directions=5, epsilon=1e-4, seed=42,fwd_config=fwd_config
        )
        if all(val > 0 for val in hessian_values):
            print("\n✓ Coercivity condition appears to hold in tested directions.")
        else:
            print("\n⚠ Coercivity condition may fail; non-positive second derivatives found.")
        verify_sparsity_condition(u_optimal, r_optimal, kappa_sparsity)
    except Exception as e:
        print(f"\n[Warning] Could not perform final analysis (second-order/sparsity): {e}")
    save_params(fwd_config, opt_config, len(cost_history)-1)
    print("\n" + "="*50 + "\n✅ OPTIMIZATION COMPLETE\n" + "="*50)
