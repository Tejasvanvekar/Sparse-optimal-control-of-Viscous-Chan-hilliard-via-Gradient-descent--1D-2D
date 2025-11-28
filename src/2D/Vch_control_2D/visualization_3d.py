"""
visualization_3d.py
===================

This module contains helper routines for visualizing the results of the
2D Cahn–Hilliard optimal control problem.  The functions here generate
high‑quality static and animated plots, including 3D surface plots of
phase fields, convergence histories, time‑lapse animations of the
evolution, and comparison panels highlighting the effect of control.

The visualizations are primarily used by :mod:`GD2_configured` after
completing the gradient descent.  They save images to disk rather than
displaying them interactively, which is suitable for headless batch runs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings



def _show_forward_final_imshow(phi_final, x, y, T):
    """Show the same imshow-style final φ plot used in the forward solver (no saving)."""
    plt.figure(figsize=(6, 5))
    extent = [x[0], x[-1], y[0], y[-1]]
    plt.imshow(
        phi_final.T, origin="lower", extent=extent,
        vmin=-1.0, vmax=1.0, cmap="RdBu_r", interpolation="bilinear"
   )
    plt.title(f"Final Profile of φ at t={T}", fontsize=16)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.colorbar(label="φ")
    plt.tight_layout()
    plt.show()
    return None


def create_3d_surface_plot(Z_data, x, y, title, filename):
    """
    Generates and saves a high-definition 3D surface plot.

    Notes:
    - Uses indexing='ij' so X.shape == (len(x), len(y)).
    - Z_data is expected to be shape (len(x), len(y)) — NO transpose needed here.
    """
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = Z_data  # keep native orientation to match (X, Y)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Create the surface plot with a high-quality colormap and fine grid
    surf = ax.plot_surface(
        X, Y, Z, cmap='viridis',
        rstride=2, cstride=2,
        linewidth=0.1, antialiased=True, alpha=0.95
    )

    # Labels and title
    ax.set_xlabel('x-axis', fontsize=15, labelpad=10)
    ax.set_ylabel('y-axis', fontsize=15, labelpad=10)
    ax.set_zlabel('Phase Field (φ)', fontsize=16, labelpad=10)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Consistent z-limits
    ax.set_zlim(-1.1, 1.1)

    # Colorbar
    cbar = fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label('φ value', fontsize=12)

    # View angle
    ax.view_init(elev=25., azim=-65)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved 3D surface plot: {filename}")


def generate_all_3d_plots(phi_initial, phi_natural_final, phi_controlled_final, phi_target, x, y):
    """
    Generate and save the set of 3D surface plots.

    Order matches main script (GD2_configured.py):
        generate_all_3d_plots(phi_initial, phi_natural_final, phi_controlled_final, phi_target, x, y)
    """
    create_3d_surface_plot(
        Z_data=phi_initial,
        x=x, y=y,
        title="1. Initial State (t=0)",
        filename="3d_plot_initial_state.png"
    )
    create_3d_surface_plot(
        Z_data=phi_natural_final,
        x=x, y=y,
        title="2. Natural Evolution (Final State with u=0)",
        filename="3d_plot_natural_evolution.png"
    )
    create_3d_surface_plot(
        Z_data=phi_target,
        x=x, y=y,
        title="3. Target State (The Goal)",
        filename="3d_plot_target_state.png"
    )
    create_3d_surface_plot(
        Z_data=phi_controlled_final,
        x=x, y=y,
        title="4. Controlled Evolution (Final State with u*)",
        filename="3d_plot_controlled_evolution.png"
    )


def plot_convergence_history(cost_hist, terminal_err_hist, tracking_err_hist):
    """Generates a 2-panel plot showing the convergence of cost and errors."""
    warnings.filterwarnings('ignore')
    iterations = np.arange(len(cost_hist))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Subplot 1: Total Cost ---
    ax1.plot(iterations, cost_hist, 'o-', color='black', label='Total Cost (J)', markersize=3)
    ax1.set_ylabel('Total Cost', fontsize=12)
    ax1.set_title('Convergence Analysis Over Iterations', fontsize=16, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Subplot 2: Relative Errors (Log Scale) ---
    if len(terminal_err_hist) > 0:
        ax2.plot(np.arange(1, len(terminal_err_hist) + 1), terminal_err_hist, 'o-', color='crimson',
                 label='Terminal Error ||φ(T) - φ_target||', markersize=3)
    if len(tracking_err_hist) > 0:
        ax2.plot(np.arange(1, len(tracking_err_hist) + 1), tracking_err_hist, 'o--', color='dodgerblue',
                 label='Tracking Error ||φ - φ_Q||', markersize=3)

    ax2.set_yscale('log')
    ax2.set_ylabel('Relative L2 Error (log scale)', fontsize=15)
    ax2.set_xlabel('Iteration', fontsize=15)
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('convergence_analysis_plot.png', dpi=200)
    plt.close(fig)
    print("✓ Generated convergence analysis plot: 'convergence_analysis_plot.png'")


def save_parameter_text_image(param_text, filename="simulation_parameters.png"):
    """Saves the provided parameter description text to a separate image file."""
    if not param_text:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    fig.text(0.01, 0.5, param_text, va='center', ha='left', fontsize=12, family='monospace')
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved parameter info: '{filename}'")


def animate_time_evolution(phi_history, x, y, t_hist, out="phase_separation_timelapse.mp4"):
    """Creates a timelapse video of the phase separation process over time."""
    warnings.filterwarnings('ignore')
    print("\n--- Creating Timelapse Video of Phase Separation ---")
    if len(phi_history) == 0:
        return

    extent = [x[0], x[-1], y[0], y[-1]]
    vlim = 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    # Use .T to match orientation with imshow/contour plots
    im = ax.imshow(phi_history[0].T, origin='lower', extent=extent,
                   vmin=-vlim, vmax=vlim, cmap='viridis', interpolation='bilinear')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Phase Field (φ)")
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, color='white', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    def update(frame_index):
        im.set_data(phi_history[frame_index].T)
        time_text.set_text(f'Time = {t_hist[frame_index]:.3f}s')
        ax.set_title("Controlled Evolution Timelapse")
        return [im, time_text]

    skip_rate = max(1, len(phi_history) // 200)
    frames_to_render = range(0, len(phi_history), skip_rate)

    ani = FuncAnimation(fig, update, frames=frames_to_render, interval=50, blit=True)
    try:
        ani.save(out, writer="ffmpeg", dpi=150, bitrate=2000)
        print(f"✓ Saved timelapse animation: {out}")
    except Exception as e:
        gif_out = out.replace(".mp4", ".gif")
        print(f"ffmpeg failed ({e}); trying to save GIF animation...")
        ani.save(gif_out, writer="pillow", dpi=120)
        print(f"✓ Saved GIF animation: {gif_out}")
    plt.close(fig)


def create_comparison_panel(phi_initial, phi_natural_final, phi_controlled_final, phi_target, x, y):
    """Creates a 2x2 panel to show the effectiveness of the control."""
    print("\n--- Creating 4-Panel Comparison Visualization ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 13), constrained_layout=True)
    fig.suptitle("Demonstrating the Effect of Optimal Control", fontsize=18, fontweight='bold')

    extent = [x[0], x[-1], y[0], y[-1]]
    vlim = 1.0
    cmap = 'viridis'
    interpolation = 'bilinear'

    data_map = {
        (0, 0): (phi_initial, "1. Initial State (t=0)"),
        (0, 1): (phi_natural_final, "2. Natural Evolution (u=0)"),
        (1, 0): (phi_target, "3. Target State (The Goal)"),
        (1, 1): (phi_controlled_final, "4. Controlled Evolution (u=u*)"),
    }

    for (row, col), (data, title) in data_map.items():
        ax = axes[row, col]
        im = ax.imshow(data.T, origin='lower', extent=extent,
                       vmin=-vlim, vmax=vlim, cmap=cmap, interpolation=interpolation)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("x", fontsize=15)
        ax.set_ylabel("y", fontsize=15)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("φ", fontsize=15)

    # Add a white dashed contour of the target on the final controlled plot
    ax = axes[1, 1]
    ax.contour(x, y, phi_target.T, levels=[0.0], colors='white',
               linewidths=2.0, linestyles='--', alpha=0.7)
    ax.text(0.05, 0.95, "White dashed: target contour",
            transform=ax.transAxes, color='white', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
            verticalalignment='top')

    plt.savefig("control_effectiveness_panel_HD.png", dpi=200, bbox_inches='tight')
    print("✓ Saved 4-panel comparison plot: 'control_effectiveness_panel_HD.png'")
    plt.close(fig)


def create_1d_slice_comparison(phi_initial, phi_controlled_final, phi_target, x):
    """Creates the 1D comparison plot by taking a cross-section from the middle."""
    print("\n--- Creating 1D Cross-Section Comparison Plot ---")

    mid_y = phi_initial.shape[1] // 2
    slice_initial = phi_initial[:, mid_y]
    slice_final = phi_controlled_final[:, mid_y]
    slice_target = phi_target[:, mid_y]

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(x, slice_initial, ':', color='gray', label='Initial State (t=0)', linewidth=2.5, alpha=0.8)
    ax.plot(x, slice_target, '--', color='red', label='Target State', linewidth=3, alpha=0.9)
    ax.plot(x, slice_final, '-', color='blue', label='Final Controlled State (t=T)', linewidth=3.5)
    ax.fill_between(x, slice_target, slice_final, alpha=0.2, color='purple', label='Difference (Target - Final)')

    ax.set_title('Effect of Optimal Control: 1D Cross-Section at y=L/2', fontsize=16, fontweight='bold')
    ax.set_xlabel('Spatial Position x', fontsize=15)
    ax.set_ylabel('Phase Field φ', fontsize=15)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    mse = np.mean((slice_final - slice_target)**2)
    ax.text(0.02, 0.98, f'MSE: {mse:.4e}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig("phi_comparison_plot_1D_SLICE_HD.png", dpi=200, bbox_inches='tight')
    print("✓ Saved 1D slice comparison plot: 'phi_comparison_plot_1D_SLICE_HD.png'")
    plt.close(fig)


def format_time_hms(seconds):
    """Converts seconds to a formatted HH:MM:SS string."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
