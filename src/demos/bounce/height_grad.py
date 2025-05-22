import time
import torch
import numpy as np
import matplotlib.pyplot as plt  # Still needed for the first plot and plt.show() / rcParams
from xitorch.optimize import rootfinder

# Import from utils.py
from demos.utils import create_scientific_subplot_plot
# Import available integrators from backward_euler.py
from demos.bounce.backward_euler import (
    fixed_step_backward_euler_integrator,
    adaptive_backward_euler_integrator,
    icpg_integrator,
    adaptive_icpg_integrator
)

# Specify the integrator to use:
# 'fixed_be' (fixed_step_backward_euler_integrator),
# 'adaptive_be' (adaptive_backward_euler_integrator),
# 'icpg' (icpg_integrator),
# 'adaptive_icpg' (adaptive_icpg_integrator)
INTEGRATOR_TYPE = 'adaptive_icpg'  # Options: 'fixed_be', 'adaptive_be', 'icpg', 'adaptive_icpg'
LOSS = 'l2_loss'  # Options: 'final_height_loss', 'l2_loss'
OUTPUT_PATH = f'height_gradient_{INTEGRATOR_TYPE}.png'


def final_height_loss(achieved_height, target_height):
    """
    Compute the loss as the squared difference between achieved height and target height.
    """
    loss = achieved_height
    return loss

def l2_loss(achieved_height, target_height):
    """
    Compute the L2 loss as the squared difference between achieved height and target height.
    """
    loss = (achieved_height - target_height) ** 2
    return loss

def get_integrator(integrator_name):
    """Returns the integrator function based on its name."""
    if integrator_name == 'fixed_be':
        return fixed_step_backward_euler_integrator
    elif integrator_name == 'adaptive_be':
        return adaptive_backward_euler_integrator
    elif integrator_name == 'icpg':
        return icpg_integrator
    elif integrator_name == 'adaptive_icpg':
        return adaptive_icpg_integrator
    else:
        raise ValueError(
            f"Unknown integrator type: {integrator_name}. "
            f"Choose 'fixed_be', 'adaptive_be', 'icpg', or 'adaptive_icpg'."
        )

def run_simulation(integrator_func, ts_or_tspan, y0_sim, sim_params_cg, res_coeff, integrator_name):
    """Runs the simulation with the chosen integrator, handling different signature requirements."""
    if integrator_name in ['adaptive_be', 'adaptive_icpg']:
        # Adaptive integrators expect t_span_val = (t_start, t_end)
        # and intregration_params (note the spelling difference from fixed step)
        t_span_val = (ts_or_tspan[0], ts_or_tspan[-1])
        # The adaptive integrators in backward_euler.py use 'intregration_params'
        # and expect dt_initial, num_predict_substeps, penetration_threshold
        # We'll use some default values here, these might need to be configurable
        times, yt_sim = integrator_func(
            t_span_val=t_span_val,
            y0=y0_sim,
            integration_params=sim_params_cg, # Make sure this matches what adaptive integrators expect
            restitution_coeff=res_coeff,
            dt_initial=0.01, # Default, make configurable if needed
            num_predict_substeps=100, # Default
            penetration_threshold=-0.001 # Default
        )
        # For consistency in return type with fixed-step, we return the states.
        # If `times` are needed by the caller, this function signature would need adjustment.
        return yt_sim
    else:
        # Fixed-step integrators expect ts (time steps array)
        return integrator_func(
            ts=ts_or_tspan,
            y0=y0_sim,
            integration_params=sim_params_cg,
            restitution_coeff=res_coeff
        )


def compute_loss_grad_bounce(py_init_val, target_h, time_steps_or_span, sim_params_cg, res_coeff=0.0):
    current_py_init = py_init_val if isinstance(py_init_val, torch.Tensor) else \
        torch.tensor(py_init_val, dtype=torch.float64, requires_grad=True)
    px0_fixed = torch.tensor(0.0, dtype=torch.float64)
    vx0_fixed = torch.tensor(1.0, dtype=torch.float64)
    vy0_fixed = torch.tensor(-1.0, dtype=torch.float64)

    y0_sim = torch.stack([px0_fixed, current_py_init, vx0_fixed, vy0_fixed])

    integrator_func = get_integrator(INTEGRATOR_TYPE)
    yt_sim = run_simulation(integrator_func, time_steps_or_span, y0_sim, sim_params_cg, res_coeff, INTEGRATOR_TYPE)

    if LOSS == 'l2_loss':
        loss = l2_loss(yt_sim[-1, 1], target_h)
    elif LOSS == 'final_height_loss':
        loss = final_height_loss(yt_sim[-1, 1], target_h)
    else:
        raise ValueError(f"Unknown loss type: {LOSS}. Use 'final_height_loss' or 'l2_loss'.")

    return loss, yt_sim


def main():
    # --- TeX Configuration ---
    tex_config_setup = {
        'use_tex': True,
        'fonts': 'serif',
        'fontsize': 12,
        'custom_preamble': True,
        'preamble': r'''
            \usepackage{amsmath,amssymb,amsfonts}
            \usepackage{physics}
            \usepackage{siunitx}
        '''
    }

    if tex_config_setup['use_tex']:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": tex_config_setup['fonts'],
            "font.size": tex_config_setup['fontsize']
        })
        if tex_config_setup['custom_preamble'] and tex_config_setup['preamble']:
            plt.rcParams["text.latex.preamble"] = tex_config_setup['preamble']
    # --- End TeX Configuration ---

    px0_main = 0.0
    py_initial_for_ref_calc = 1.0
    vx0_main = 1.0
    vy0_main = -1.0
    dt_main = 0.01
    sim_time_main = 2.0
    steps_main = int(sim_time_main / dt_main)
    restitution_main = 1.0

    gravity_val = torch.tensor(0.0, dtype=torch.float64)
    sim_params_main = [gravity_val]

    # For fixed step integrators
    ts_main = torch.linspace(0, sim_time_main, steps_main + 1, dtype=torch.float64)
    # For adaptive integrators (t_start, t_end)
    t_span_main = (torch.tensor(0.0, dtype=torch.float64), torch.tensor(sim_time_main, dtype=torch.float64))

    time_input = ts_main # Default to ts_main for fixed step
    if INTEGRATOR_TYPE in ['adaptive_be', 'adaptive_icpg']:
        time_input = t_span_main


    y0_for_ref_calc = torch.tensor([px0_main, py_initial_for_ref_calc, vx0_main, vy0_main], dtype=torch.float64)
    print(f"Calculating reference trajectory using {INTEGRATOR_TYPE} integrator...")

    integrator_func_ref = get_integrator(INTEGRATOR_TYPE)
    # Note: The adaptive integrators from backward_euler.py return (times, states)
    # The fixed-step integrators return states. We need to handle this.
    # For the reference trajectory, we only need the states.

    yt_ref_main = run_simulation(integrator_func_ref, time_input, y0_for_ref_calc, sim_params_main, restitution_main, INTEGRATOR_TYPE)

    actual_ref_height = yt_ref_main[-1, 1].detach()
    print(f"Reference trajectory from y0={py_initial_for_ref_calc:.2f}m achieves: {actual_ref_height.numpy():.6f} m.")

    pos_np_ref_main = yt_ref_main[:, :2].detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(pos_np_ref_main[:, 0], pos_np_ref_main[:, 1], 'b-',
             label=fr'Example Trajectory ($py_0={py_initial_for_ref_calc:.2f}$m, Integrator: {INTEGRATOR_TYPE})')
    plt.scatter(pos_np_ref_main[0, 0], pos_np_ref_main[0, 1], c='g', s=100, label='Start')
    plt.scatter(pos_np_ref_main[-1, 0], pos_np_ref_main[-1, 1], c='r', s=100, label='End')
    plt.axhline(y=actual_ref_height.numpy(), color='k', linestyle='--',
                label=fr'Achieved Height (${actual_ref_height.numpy():.2f}$ m)')
    plt.xlabel(r'X Position (m)')
    plt.ylabel(r'Y Position (m)')
    plt.title(r'Example Trajectory and Achieved Height')
    plt.legend()
    plt.grid(True)
    plt.savefig('target_trajectory_example_tex.png')
    plt.close()

    py_range_sweep = np.linspace(0.5, 1.5, 90)
    achieved_height_values_sweep = []
    grad_values_sweep = []

    print(f"\nCalculating achieved heights and gradients over a range of initial Y positions using {INTEGRATOR_TYPE} integrator...")
    for py_idx, py_val_sweep in enumerate(py_range_sweep):
        py_tensor_sweep = torch.tensor(py_val_sweep, dtype=torch.float64, requires_grad=True)

        # Use the same time_input (ts_main or t_span_main) for the sweep
        achieved_height, _ = compute_loss_grad_bounce(py_tensor_sweep, actual_ref_height, time_input,
                                                      sim_params_cg=sim_params_main, res_coeff=restitution_main)

        grad_py_val_sweep = torch.tensor(0.0, dtype=torch.float64)
        if achieved_height.requires_grad:
            try:
                grad_py_list = torch.autograd.grad(outputs=achieved_height, inputs=py_tensor_sweep,
                                                   allow_unused=False)
                if grad_py_list[0] is not None:
                    grad_py_val_sweep = grad_py_list[0]
                else:
                    print(f"Warning: Gradient is None for py={py_val_sweep:.2f} despite requires_grad=True on output.")
            except RuntimeError as e:
                print(f"RuntimeError during autograd.grad for py={py_val_sweep:.2f}: {e}")
        else:
            print(f"Warning: achieved_height for py={py_val_sweep:.2f} does not require grad. Grad will be 0.")

        print(
            f"  Initial y: {py_val_sweep:.2f}, Achieved Height: {achieved_height.item():.6f}, Gradient: {grad_py_val_sweep.item():.6f} ({py_idx + 1}/{len(py_range_sweep)})")
        achieved_height_values_sweep.append(achieved_height.item())
        grad_values_sweep.append(grad_py_val_sweep.item())

    print("\nPlotting with scientific plotter...")
    plot_data_scientifc = {
        r'Achieved Final Height $h^k$': {
            'achieved_h_series': {
                'data': np.array(achieved_height_values_sweep),
                'label': r'Simulated $h^k$',
                'color': 'blue',
                'linewidth': 2,
            }
        },
        r'Gradient $\grad h^k$': {
            'gradient_series': {
                'data': np.array(grad_values_sweep),
                'label': r'$\displaystyle\pdv{h^k}{h^0}$',
                'color': 'red',
                'linewidth': 2,
            }
        }
    }

    config_scientific = {
        'plot_arrangement': 'horizontal',
        'figsize': (10, 5),
        'suptitle': fr'Integrator: {INTEGRATOR_TYPE.replace("_", " ").title()}',
        'x_label': r'Initial Height $h^0$ (m)',
        'y_labels': {
            r'Achieved Final Height $h^k$': r'$h^k$ (m)',
            r'Gradient $\grad h^k$': r'$\displaystyle\pdv{h^k}{h_0}$ (-)'
        },
        'plot_titles': {
            r'Achieved Final Height $h^k$': r'Final Height vs. Initial Height',
            r'Gradient $\grad h^k$': r'Gradient of Final Height wrt. Initial Height'
        },
        'legend': {'show': True, 'location': 'best', 'fontsize': 12},
        'grid': {'show': True, 'linestyle': '--', 'alpha': 0.8},
        'font_sizes': {'axis_label': 12, 'tick_label': 12, 'title': 13, 'suptitle': 15},
    }

    fig_scientific, axes_scientific = create_scientific_subplot_plot(
        time_data=py_range_sweep,
        plot_data=plot_data_scientifc,
        config=config_scientific,
        save_path=OUTPUT_PATH,
    )
    print(f"Scientific plot saved to {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()