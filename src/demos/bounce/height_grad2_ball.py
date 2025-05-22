import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from demos.utils import create_scientific_subplot_plot
from demos.bounce.backward_euler import (
    fixed_step_backward_euler_integrator,
    icpg_integrator,
    adaptive_icpg_integrator
)

ALL_INTEGRATOR_TYPES = ['icpg', 'adaptive_icpg']
ALL_INTEGRATOR_TYPES = ['icpg']
LOSS = 'l2_loss'

# --- Loss Functions ---
def final_height_loss(achieved_height_tensor, target_height_tensor):
    return achieved_height_tensor

def l2_loss(achieved_height_tensor, target_height_tensor):
    return (achieved_height_tensor - target_height_tensor) ** 2

# --- Integrator Getter ---
def get_integrator(integrator_name):
    if integrator_name == 'fixed_be':
        return fixed_step_backward_euler_integrator
    elif integrator_name == 'icpg':
        return icpg_integrator
    elif integrator_name == 'adaptive_icpg':
        return adaptive_icpg_integrator
    else:
        raise ValueError(f"Unknown integrator type: {integrator_name}. Choose from {ALL_INTEGRATOR_TYPES}.")

# --- Simulation Runner ---
def run_simulation(integrator_func, ts_or_tspan, y0_sim, integration_params_list, res_coeff, integrator_name, full_params_dict):
    if 'adaptive' in integrator_name:
        t_span_val = (ts_or_tspan[0], ts_or_tspan[-1])
        dt_initial_val = full_params_dict.get('base_dt', 0.01)
        num_predict_substeps_val = full_params_dict.get('num_predict_substeps', 100)
        penetration_threshold_val = full_params_dict.get('penetration_threshold', -0.001)
        if 'icpg' in integrator_name:
            min_k_eff_val = full_params_dict.get('min_k_eff', 0.0)
            max_k_eff_val = full_params_dict.get('max_k_eff', 1e6)
            min_c_eff_val = full_params_dict.get('min_c_eff', 0.0)
            max_c_eff_val = full_params_dict.get('max_c_eff', 1e3)
            w_k_val = full_params_dict.get('w_k', 1e-4)
            w_c_val = full_params_dict.get('w_c', 1.0)
            output = integrator_func(
                t_span=t_span_val, y0=y0_sim, integration_params=integration_params_list,
                restitution_coeff=res_coeff, base_dt=dt_initial_val, num_predict_substeps=num_predict_substeps_val,
                penetration_threshold=penetration_threshold_val, min_k_eff=min_k_eff_val, max_k_eff=max_k_eff_val,
                min_c_eff=min_c_eff_val, max_c_eff=max_c_eff_val, w_k=w_k_val, w_c=w_c_val
            )
        return output[1] if isinstance(output, tuple) and len(output) == 2 else output
    else:
        if 'icpg' in integrator_name:
            min_k_eff_val = full_params_dict.get('min_k_eff', 0.0)
            max_k_eff_val = full_params_dict.get('max_k_eff', 1e6)
            min_c_eff_val = full_params_dict.get('min_c_eff', 0.0)
            max_c_eff_val = full_params_dict.get('max_c_eff', 1e3)
            w_k_val = full_params_dict.get('w_k', 1e-4)
            w_c_val = full_params_dict.get('w_c', 1.0)
            return integrator_func(
                ts=ts_or_tspan, y0=y0_sim, integration_params=integration_params_list,
                restitution_coeff=res_coeff, min_k_eff=min_k_eff_val, max_k_eff=max_k_eff_val,
                min_c_eff=min_c_eff_val, max_c_eff=max_c_eff_val, w_k=w_k_val, w_c=w_c_val
            )
        else:
            return integrator_func(ts=ts_or_tspan, y0=y0_sim, integration_params=integration_params_list, restitution_coeff=res_coeff)

def compute_loss_grad_bounce(py_init_val, target_h_tensor, time_steps_or_span, integration_params_list_for_sim,
                             res_coeff, integrator_type_arg, full_params_dict_for_sim):
    current_py_init = py_init_val if isinstance(py_init_val, torch.Tensor) else torch.tensor(py_init_val, dtype=torch.float64, requires_grad=True)
    px0_fixed = torch.tensor(0.0, dtype=torch.float64)
    vx0_fixed = torch.tensor(1.0, dtype=torch.float64)
    vy0_fixed = torch.tensor(-1.0, dtype=torch.float64)
    y0_sim = torch.stack([px0_fixed, current_py_init, vx0_fixed, vy0_fixed])
    integrator_func = get_integrator(integrator_type_arg)
    yt_sim = run_simulation(integrator_func, time_steps_or_span, y0_sim, integration_params_list_for_sim,
                            res_coeff, integrator_type_arg, full_params_dict_for_sim)
    achieved_final_y_pos = yt_sim[-1, 1]
    print(f"  {integrator_type_arg} - Initial y: {current_py_init.item():.2f}, final y: {achieved_final_y_pos.item():.6f}, target y: {target_h_tensor.item():.6f}")
    if LOSS == 'l2_loss':
        loss = l2_loss(achieved_final_y_pos, target_h_tensor)
    elif LOSS == 'final_height_loss':
        loss = final_height_loss(achieved_final_y_pos, target_h_tensor)
    else:
        raise ValueError(f"Unknown loss type: {LOSS}.")
    return loss, current_py_init

def analytical_height(x0: torch.Tensor, v0: torch.Tensor, t: float, restitution: float):
    t_impact = (1.0 - x0[1]) / v0[1]
    if t < t_impact:
        return x0[1] + v0[1] * t
    elif t == t_impact:
        return 0.0
    else:
        return 1.0 * (restitution + 1) - restitution * v0[1] * t - restitution * x0[1]

def analytical_gradient(x0: torch.Tensor, v0: torch.Tensor, t: float, restitution: float, target_height: torch.Tensor):
    t_impact = (1.0 - x0[1]) / v0[1]
    predicted_height = analytical_height(x0, v0, t, restitution)
    if t < t_impact:
        dheight_dh0 = 1.0
    elif t == t_impact:
        dheight_dh0 = 0.0
    else:
        dheight_dh0 = -restitution
    return 2 * (predicted_height - target_height) * dheight_dh0

# --- Main Function ---
def main():
    tex_config_setup = {
        'use_tex': True,
        'fonts': 'serif',
        'fontsize': 12,
        'custom_preamble': True,
        'preamble': r'\usepackage{amsmath,amssymb,amsfonts}\usepackage{physics}\usepackage{siunitx}'
    }
    if tex_config_setup['use_tex']:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": tex_config_setup['fonts'],
            "font.size": tex_config_setup['fontsize']
        })
        if tex_config_setup['custom_preamble'] and tex_config_setup['preamble']:
            plt.rcParams["text.latex.preamble"] = tex_config_setup['preamble']

    # --- Simulation Parameters ---
    x0_x = 0.0
    x0_y_gt = 4.5
    x0 = torch.tensor([x0_x, x0_y_gt], dtype=torch.float64)
    v0_x = 5.0
    v0_y = -5.0
    v0 = torch.tensor([v0_x, v0_y], dtype=torch.float64)
    dt = 0.01
    sim_time = 1.0
    num_steps = int(sim_time / dt)
    restitution = 1.0
    g_accel = torch.tensor(0.0, dtype=torch.float64)
    simulation_parameters_dict = {
        'gravity': g_accel,
        'base_dt': dt,
        'num_predict_substeps': 100,
        'penetration_threshold': -0.001,
        'min_k_eff': 0.0,
        'max_k_eff': 1e6,
        'min_c_eff': 0.0,
        'max_c_eff': 1e3,
        'w_k': 1e-4,
        'w_c': 1.0
    }
    integration_params_list_arg = [simulation_parameters_dict['gravity']]
    ts = torch.linspace(0, sim_time, num_steps + 1, dtype=torch.float64)
    t_span = (torch.tensor(0.0, dtype=torch.float64), torch.tensor(sim_time, dtype=torch.float64))

    # --- Determine Target Height ---
    target_height = analytical_height(x0, v0, sim_time, restitution)
    target_height_actual = target_height.detach().clone()
    print(f"Target height from y0={x0_y_gt:.2f}m is: {target_height_actual.numpy():.6f} m.")

    # --- Sweep through initial y positions ---
    py_range_sweep = np.linspace(3.3, 5.7, 31)

    # Compute analytical loss and gradient
    analytical_losses = []
    analytical_gradients = []
    for py_val_sweep in py_range_sweep:
        x0_sweep = torch.tensor([0.0, py_val_sweep], dtype=torch.float64)
        analytical_h = analytical_height(x0_sweep, v0, sim_time, restitution)
        analytical_loss = (analytical_h - target_height_actual) ** 2
        analytical_grad = analytical_gradient(x0_sweep, v0, sim_time, restitution, target_height_actual)
        analytical_losses.append(analytical_loss.item())
        analytical_gradients.append(analytical_grad.item())

    results_data = {}
    for integrator_name in ALL_INTEGRATOR_TYPES:
        print(f"\nProcessing Integrator: {integrator_name}")
        current_loss_values = []
        current_gradients = []
        current_comp_times = []
        time_input_for_current_integrator = ts if integrator_name not in ['adaptive_icpg'] else t_span
        for py_idx, py_val_sweep in enumerate(py_range_sweep):
            loop_start_time = time.time()
            loss_value, py_tensor_for_grad = compute_loss_grad_bounce(
                py_val_sweep, target_height_actual, time_input_for_current_integrator,
                integration_params_list_arg, restitution, integrator_name, simulation_parameters_dict
            )
            grad_py_val = torch.tensor(0.0, dtype=torch.float64)
            if loss_value.requires_grad:
                try:
                    grad_list = torch.autograd.grad(outputs=loss_value, inputs=py_tensor_for_grad, allow_unused=False, create_graph=False)
                    if grad_list[0] is not None:
                        grad_py_val = grad_list[0]
                except RuntimeError as e:
                    print(f"  RuntimeError during autograd.grad for {integrator_name}, py={py_val_sweep:.2f}: {e}")
            else:
                print(f"  Warning: loss_value for {integrator_name}, py={py_val_sweep:.2f} does not require grad.")
            loop_end_time = time.time()
            current_comp_times.append(loop_end_time - loop_start_time)
            current_loss_values.append(loss_value.item())
            current_gradients.append(grad_py_val.item())
            print(f"  {integrator_name} - Initial y: {py_val_sweep:.2f}, L2 Loss: {loss_value.item():.6e}, Grad: {grad_py_val.item():.6e} ({py_idx + 1}/{len(py_range_sweep)})")
        results_data[integrator_name] = {
            'loss_values': np.array(current_loss_values),
            'grads': np.array(current_gradients),
            'avg_time': np.mean(current_comp_times) if current_comp_times else 0
        }
        if current_comp_times:
            print(f"Integrator {integrator_name} average computation time per py_val: {results_data[integrator_name]['avg_time']:.4f}s")

    # --- Plotting with create_scientific_subplot_plot ---
    plot_data_for_scientific = {}
    y_labels_for_scientific = {}
    plot_titles_for_scientific = {}
    integrator_config = {
        # 'fixed_be': {'label': 'Backward Euler', 'color': '#1f77b4', 'linewidth': 2.5},
        # 'icpg': {'label': 'ICPG', 'color': '#ff7f0e', 'linewidth': 2.5},
        # 'adaptive_icpg': {'label': 'Adaptive ICPG', 'color': '#2ca02c', 'linewidth': 2.5}
    }

    # 1. Loss Subplot
    loss_subplot_title_key = r'Loss Value'
    plot_data_for_scientific[loss_subplot_title_key] = {}
    plot_data_for_scientific[loss_subplot_title_key]['analytical_loss_series'] = {
        'data': np.array(analytical_losses),
        'label': 'Analytical Loss',
        'color': '#1f77b4',
        'linewidth': 2.0,
        'linestyle': '--'
    }
    # for integrator, integrator_cfg in integrator_config.items():
    #     series_name_key = f'{integrator}_loss_series'
    #     plot_data_for_scientific[loss_subplot_title_key][series_name_key] = {
    #         'data': results_data[integrator]['loss_values'],
    #         'label': integrator_cfg['label'],
    #         'color': integrator_cfg['color'],
    #         'linewidth': integrator_cfg['linewidth'],
    #         'alpha': 0.8
    #     }

    y_labels_for_scientific[loss_subplot_title_key] = r'Loss (-)'
    plot_titles_for_scientific[loss_subplot_title_key] = r'Loss vs. Initial Height'

    # 2. Separate Analytical Gradient Subplot
    analytical_grad_subplot_title_key = r'Analytical Gradient'
    plot_data_for_scientific[analytical_grad_subplot_title_key] = {
        'analytical_grad_series': {
            'data': np.array(analytical_gradients),
            'label': r'Analytical $\displaystyle\pdv{l}{x_z^0}$',
            'color': '#ff7f0e',
            'linewidth': 2.0,
            'linestyle': '--'
        }
    }
    y_labels_for_scientific[analytical_grad_subplot_title_key] = r'$\displaystyle\pdv{l}{x_z^0}$ (-)'
    plot_titles_for_scientific[analytical_grad_subplot_title_key] = r'Analytical Gradient'

    # # 3. Integrator Gradient Subplots
    # for integrator, integrator_cfg in integrator_config.items():
    #     grad_subplot_title_key = fr'Gradient: {integrator_cfg["label"]}'
    #     plot_data_for_scientific[grad_subplot_title_key] = {
    #         f'{integrator}_grad_series': {
    #             'data': results_data[integrator]['grads'],
    #             'label': r'$\displaystyle\pdv{l}{h^0}$',
    #             'color': integrator_cfg['color'],
    #             'linewidth': integrator_cfg['linewidth'],
    #         }
    #     }
    #     y_labels_for_scientific[grad_subplot_title_key] = r'$\displaystyle\pdv{l}{h^0}$ (-)'
    #     plot_titles_for_scientific[grad_subplot_title_key] = fr'Gradient for {integrator_cfg["label"]}'

    num_total_subplots = len(ALL_INTEGRATOR_TYPES) + 1  # Loss + Integrator Gradients + Analytical Gradient
    plot_config_scientific = {
        'plot_arrangement': 'horizontal',
        'figsize': (5 * num_total_subplots, 5),
        'suptitle': fr'',
        'x_label': r'Initial $z$-Position $x_z^0$ (m)',
        'y_labels': y_labels_for_scientific,
        'plot_titles': plot_titles_for_scientific,
        'shared_x': True,
        'shared_y': False,
        'dpi': 200,
        'legend': {'show': True, 'location': 'best', 'fontsize': 13},
        'grid': {'show': True, 'linestyle': '--', 'alpha': 0.8},
        'font_sizes': {'axis_label': 13, 'tick_label': 13, 'title': 13, 'suptitle': 14},
        'tight_layout_params': {'rect': [0, 0.02, 1, 0.95]}
    }

    scientific_plot_filename = f'comparison_height_grad.png'
    fig_sci, _ = create_scientific_subplot_plot(
        time_data=py_range_sweep,
        plot_data=plot_data_for_scientific,
        config=plot_config_scientific,
        save_path=scientific_plot_filename
    )
    print(f"Scientific comparison plot saved to {scientific_plot_filename}")
    plt.close(fig_sci)

    # --- Performance Bar Chart ---
    print("\n--- Performance Summary ---")
    integrator_names_for_bar_chart = [integrator_name.replace("_", " ").title() for integrator_name in ALL_INTEGRATOR_TYPES]
    avg_times_for_bar_chart = [results_data[integrator_name]['avg_time'] for integrator_name in ALL_INTEGRATOR_TYPES]
    for i, integrator_name in enumerate(ALL_INTEGRATOR_TYPES):
        print(f"{integrator_name:<25} | Avg. Time/Sim (s): {results_data[integrator_name]['avg_time']:<20.5f}")
    fig_perf, ax_perf = plt.subplots(figsize=(max(8, 3 * len(ALL_INTEGRATOR_TYPES)), 5))
    bar_positions = np.arange(len(integrator_names_for_bar_chart))
    ax_perf.bar(bar_positions, avg_times_for_bar_chart, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
    ax_perf.set_xticks(bar_positions)
    ax_perf.set_xticklabels(integrator_names_for_bar_chart, rotation=30, ha='right', fontsize=9)
    ax_perf.set_ylabel('Average Time per Simulation (s)', fontsize=10)
    ax_perf.set_title('Integrator Computation Time Comparison', fontsize=12)
    ax_perf.grid(True, axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(avg_times_for_bar_chart):
        ax_perf.text(i, v + 0.01 * max(avg_times_for_bar_chart) if avg_times_for_bar_chart else 0.01, f"{v:.3f}", 
                     color='blue', ha='center', va='bottom', fontsize=8)
    fig_perf.tight_layout()
    performance_plot_filename = f'comparison_performance_times_{LOSS}.png'
    plt.savefig(performance_plot_filename, dpi=300)
    print(f"Performance plot saved to {performance_plot_filename}")
    plt.close(fig_perf)

if __name__ == "__main__":
    main()