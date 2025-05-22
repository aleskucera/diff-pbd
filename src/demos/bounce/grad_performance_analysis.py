import time
import torch
import numpy as np
import matplotlib.pyplot as plt  # Typically imported by utils but good to have if direct plt use is needed
from xitorch.optimize import rootfinder

# Assuming utils.py is in the same directory or Python path
from demos.utils import create_scientific_subplot_plot
from demos.bounce.backward_euler import fixed_step_backward_euler_integrator
from demos.bounce.backward_euler import icpg_integrator

def run_timing_experiment():
    """Runs the timing experiment and plots the results."""

    # Experiment Configuration
    sim_time_total = 1.0  # Total simulation time
    num_steps_options = [10, 50, 100, 200, 400, 800]

    recorded_times_no_grad = []
    recorded_times_fwd_pass_graph_enabled = []
    recorded_times_bwd_pass_gradient_calc = []
    recorded_times_total_with_grad = []

    # Fixed initial conditions and physical parameters for all runs
    initial_x_pos = 0.0
    initial_y_pos_val = 2.0
    initial_x_vel = 1.0
    initial_y_vel = 1.0
    gravity_g = torch.tensor(9.81, dtype=torch.float64)
    simulation_params = [gravity_g]  # Params for backward_euler_integrator
    restitution_val = 0.8
    dummy_target_height = 0.5  # For loss calculation

    print("Starting timing experiment...")
    for steps_count in num_steps_options:
        print(f"  Processing for N_steps = {steps_count}")
        current_dt = sim_time_total / steps_count
        time_steps_ts = torch.linspace(0, sim_time_total, steps_count + 1, dtype=torch.float64)

        # --- Measurement 1: Simulation without Gradient Tracking ---
        y0_no_grad_setup = torch.tensor([initial_x_pos, initial_y_pos_val, initial_x_vel, initial_y_vel],
                                        dtype=torch.float64, requires_grad=False)

        t_start_no_grad = time.perf_counter()
        with torch.no_grad():
            _ = icpg_integrator(time_steps_ts,
                                y0_no_grad_setup,
                                integration_params=simulation_params,
                                restitution_coeff=restitution_val)
        t_end_no_grad = time.perf_counter()
        time_taken_no_grad = t_end_no_grad - t_start_no_grad
        recorded_times_no_grad.append(time_taken_no_grad)
        print(f"    Time (no_grad): {time_taken_no_grad:.6f} s")

        # --- Measurement 2: Simulation with Gradient Tracking and Computation ---
        # Initial y position is the variable we'll compute gradient w.r.t.
        py_init_tensor_for_grad = torch.tensor(initial_y_pos_val, dtype=torch.float64, requires_grad=True)
        y0_with_grad_setup = torch.stack([
            torch.tensor(initial_x_pos, dtype=torch.float64),
            py_init_tensor_for_grad,  # This part requires grad
            torch.tensor(initial_x_vel, dtype=torch.float64),
            torch.tensor(initial_y_vel, dtype=torch.float64)
        ])

        # Time the forward pass (which builds the computation graph)
        t_start_fwd_pass = time.perf_counter()
        yt_trajectory_fwd = icpg_integrator(time_steps_ts, y0_with_grad_setup,
                                                      integration_params=simulation_params,
                                                      restitution_coeff=restitution_val)
        # Compute a scalar loss based on the forward pass output
        final_y_height = yt_trajectory_fwd[-1, 1]
        loss_val = (final_y_height - dummy_target_height) ** 2
        t_end_fwd_pass = time.perf_counter()
        time_taken_fwd_pass = t_end_fwd_pass - t_start_fwd_pass
        recorded_times_fwd_pass_graph_enabled.append(time_taken_fwd_pass)
        print(f"    Time (fwd_pass_graph_enabled): {time_taken_fwd_pass:.6f} s")

        # Time the backward pass (gradient computation)
        t_start_bwd_pass = time.perf_counter()
        loss_val.backward()  # Computes gradients, e.g., d(loss_val)/d(py_init_tensor_for_grad)
        # grad_of_py_init = py_init_tensor_for_grad.grad # Can access gradient if needed
        t_end_bwd_pass = time.perf_counter()
        time_taken_bwd_pass = t_end_bwd_pass - t_start_bwd_pass
        recorded_times_bwd_pass_gradient_calc.append(time_taken_bwd_pass)
        print(f"    Time (bwd_pass_gradient_calc): {time_taken_bwd_pass:.6f} s")

        recorded_times_total_with_grad.append(time_taken_fwd_pass + time_taken_bwd_pass)

    # --- Plotting Results ---
    plot_data_dict = {
        'Simulation Performance': {  # This key acts as a subplot identifier/title component
            'No Gradient (Simulation Only)': {
                'data': np.array(recorded_times_no_grad),
                'label': 'Time (torch.no_grad)',
                'color': '#1F77B4', # Blue
                'marker': 'o',
                'linestyle': '-',
                'linewidth': 2.0
            },
            'Forward Pass (Graph Enabled)': {
                'data': np.array(recorded_times_fwd_pass_graph_enabled),
                'label': 'Time (Forward Pass with Graph)',
                'color': '#2CA02C', # Green
                'marker': 's',
                'linestyle': '-',
                'linewidth': 2.0
            },
            'Backward Pass (Gradient Calc)': {
                'data': np.array(recorded_times_bwd_pass_gradient_calc),
                'label': 'Time (Backward Pass - grad())',
                'color': '#FF7F0E', # Orange
                'marker': 'x',
                'linestyle': '--',
                'linewidth': 2.0
            },
            'Total (Forward Graph + Backward)': {
                'data': np.array(recorded_times_total_with_grad),
                'label': 'Time (Total: Fwd Graph + Bwd)',
                'color': '#6B42F0', # Purple
                'marker': '^',
                'linestyle': '--',
                'linewidth': 2.0
            }
        }
    }

    plot_config = {
        'figsize': (8, 5),
        'suptitle': '',
        # 'suptitle': 'Backward Euler Simulation: Computation Time Analysis',
        'x_label': 'Number of Simulation Steps',
        'y_labels': {'Simulation Performance': 'Execution Time (s)'},
        'plot_titles': {'Simulation Performance': ''},
        # 'plot_titles': {'Simulation Performance': 'Timing vs. Number of Steps'},
        'legend': {'show': True, 'location': 'upper left', 'fontsize': 10},
        'grid': {'show': True, 'linestyle': ':', 'alpha': 0.7},
        'font_sizes': {'title': 14, 'suptitle': 16, 'axis_label': 12, 'tick_label': 12},
        'tight_layout': True,
        # 'log_y_axis': True, # Enable if times vary over orders of magnitude
    }

    # Using create_scientific_subplot_plot
    # time_data is the common x-axis values
    x_axis_values = np.array(num_steps_options)

    fig, axes = create_scientific_subplot_plot(
        time_data=x_axis_values,
        plot_data=plot_data_dict,
        config=plot_config,
        save_path="simulation_timing_performance.png"
    )
    print("\nExperiment complete. Plot saved to simulation_timing_performance.png")
    plt.show() # Uncomment to display plot interactively if environment supports it


if __name__ == "__main__":
    run_timing_experiment()