import time
import torch
import numpy as np
import matplotlib.pyplot as plt  # Typically imported by utils but good to have if direct plt use is needed
from xitorch.optimize import rootfinder

# Assuming utils.py is in the same directory or Python path
from demos.utils import create_scientific_subplot_plot


def fb(a: torch.Tensor, b: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Fischer-Burmeister function for complementarity."""
    return a + b - torch.sqrt(a ** 2 + b ** 2 + epsilon)


def custom_newton_solver(fcn, x0, params, **options):
    """
    Custom Newton solver.
    'options' will receive 'tol', 'maxiter' etc. from xitorch.rootfinder.
    """
    tol = options.get("tol", 1e-7)
    maxiter = options.get("maxiter", 10)
    x = x0.clone()  # Work on a clone

    for i in range(maxiter):
        fx = fcn(x, *params)

        # Compute Jacobian
        # Requires x to be a tensor that allows jacobian computation.
        # strict=False allows non-tensor outputs from fcn if any, but fcn should return tensor for rootfinding.
        # create_graph=False is default, meaning Jacobian computation itself is not part of AD graph if not needed.
        J_val = torch.autograd.functional.jacobian(lambda y_jac: fcn(y_jac, *params), x, strict=False)

        # jacobian might return a tuple if fcn has multiple outputs, take the first.
        # For a typical rootfinder fcn, output is a single tensor.
        if isinstance(J_val, tuple):
            J = J_val[0]
        else:
            J = J_val

        # Ensure J is 2D if fx is 1D
        if J.ndim == 1 and fx.ndim == 1 and J.shape[0] == fx.shape[
            0]:  # If J is (N,) and fx is (N,) -> J should be (N,N)
            # This case implies fcn output dim matches input dim, but jacobian squeezed it for some reason.
            # Or, if x is scalar, J will be scalar.
            if x.numel() == 1 and J.numel() == 1:  # scalar case
                pass  # J is fine as scalar
            # This needs careful handling based on expected structure of J for fcn(x)
            # Assuming fcn output vector dim = x vector dim, J should be square.

        if x.numel() == 1 and J.numel() == 1:  # Scalar case
            if J.abs() < 1e-9:  # Avoid division by zero
                delta = -fx * 1e9  # Large step if derivative is zero
            else:
                delta = -fx / J
        else:  # Vector case
            try:
                # J should be (N, N) and fx (N,)
                if J.shape[0] != J.shape[1] or J.shape[1] != fx.shape[0]:
                    # print(f"Warning: Jacobian shape {J.shape} or fx shape {fx.shape} not standard for square system. Using lstsq.")
                    delta = torch.linalg.lstsq(J, -fx).solution
                else:
                    delta = torch.linalg.solve(J, -fx)
            except torch.linalg.LinAlgError:
                # print(f"Warning: Singular Jacobian {J}. Using pseudo-inverse (lstsq).")
                delta = torch.linalg.lstsq(J, -fx).solution

        x = x + delta
        if torch.norm(delta) < tol:
            break
    # if i == maxiter -1:
    #     print(f"Warning: Newton solver reached max iterations ({maxiter}). Norm(delta): {torch.norm(delta)}")
    return x


def backward_euler_integrator(ts, y0_in, integration_params, restitution_coeff=0.0):
    """
    Backward Euler integrator for bouncing ball dynamics.
    'integration_params' is a list, expected to contain [gravity_tensor].
    'fcn' (the explicit dynamics function) is not used here as equations are in residual.
    """
    g = integration_params[0]
    e = restitution_coeff
    n_steps = len(ts) - 1

    yt_list = [y0_in]

    for i in range(n_steps):
        t_curr, y_curr = ts[i], yt_list[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr

        # Residual function for the implicit step with contact
        def residual_function(z_next_vars, y_curr_res, dt_res, g_res, collision_active):
            vx_next, vy_next, lambda_next = z_next_vars  # lambda_next is contact impulse term (scaled)

            # Equations of motion (implicit form)
            # dvx/dt = 0  => (vx_next - y_curr_res[2])/dt = 0
            res1 = vx_next - y_curr_res[2]
            # dvy/dt = -g + lambda_contact/dt => (vy_next - y_curr_res[3])/dt = -g_res + lambda_next/dt (if lambda_next is impulse)
            # Or, if lambda_next is an acceleration term: (vy_next - y_curr_res[3])/dt = -g_res + lambda_next
            # Original: res2 = vy_next - y_curr_res[3] + dt * g - lambda_next
            # This means lambda_next is an acceleration-like term from contact.
            res2 = vy_next - y_curr_res[3] + dt_res * g_res - lambda_next

            if collision_active:
                # y_curr_res[1] is current y position. y_curr_res[3] is current y velocity.
                # b_err is a penetration correction term based on current penetration.
                # Using .detach() on y_curr_res[1] for b_err if its gradient contribution is not desired (numerical stabilization).
                _y_curr_1_detached = y_curr_res[1].detach()
                b_err = (0.01 / dt_res) * _y_curr_1_detached

                # b_rest incorporates restitution based on pre-impact velocity (y_curr_res[3])
                b_rest = e * y_curr_res[3]

                # Complementarity condition: lambda_next >= 0, (vy_next + penetration_corr + restitution_eff_vy) >= 0
                res3 = fb(lambda_next, vy_next + b_err + b_rest)
            else:
                res3 = -lambda_next  # No contact force/impulse term
            return torch.stack([res1, res2, res3])

        # Initial guess for z_next = [vx_next, vy_next, lambda_next]
        # Forward Euler step for velocity guess, lambda guess = 0
        vx_guess = y_curr[2]
        vy_guess = y_curr[3] - dt * g
        z_guess = torch.tensor([vx_guess, vy_guess, 0.0], dtype=y0_in.dtype, device=y0_in.device)

        # Collision detection based on current state at the beginning of the step
        # (More advanced methods might predict collision within the step)
        is_collision = False
        if y_curr[1] < 0.0:  # If already penetrating or on the ground
            is_collision = True

        # Solve for z_next using rootfinder
        # The custom_newton_solver will be used by rootfinder.
        # tol and maxiter are passed from rootfinder to custom_newton_solver's **options.
        z_next_solution = rootfinder(
            residual_function,
            z_guess,
            params=(y_curr, dt, g, is_collision),  # These are passed as *args to residual_function after z_next_vars
            method=custom_newton_solver,
            tol=1e-9,  # Tolerance for rootfinder, passed to method's options
            maxiter=20  # Max iterations for rootfinder, passed to method's options
        )

        # Extract solved state variables
        vx_next_sol, vy_next_sol = z_next_solution[:2]

        # Update positions using solved next velocities
        x_next_pos = y_curr[0] + dt * vx_next_sol
        y_next_pos = y_curr[1] + dt * vy_next_sol

        yt_list.append(torch.stack([x_next_pos, y_next_pos, vx_next_sol, vy_next_sol]))

    return torch.stack(yt_list, dim=0)


def run_timing_experiment():
    """Runs the timing experiment and plots the results."""

    # Experiment Configuration
    sim_time_total = 1.0  # Total simulation time
    num_steps_options = [10, 50, 100, 200, 300, 400, 800]

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
            _ = backward_euler_integrator(time_steps_ts, y0_no_grad_setup,
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
        yt_trajectory_fwd = backward_euler_integrator(time_steps_ts, y0_with_grad_setup,
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

import timeit

def run_timing_experiment():
    """Runs the timing experiment and plots the results."""

    # Experiment Configuration
    sim_time_total = 1.0  # Total simulation time
    num_steps_options = [10, 50, 100, 200, 300, 400]

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

    # Number of repetitions for timeit
    num_repeats = 20

    print("Starting timing experiment...")
    for steps_count in num_steps_options:
        print(f"  Processing for N_steps = {steps_count}")
        current_dt = sim_time_total / steps_count
        time_steps_ts = torch.linspace(0, sim_time_total, steps_count + 1, dtype=torch.float64)

        # --- Measurement 1: Simulation without Gradient Tracking ---
        setup_no_grad = f"""
import torch
from demos.utils import create_scientific_subplot_plot
from xitorch.optimize import rootfinder
from __main__ import backward_euler_integrator, custom_newton_solver, fb
y0_no_grad_setup = torch.tensor([{initial_x_pos}, {initial_y_pos_val}, {initial_x_vel}, {initial_y_vel}],
                                dtype=torch.float64, requires_grad=False)
time_steps_ts = torch.linspace(0, {sim_time_total}, {steps_count + 1}, dtype=torch.float64)
gravity_g = torch.tensor(9.81, dtype=torch.float64)
simulation_params = [gravity_g]
restitution_val = {restitution_val}
        """
        code_no_grad = """
with torch.no_grad():
    _ = backward_euler_integrator(time_steps_ts, y0_no_grad_setup,
                                  integration_params=simulation_params,
                                  restitution_coeff=restitution_val)
        """
        time_taken_no_grad = timeit.timeit(code_no_grad, setup=setup_no_grad, number=num_repeats) / num_repeats
        recorded_times_no_grad.append(time_taken_no_grad)
        print(f"    Time (no_grad, avg of {num_repeats} runs): {time_taken_no_grad:.6f} s")

        # --- Measurement 2: Simulation with Gradient Tracking (Forward Pass) ---
        setup_fwd_pass = f"""
import torch
from demos.utils import create_scientific_subplot_plot
from xitorch.optimize import rootfinder
from __main__ import backward_euler_integrator, custom_newton_solver, fb
py_init_tensor_for_grad = torch.tensor({initial_y_pos_val}, dtype=torch.float64, requires_grad=True)
y0_with_grad_setup = torch.stack([
    torch.tensor({initial_x_pos}, dtype=torch.float64),
    py_init_tensor_for_grad,
    torch.tensor({initial_x_vel}, dtype=torch.float64),
    torch.tensor({initial_y_vel}, dtype=torch.float64)
])
time_steps_ts = torch.linspace(0, {sim_time_total}, {steps_count + 1}, dtype=torch.float64)
gravity_g = torch.tensor(9.81, dtype=torch.float64)
simulation_params = [gravity_g]
restitution_val = {restitution_val}
dummy_target_height = {dummy_target_height}
        """
        code_fwd_pass = """
yt_trajectory_fwd = backward_euler_integrator(time_steps_ts, y0_with_grad_setup,
                                              integration_params=simulation_params,
                                              restitution_coeff=restitution_val)
final_y_height = yt_trajectory_fwd[-1, 1]
loss_val = (final_y_height - dummy_target_height) ** 2
        """
        time_taken_fwd_pass = timeit.timeit(code_fwd_pass, setup=setup_fwd_pass, number=num_repeats) / num_repeats
        recorded_times_fwd_pass_graph_enabled.append(time_taken_fwd_pass)
        print(f"    Time (fwd_pass_graph_enabled, avg of {num_repeats} runs): {time_taken_fwd_pass:.6f} s")

        # --- Measurement 3: Backward Pass (Gradient Computation) ---
        # Time both forward and backward passes, then subtract forward time to estimate backward time
        code_fwd_bwd_pass = """
yt_trajectory_fwd = backward_euler_integrator(time_steps_ts, y0_with_grad_setup,
                                              integration_params=simulation_params,
                                              restitution_coeff=restitution_val)
final_y_height = yt_trajectory_fwd[-1, 1]
loss_val = (final_y_height - dummy_target_height) ** 2
loss_val.backward()
        """
        time_taken_fwd_bwd = timeit.timeit(code_fwd_bwd_pass, setup=setup_fwd_pass, number=num_repeats) / num_repeats
        # Estimate backward time by subtracting forward time
        time_taken_bwd_pass = time_taken_fwd_bwd - time_taken_fwd_pass
        recorded_times_bwd_pass_gradient_calc.append(time_taken_bwd_pass)
        print(f"    Time (bwd_pass_gradient_calc, avg of {num_repeats} runs): {time_taken_bwd_pass:.6f} s")

        recorded_times_total_with_grad.append(time_taken_fwd_pass + time_taken_bwd_pass)

    # --- Plotting Results ---
    plot_data_dict = {
        'Simulation Performance': {
            'No Gradient (Simulation Only)': {
                'data': np.array(recorded_times_no_grad),
                'label': 'Time (torch.no_grad)',
                'color': '#1F77B4',
                'marker': 'o',
                'linestyle': '-',
                'linewidth': 2.0
            },
            'Forward Pass (Graph Enabled)': {
                'data': np.array(recorded_times_fwd_pass_graph_enabled),
                'label': 'Time (Forward Pass with Graph)',
                'color': '#2CA02C',
                'marker': 's',
                'linestyle': '-',
                'linewidth': 2.0
            },
            'Backward Pass (Gradient Calc)': {
                'data': np.array(recorded_times_bwd_pass_gradient_calc),
                'label': 'Time (Backward Pass - grad())',
                'color': '#FF7F0E',
                'marker': 'x',
                'linestyle': '--',
                'linewidth': 2.0
            },
            'Total (Forward Graph + Backward)': {
                'data': np.array(recorded_times_total_with_grad),
                'label': 'Time (Total: Fwd Graph + Bwd)',
                'color': '#6B42F0',
                'marker': '^',
                'linestyle': '--',
                'linewidth': 2.0
            }
        }
    }

    plot_config = {
        'figsize': (8, 5),
        'suptitle': '',
        'x_label': 'Number of Simulation Steps',
        'y_labels': {'Simulation Performance': 'Execution Time (s)'},
        'plot_titles': {'Simulation Performance': ''},
        'legend': {'show': True, 'location': 'upper left', 'fontsize': 10},
        'grid': {'show': True, 'linestyle': ':', 'alpha': 0.7},
        'font_sizes': {'title': 14, 'suptitle': 16, 'axis_label': 12, 'tick_label': 12},
        'tight_layout': True,
    }

    x_axis_values = np.array(num_steps_options)
    fig, axes = create_scientific_subplot_plot(
        time_data=x_axis_values,
        plot_data=plot_data_dict,
        config=plot_config,
        save_path="simulation_timing_performance.png"
    )
    print("\nExperiment complete. Plot saved to simulation_timing_performance.png")
    plt.show()

if __name__ == "__main__":
    run_timing_experiment()