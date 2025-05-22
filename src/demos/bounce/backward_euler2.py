import time
import torch
import numpy as np
import matplotlib.pyplot as plt  # Kept for potential direct use
from xitorch.optimize import rootfinder


def fb(a: torch.Tensor, b: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Standard Fischer-Burmeister function for complementarity."""
    return a + b - torch.sqrt(a ** 2 + b ** 2 + epsilon)


def custom_newton_solver(fcn, x0, params, **options):
    """
    Custom Newton solver with improved stability for Jacobian inversion.
    'options' receives 'tol', 'maxiter', etc., from xitorch.rootfinder.
    """
    tol = options.get("tol", 1e-7)
    maxiter = options.get("maxiter", 20)
    x = x0.clone()

    for i in range(maxiter):
        fx = fcn(x, *params)
        J_val = torch.autograd.functional.jacobian(lambda y_jac: fcn(y_jac, *params), x, strict=False)

        if isinstance(J_val, tuple):
            J = J_val[0]
        else:
            J = J_val

        if x.numel() == 1 and J.numel() == 1:  # Scalar case
            if J.abs() < 1e-9:
                delta = -fx * 1e3
            else:
                delta = -fx / J
        else:  # Vector case
            try:
                if J.ndim == 1 and fx.ndim == 1 and J.shape[0] == fx.shape[0] and x.ndim == 1 and x.shape[0] == J.shape[
                    0]:
                    J = J.diag()
                elif J.shape == fx.shape and J.ndim == 1:
                    J = torch.diag(J)

                if J.shape[0] != J.shape[1] or J.shape[1] != fx.shape[0]:
                    delta = torch.linalg.lstsq(J, -fx).solution
                else:
                    delta = torch.linalg.solve(J, -fx)
            except torch.linalg.LinAlgError:
                delta = torch.linalg.lstsq(J, -fx).solution

        x = x + delta
        if torch.norm(delta) < tol:
            break
    return x


def backward_euler_contact_step(y_curr, dt, g_param, restitution_coeff_float, collision_active):
    """
    Perform a single backward Euler step with contact handling.
    restitution_coeff_float is a Python float.
    Returns: y_next_state, lambda_contact_sol
    """
    e = restitution_coeff_float  # Is a float

    def residual_contact(z_next_vars, y_c_solver, dt_r_solver, g_r_solver, collision_flg_solver: bool):
        vx_next, vy_next, lambda_next = z_next_vars
        res1 = vx_next - y_c_solver[2]
        res2 = vy_next - y_c_solver[3] + dt_r_solver * g_r_solver - lambda_next

        if collision_flg_solver:
            b_err = (0.01 / dt_r_solver) * y_c_solver[1]
            b_rest = e * y_c_solver[3]  # e is float
            res3 = fb(lambda_next, vy_next + b_err + b_rest)
        else:
            res3 = lambda_next
        return torch.stack([res1, res2, res3])

    vx_guess = y_curr[2]
    vy_guess = y_curr[3] - dt * g_param
    z_guess = torch.tensor([vx_guess, vy_guess, 0.0], dtype=y_curr.dtype, device=y_curr.device)

    y_curr_for_solver = y_curr.detach()  # Always detach for BE part of ICPG

    z_next_solution = rootfinder(
        residual_contact,
        z_guess,
        method=custom_newton_solver,
        params=(y_curr_for_solver, dt, g_param, collision_active),  # collision_active is crucial
        tol=1e-7,
        maxiter=30
    )

    vx_next_sol, vy_next_sol, lambda_contact_sol = z_next_solution[0], z_next_solution[1], z_next_solution[2]

    if not collision_active:
        lambda_contact_sol = torch.zeros_like(lambda_contact_sol)

    x_next_pos = y_curr[0] + dt * vx_next_sol
    y_next_pos = y_curr[1] + dt * vy_next_sol

    y_next_state = torch.stack([x_next_pos, y_next_pos, vx_next_sol, vy_next_sol])
    return y_next_state, lambda_contact_sol





def fixed_step_backward_euler_integrator(ts_fixed, y0_in_fixed, integration_params_fixed, restitution_coeff_fixed=0.0,
                                         collision_threshold_fixed=0.0):
    g_fixed_orig = integration_params_fixed[0]
    n_steps_fixed = len(ts_fixed) - 1

    yt_list_fixed = [y0_in_fixed.clone()]
    y_curr_fixed = y0_in_fixed.clone()

    device = y_curr_fixed.device
    dtype = y_curr_fixed.dtype

    if not isinstance(g_fixed_orig, torch.Tensor):
        g_fixed = torch.tensor(g_fixed_orig, device=device, dtype=dtype)
    elif g_fixed_orig.device != device or g_fixed_orig.dtype != dtype:
        g_fixed = g_fixed_orig.to(device=device, dtype=dtype)
    else:
        g_fixed = g_fixed_orig

    for i in range(n_steps_fixed):
        t_curr_fixed, t_next_fixed = ts_fixed[i], ts_fixed[i + 1]
        dt_val = t_next_fixed - t_curr_fixed

        if not isinstance(dt_val, torch.Tensor):
            dt_fixed = torch.tensor(dt_val, device=device, dtype=dtype)
        elif dt_val.device != device or dt_val.dtype != dtype:
            dt_fixed = dt_val.to(device=device, dtype=dtype)
        else:
            dt_fixed = dt_val

        is_collision_active = bool(y_curr_fixed[1].item() < collision_threshold_fixed)

        y_next_state, _ = backward_euler_contact_step(
            y_curr_fixed,
            dt_fixed,
            g_fixed,
            restitution_coeff_fixed,  # Pass as float
            is_collision_active
        )
        yt_list_fixed.append(y_next_state)
        y_curr_fixed = y_next_state

    return torch.stack(yt_list_fixed, dim=0)


if __name__ == '__main__':
    g_gravity_val = 0.0
    g_gravity = torch.tensor(g_gravity_val, dtype=torch.float64)

    sim_params = [g_gravity]
    initial_y_pos = 2.0
    y_initial_state = torch.tensor([0.0, initial_y_pos, 5.0, -5.0], dtype=torch.float64)
    restitution = 1.0

    sim_time = 1.0
    dt_val = 0.005
    num_steps = int(sim_time / dt_val)
    ts = torch.linspace(0, sim_time, num_steps + 1, dtype=torch.float64)

    print("--- Testing ICPG Integrator ---")
    y_initial_icpg = y_initial_state.clone().requires_grad_(True)

    collision_thresh_icpg = 0.0
    k_eff_eps_icpg = 1e-5
    min_k_icpg = 0.0
    max_k_icpg = 1e7

    start_time_icpg = time.time()
    states_icpg = icpg_integrator(
        ts=ts,
        y0=y_initial_icpg,
        integration_params=sim_params,
        restitution_coeff=restitution,
        min_k_eff=min_k_icpg,
        max_k_eff=max_k_icpg
    )
    end_time_icpg = time.time()
    print(f"ICPG Integrator: Ran for {len(states_icpg) - 1} steps in {end_time_icpg - start_time_icpg:.4f}s.")

    if states_icpg.requires_grad:
        final_height_icpg = states_icpg[-1, 1]
        loss_icpg = final_height_icpg

        loss_icpg.backward()
        print(f"Loss (final height): {final_height_icpg.item()}")
        if y_initial_icpg.grad is not None:
            print(f"Gradient of loss w.r.t y_initial_state[1] (ICPG): {y_initial_icpg.grad[1].item()}")
        else:
            print("No gradient computed for y_initial_icpg.")

    print("\n--- Testing Fixed-Step Backward Euler (for comparison) ---")
    y_initial_fixed = y_initial_state.clone().requires_grad_(True)
    if y_initial_fixed.grad is not None: y_initial_fixed.grad.zero_()

    start_time_be = time.time()
    states_fixed = fixed_step_backward_euler_integrator(
        ts_fixed=ts,
        y0_in_fixed=y_initial_fixed,
        integration_params_fixed=sim_params,
        restitution_coeff_fixed=restitution,
        collision_threshold_fixed=collision_thresh_icpg
    )
    end_time_be = time.time()
    print(f"Fixed-Step BE: Ran for {len(states_fixed) - 1} steps in {end_time_be - start_time_be:.4f}s.")

    if states_fixed.requires_grad:
        final_height_fixed = states_fixed[-1, 1]
        loss_fixed = final_height_fixed

        loss_fixed.backward()
        print(f"Loss (final height): {final_height_fixed.item()}")
        if y_initial_fixed.grad is not None:
            print(f"Gradient of loss w.r.t y_initial_state[1] (Fixed BE): {y_initial_fixed.grad[1].item()}")
        else:
            print("No gradient computed for y_initial_fixed.")

    plt.figure(figsize=(10, 5))
    plt.plot(states_icpg[:, 0].detach().numpy(), states_icpg[:, 1].detach().numpy(), 'b-o', markersize=2,
             label=f'ICPG (Final y: {states_icpg[-1, 1].item():.3f})')
    plt.plot(states_fixed[:, 0].detach().numpy(), states_fixed[:, 1].detach().numpy(), 'r-s', markersize=2,
             label=f'Fixed BE (Final y: {states_fixed[-1, 1].item():.3f})')

    plt.scatter(states_icpg[0, 0].detach().numpy(), states_icpg[0, 1].detach().numpy(), c='blue', marker='X', s=100,
                label='Start ICPG')
    plt.scatter(states_fixed[0, 0].detach().numpy(), states_fixed[0, 1].detach().numpy(), c='red', marker='P', s=100,
                label='Start BE')

    plt.title('Trajectory Comparison: ICPG vs Fixed BE')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axhline(0, color='k', linestyle='--', linewidth=0.8, label='Ground')
    plt.legend()
    plt.grid(True)
    # plt.axis('equal') # Can make trajectories very flat if x >> y
    plt.gca().set_aspect('equal', adjustable='box')  # Better for aspect ratio
    plt.savefig("icpg_vs_be_trajectory_comparison.png")
    print("\nSaved example trajectories plot to 'icpg_vs_be_trajectory_comparison.png'")
    # plt.show()