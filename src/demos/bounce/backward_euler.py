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


def backward_euler_contact_step(y_curr, dt, g_param, restitution_coeff, collision_active):
    """
    Perform a single backward Euler step with contact handling.
    This is the core reusable step logic.
    'g_param' is a scalar tensor for gravity.
    """
    e = restitution_coeff

    def residual_contact(z_next_vars, y_c, dt_r, g_r, collision_flg: bool):
        vx_next, vy_next, lambda_next = z_next_vars
        res1 = vx_next - y_c[2]
        res2 = vy_next - y_c[3] + dt_r * g_r - lambda_next

        if collision_flg:
            # b_err = (0.01 / dt_r) * y_c[1].detach()
            b_rest = e * y_c[3]
            res3 = fb(lambda_next, vy_next + b_rest)
        else:
            res3 = -lambda_next
        return torch.stack([res1, res2, res3])

    vx_guess = y_curr[2]
    vy_guess = y_curr[3] - dt * g_param
    z_guess = torch.tensor([vx_guess, vy_guess, 0.0], dtype=y_curr.dtype, device=y_curr.device)

    z_next_solution = rootfinder(
        residual_contact,
        z_guess,
        method=custom_newton_solver,
        params=(y_curr, dt, g_param, collision_active),
        tol=1e-9,
        maxiter=20
    )

    vx_next_sol, vy_next_sol = z_next_solution[:2]
    lambda_contact_sol = z_next_solution[2]
    x_next_pos = y_curr[0] + dt * vx_next_sol
    y_next_pos = y_curr[1] + dt * vy_next_sol

    y_next_state = torch.stack([x_next_pos, y_next_pos, vx_next_sol, vy_next_sol])
    return y_next_state, lambda_contact_sol

def fixed_step_backward_euler_integrator(ts, y0, integration_params, restitution_coeff=0.0):
    """
    Fixed-step Backward Euler integrator.
    Uses 'backward_euler_contact_step' for each step.
    'integration_params_fixed' is a list, expected to contain [gravity_tensor].
    """
    g_fixed = integration_params[0]
    n_steps_fixed = len(ts) - 1

    yt_list_fixed = [y0.clone()]

    for i in range(n_steps_fixed):
        t_curr_fixed, y_curr_fixed = ts[i], yt_list_fixed[i]
        t_next_fixed = ts[i + 1]
        dt_fixed = t_next_fixed - t_curr_fixed

        is_collision_active = bool(y_curr_fixed[1].item() < 0.0)

        # Use the unified single-step function
        y_next_state, _ = backward_euler_contact_step(
            y_curr_fixed,
            dt_fixed,
            g_fixed,
            restitution_coeff,
            is_collision_active
        )

        yt_list_fixed.append(y_next_state)

    return torch.stack(yt_list_fixed, dim=0)


def forward_euler_dyn(t: torch.Tensor, y: torch.Tensor, g_param: torch.Tensor) -> torch.Tensor:
    """ Explicit Euler step content for free-fall dynamics. """
    dx = y[2]
    dy = y[3]
    dvx = torch.tensor(0.0, dtype=y.dtype, device=y.device)
    dvy = -g_param
    return torch.stack([dx, dy, dvx, dvy], dim=0)


def free_fall_substeps_predict(t_start_sub, y_start_sub, dt_total_sub, num_substeps, g_param):
    """ Computes free-fall states at substep times using explicit Euler for prediction. """
    with torch.no_grad():
        sub_times_val = torch.linspace(t_start_sub, t_start_sub + dt_total_sub, num_substeps,
                                       dtype=y_start_sub.dtype, device=y_start_sub.device)
        sub_states_val = torch.zeros((num_substeps, 4), dtype=y_start_sub.dtype, device=y_start_sub.device)

        if num_substeps == 0: return sub_times_val, sub_states_val
        sub_states_val[0] = y_start_sub
        if num_substeps == 1: return sub_times_val, sub_states_val

        sub_dt_val = dt_total_sub / (num_substeps - 1)
        for i in range(1, num_substeps):
            y_prev_sub = sub_states_val[i - 1]
            dydt_sub = forward_euler_dyn(sub_times_val[i - 1], y_prev_sub, g_param)
            y_next_sub = y_prev_sub + sub_dt_val * dydt_sub
            sub_states_val[i] = y_next_sub
    return sub_times_val, sub_states_val


def adaptive_backward_euler_integrator(t_span_val, y0, integration_params, restitution_coeff=0.0,
                                       dt_initial=0.01, num_predict_substeps=10, penetration_threshold=-0.001):
    """ Adaptive time-stepping backward Euler integrator. """
    g_param = integration_params[0]
    t_current_sim, t_end_sim = t_span_val
    y_current_sim = y0.clone()
    dt_current_step = dt_initial

    times_history = [t_current_sim.clone() if isinstance(t_current_sim, torch.Tensor) else torch.tensor(t_current_sim,
                                                                                                        dtype=y0.dtype,
                                                                                                        device=y0.device)]
    states_history = [y_current_sim.clone()]

    max_sim_steps = int((t_end_sim - t_current_sim).item() / (dt_initial * 0.1)) if isinstance(t_end_sim,
                                                                                               torch.Tensor) else int(
        (t_end_sim - t_current_sim) / (dt_initial * 0.1))  # Safety break
    step_count = 0

    while t_current_sim < t_end_sim and step_count < max_sim_steps:
        step_count += 1
        # Ensure actual_dt_this_segment is a scalar tensor or float
        _t_end_sim = t_end_sim.item() if isinstance(t_end_sim, torch.Tensor) else t_end_sim
        _t_current_sim = t_current_sim.item() if isinstance(t_current_sim, torch.Tensor) else t_current_sim
        actual_dt_this_segment = min(dt_current_step, _t_end_sim - _t_current_sim)

        if actual_dt_this_segment <= 1e-7: break

        with torch.no_grad():
            sub_times, sub_states = free_fall_substeps_predict(
                t_current_sim, y_current_sim, actual_dt_this_segment,
                num_predict_substeps, g_param
            )
            y_positions_sub = sub_states[:, 1]
            is_penetrating_sub = y_positions_sub < penetration_threshold
            first_penetration_idx = None
            if is_penetrating_sub.any():
                first_penetration_idx = torch.where(is_penetrating_sub)[0][0].item()

        if first_penetration_idx is not None and first_penetration_idx > 0:
            dt_to_pre_collision = sub_times[first_penetration_idx - 1] - t_current_sim

            if dt_to_pre_collision > 1e-7:
                dydt_free_fall = forward_euler_dyn(t_current_sim, y_current_sim, g_param)
                y_current_sim = y_current_sim + dt_to_pre_collision * dydt_free_fall
                t_current_sim = t_current_sim + dt_to_pre_collision
                states_history.append(y_current_sim.clone())
                times_history.append(
                    t_current_sim.clone() if isinstance(t_current_sim, torch.Tensor) else torch.tensor(t_current_sim,
                                                                                                       dtype=y0.dtype,
                                                                                                       device=y0.device))

            if t_current_sim >= t_end_sim: break

            dt_implicit_collision_step = sub_times[first_penetration_idx] - t_current_sim
            _t_end_sim = t_end_sim.item() if isinstance(t_end_sim, torch.Tensor) else t_end_sim
            _t_current_sim = t_current_sim.item() if isinstance(t_current_sim, torch.Tensor) else t_current_sim
            dt_implicit_collision_step = min(dt_implicit_collision_step.item(), _t_end_sim - _t_current_sim)

            if dt_implicit_collision_step > 1e-7:
                y_next_contact, _ = backward_euler_contact_step(  # Using the core step function
                    y_current_sim,
                    torch.tensor(dt_implicit_collision_step, dtype=y_current_sim.dtype, device=y_current_sim.device),
                    g_param, restitution_coeff, collision_active=True
                )
                y_current_sim = y_next_contact
                t_current_sim = t_current_sim + torch.tensor(dt_implicit_collision_step, dtype=y_current_sim.dtype,
                                                             device=y_current_sim.device)
                states_history.append(y_current_sim.clone())
                times_history.append(
                    t_current_sim.clone() if isinstance(t_current_sim, torch.Tensor) else torch.tensor(t_current_sim,
                                                                                                       dtype=y0.dtype,
                                                                                                       device=y0.device))

            dt_current_step = dt_initial
        else:
            dydt_free_fall = forward_euler_dyn(t_current_sim, y_current_sim, g_param)
            y_current_sim = y_current_sim + torch.tensor(actual_dt_this_segment, dtype=y_current_sim.dtype,
                                                         device=y_current_sim.device) * dydt_free_fall
            t_current_sim = t_current_sim + torch.tensor(actual_dt_this_segment, dtype=y_current_sim.dtype,
                                                         device=y_current_sim.device)
            states_history.append(y_current_sim.clone())
            times_history.append(
                t_current_sim.clone() if isinstance(t_current_sim, torch.Tensor) else torch.tensor(t_current_sim,
                                                                                                   dtype=y0.dtype,
                                                                                                   device=y0.device))

    return (torch.stack(times_history) if isinstance(times_history[0], torch.Tensor) else torch.tensor(times_history,
                                                                                                       dtype=y0.dtype,
                                                                                                       device=y0.device),
            torch.stack(states_history))


def icpg_integrator(ts, y0, integration_params, restitution_coeff=0.0,
                    min_k_eff=0.0, max_k_eff=1e6, min_c_eff=0.0, max_c_eff=1e3,
                    w_k=1e-4, w_c=1.0):
    """
    Integrator with spring-damper contact model, computing weighted minimum-norm k_eff and c_eff.

    Args:
        ts: Time steps (tensor of shape [n_steps + 1])
        y0: Initial state [x, y, vx, vy] (tensor of shape [4])
        integration_params: List containing gravity g
        restitution_coeff: Coefficient of restitution (default: 0.0)
        min_k_eff, max_k_eff: Bounds for effective spring stiffness
        min_c_eff, max_c_eff: Bounds for effective damping coefficient
        w_k: Weight for stiffness in the norm (default: 1.0)
        w_c: Weight for damping in the norm (default: 1.0)

    Returns:
        torch.Tensor: Trajectory of states [n_steps + 1, 4]
    """
    g = integration_params[0]
    n_steps = len(ts) - 1

    yt_list = [y0.clone()]

    for i in range(n_steps):
        y_current = yt_list[i]
        t_curr, t_next = ts[i], ts[i + 1]
        dt = t_next - t_curr

        is_collision_active = bool(y_current[1].item() < 0.0)

        with torch.no_grad():
            _, lambda_n = backward_euler_contact_step(
                y_current,
                dt,
                g,
                restitution_coeff,
                is_collision_active
            )

            if is_collision_active:
                y_penetration = y_current[1]  # Penetration depth (negative)
                v_y = y_current[3]  # Vertical velocity

                # Compute weighted minimum-norm solution for k_eff and c_eff
                denom = y_penetration ** 2 * w_c + v_y ** 2 * w_k + 1e-6  # Add epsilon
                k_eff = -lambda_n * y_penetration * w_c / denom
                c_eff = -lambda_n * v_y * w_k / denom

                # Apply bounds
                k_eff = torch.clamp(k_eff, min=min_k_eff, max=max_k_eff)
                c_eff = torch.clamp(c_eff, min=min_c_eff, max=max_c_eff)
            else:
                k_eff = 0.0
                c_eff = 0.0

        # Symplectic Euler step
        vx_next = y_current[2]
        vy_next = y_current[3] - dt * g - (k_eff * y_current[1] + c_eff * y_current[3])
        x_next = y_current[0] + dt * y_current[2]
        y_next = y_current[1] + dt * y_current[3]

        y_next = torch.stack([x_next, y_next, vx_next, vy_next])

        yt_list.append(y_next)

    return torch.stack(yt_list, dim=0)


def adaptive_icpg_integrator(t_span, y0, integration_params, restitution_coeff=0.0,
                             base_dt=0.01, num_predict_substeps=10, penetration_threshold=-0.001,
                             min_k_eff=0.0, max_k_eff=1e6, min_c_eff=0.0, max_c_eff=1e3,
                             w_k=1e-4, w_c=1.0):
    """
    Adaptive ICPG integrator combining adaptive backward Euler with spring-damper contact model.

    Args:
        t_span: Tuple of (t_start, t_end) for simulation time span
        y0: Initial state [x, y, vx, vy] (tensor of shape [4])
        integration_params: List containing gravity g
        restitution_coeff: Coefficient of restitution (default: 0.0)
        base_dt: Initial time step (default: 0.01)
        num_predict_substeps: Number of substeps for collision prediction (default: 10)
        penetration_threshold: Threshold for detecting penetration (default: -0.001)
        min_k_eff, max_k_eff: Bounds for effective spring stiffness
        min_c_eff, max_c_eff: Bounds for effective damping coefficient
        w_k: Weight for stiffness in the norm (default: 1e-4)
        w_c: Weight for damping in the norm (default: 1.0)

    Returns:
        tuple: (times, states) where times is a tensor of time points and states is a tensor
               of shape [n_steps, 4] containing [x, y, vx, vy] at each time
    """
    g_accel = integration_params[0]
    _t, t_end = t_span
    _y = y0.clone()

    times_history = [_t]
    states_history = [_y.clone()]

    step_count = 0
    while _t < t_end:
        step_count += 1
        actual_dt_this_segment = min(base_dt, t_end - _t)

        if actual_dt_this_segment <= 1e-7:
            break

        with torch.no_grad():
            sub_times, sub_states = free_fall_substeps_predict(
                _t, _y, actual_dt_this_segment,
                num_predict_substeps, g_accel
            )
            is_penetrating_sub = sub_states[:, 1] < penetration_threshold
            first_penetration_idx = None
            if is_penetrating_sub.any():
                first_penetration_idx = torch.where(is_penetrating_sub)[0][0].item()

        if first_penetration_idx is not None and first_penetration_idx > 0:
            # Move to just before collision using forward Euler
            dt_pre_collision = sub_times[first_penetration_idx - 1] - _t

            if dt_pre_collision > 1e-7:
                dydt_free_fall = forward_euler_dyn(_t, _y, g_accel)

                _y = _y + dt_pre_collision * dydt_free_fall
                _t = _t + dt_pre_collision

                times_history.append(_t.clone())
                states_history.append(_y.clone())


            if _t >= t_end:
                break

            # Contact step
            dt_collision = min((sub_times[first_penetration_idx] - _t).item(), t_end - _t)

            if dt_collision > 1e-7:
                with torch.no_grad():
                    _, lambda_n = backward_euler_contact_step(
                        _y,
                        torch.tensor(dt_collision, dtype=_y.dtype, device=_y.device),
                        g_accel, restitution_coeff, collision_active=True
                    )

                    # Compute k_eff and c_eff using weighted minimum-norm solution
                    y_penetration = _y[1]  # Penetration depth (negative)
                    v_y = _y[3]  # Vertical velocity
                    denom = y_penetration ** 2 * w_c + v_y ** 2 * w_k + 1e-6  # Add epsilon
                    k_eff = -lambda_n * y_penetration * w_c / denom
                    c_eff = -lambda_n * v_y * w_k / denom

                    # Apply bounds
                    k_eff = torch.clamp(k_eff, min=min_k_eff, max=max_k_eff)
                    c_eff = torch.clamp(c_eff, min=min_c_eff, max=max_c_eff)

                # Symplectic Euler step for contact
                vx_next = _y[2]
                vy_next = _y[3] - dt_collision * g_accel - (k_eff * _y[1] + c_eff * _y[3])
                x_next = _y[0] + dt_collision * vx_next
                y_next = _y[1] + dt_collision * vy_next

                _y = torch.stack([x_next, y_next, vx_next, vy_next])
                _t = _t + dt_collision
                states_history.append(_y.clone())
                times_history.append(_t)

        else:
            # print(f"First penetration index: {first_penetration_idx}")
            # Free-fall step using forward Euler
            dydt_free_fall = forward_euler_dyn(_t, _y, g_accel)
            _y = _y + torch.tensor(actual_dt_this_segment, dtype=_y.dtype,
                                                         device=_y.device) * dydt_free_fall
            _t = _t + torch.tensor(actual_dt_this_segment, dtype=_y.dtype,
                                                         device=_y.device)
            states_history.append(_y.clone())
            times_history.append(_t)

    return torch.tensor(times_history, dtype=y0.dtype, device=y0.device), torch.stack(states_history)

# --- Example Usage (Optional) ---
if __name__ == '__main__':
    g_gravity = torch.tensor(9.81, dtype=torch.float64)
    sim_params = [g_gravity]
    initial_y_pos = 2.0
    y_initial_state = torch.tensor([0.0, initial_y_pos, 1.0, 0.0], dtype=torch.float64)
    restitution = 0.2
    simulation_time_span = (0.0, 2.0)

    print("Testing Adaptive Integrator...")
    t_span_adaptive = (torch.tensor(simulation_time_span[0], dtype=torch.float64),
                       torch.tensor(simulation_time_span[1], dtype=torch.float64))
    y_initial_adaptive = y_initial_state.clone().requires_grad_(True)
    times_adaptive, states_adaptive = adaptive_backward_euler_integrator(
        t_span_val=t_span_adaptive,
        y0=y_initial_adaptive,
        integration_params=sim_params,
        restitution_coeff=restitution,
        dt_initial=0.05
    )
    print(f"Adaptive Integrator: Ran for {len(times_adaptive)} steps.")
    if states_adaptive.requires_grad:
        loss_adaptive = torch.sum(states_adaptive[-1, :2] ** 2)
        loss_adaptive.backward()
        print(f"Gradient of loss w.r.t y_initial_state (adaptive): {y_initial_adaptive.grad}")

    print("\nTesting Fixed-Step Integrator...")
    num_fixed_steps = 300
    ts_fixed_val = torch.linspace(simulation_time_span[0], simulation_time_span[1], num_fixed_steps + 1,
                                  dtype=torch.float64)
    y_initial_fixed = y_initial_state.clone().requires_grad_(True)
    if y_initial_fixed.grad is not None: y_initial_fixed.grad.zero_()

    states_fixed = fixed_step_backward_euler_integrator(
        ts=ts_fixed_val,
        y0=y_initial_fixed,
        integration_params=sim_params,
        restitution_coeff=restitution
    )
    print(f"Fixed-Step Integrator: Ran for {len(states_fixed) - 1} steps.")
    if states_fixed.requires_grad:
        loss_fixed = torch.sum(states_fixed[-1, :2] ** 2)
        loss_fixed.backward()
        print(f"Gradient of loss w.r.t y_initial_state (fixed-step): {y_initial_fixed.grad}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(states_adaptive[:, 0].detach().numpy(), states_adaptive[:, 1].detach().numpy(), 'b-o',
             label='Adaptive Integrator', markersize=3)
    plt.title('Adaptive Integrator')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(states_fixed[:, 0].detach().numpy(), states_fixed[:, 1].detach().numpy(), 'r-s',
             label='Fixed-Step Integrator', markersize=3)
    plt.title('Fixed-Step Integrator')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("integrator_comparison_trajectories_consolidated.png")
    print("\nSaved example trajectories plot to 'integrator_comparison_trajectories_consolidated.png'")