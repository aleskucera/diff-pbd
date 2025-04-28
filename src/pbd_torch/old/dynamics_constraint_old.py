from typing import Tuple
import time

import torch
import warp as wp
from torch import Tensor
from jaxtyping import Float
from pbd_torch.model import Model


# Existing dynamics_residual and dynamics_derivatives remain unchanged
def dynamics_residual(
        mass_matrix: Float[torch.Tensor, "B 6 6"],
        g_accel: Float[torch.Tensor, "6 1"],
        joint_parent: Tensor,  # [D]
        joint_child: Tensor,  # [D]
        body_vel: Float[torch.Tensor, "B 6 1"],
        body_vel_prev: Float[torch.Tensor, "B 6 1"],
        lambda_n: Float[torch.Tensor, "B C 1"],
        lambda_t: Float[torch.Tensor, "B 2C 1"],
        lambda_j: Float[torch.Tensor, "5D 1"],  # [5*D, 1]
        body_f: Float[torch.Tensor, "B 6 1"],
        J_n: Float[torch.Tensor, "B C 6"],
        J_t: Float[torch.Tensor, "B 2C 6"],
        J_j_p: Float[torch.Tensor, "D 5 6"],  # [D, 5, 6]
        J_j_c: Float[torch.Tensor, "D 5 6"],  # [D, 5, 6]
        dt: float,
) -> Float[torch.Tensor, "B 6 1"]:
    """
    Compute the residuals for the dynamics equations in a physics simulation.

    Args:
        mass_matrix: Mass matrix of bodies, shape [B, 6, 6].
        g_accel: Gravitational acceleration, shape [6, 1].
        joint_parent: Parent body indices for joints, shape [D].
        joint_child: Child body indices for joints, shape [D].
        body_vel: Current body velocities, shape [B, 6, 1].
        body_vel_prev: Previous body velocities, shape [B, 6, 1].
        lambda_n: Normal contact impulses, shape [B, C, 1].
        lambda_t: Tangential friction impulses, shape [B, 2C, 1].
        lambda_j: Joint impulses, shape [5D, 1].
        body_f: External forces on bodies, shape [B, 6, 1].
        J_n: Normal contact Jacobians, shape [B, C, 6].
        J_t: Tangential friction Jacobians, shape [B, 2C, 6].
        J_j_p: Joint Jacobians for parent bodies, shape [D, 5, 6].
        J_j_c: Joint Jacobians for child bodies, shape [D, 5, 6].
        dt: Time step (scalar).

    Returns:
        Residuals of the dynamics constraints, shape [B, 6, 1].
    """
    B = body_vel.shape[0]
    D = joint_parent.shape[0]

    res = (
            torch.matmul(mass_matrix, (body_vel - body_vel_prev))
            - torch.matmul(J_n.transpose(1, 2), lambda_n)
            - torch.matmul(J_t.transpose(1, 2), lambda_t)
            - body_f * dt
            - torch.matmul(mass_matrix, g_accel.expand(B, 6, 1)) * dt
    )

    lambda_j_batch = lambda_j.view(D, 5, 1)  # [D, 5, 1]
    impulse_p = torch.matmul(J_j_p.transpose(1, 2), lambda_j_batch)  # [D, 6, 1]
    impulse_c = torch.matmul(J_j_c.transpose(1, 2), lambda_j_batch)  # [D, 6, 1]
    joint_impulse = torch.zeros(B, 6, 1, device=mass_matrix.device)
    joint_impulse.index_add_(0, joint_parent, impulse_p)
    joint_impulse.index_add_(0, joint_child, impulse_c)
    res = res - joint_impulse

    return res


def dynamics_derivatives(
        mass_matrix: Float[torch.Tensor, "B 6 6"],
        joint_parent: Tensor,  # [D]
        joint_child: Tensor,  # [D]
        J_n: Float[torch.Tensor, "B C 6"],
        J_t: Float[torch.Tensor, "B 2C 6"],
        J_j_p: Float[torch.Tensor, "D 5 6"],  # [D, 5, 6]
        J_j_c: Float[torch.Tensor, "D 5 6"],  # [D, 5, 6]
) -> Tuple[
    Float[torch.Tensor, "B 6 6"],
    Float[torch.Tensor, "B 6 C"],
    Float[torch.Tensor, "B 6 2C"],
    Float[torch.Tensor, "B 6 5D"],
]:
    """
    Compute the derivatives of the dynamics residuals with respect to velocities and impulses.

    Args:
        mass_matrix: Mass matrix of bodies, shape [B, 6, 6].
        joint_parent: Parent body indices for joints, shape [D].
        joint_child: Child body indices for joints, shape [D].
        J_n: Normal contact Jacobians, shape [B, C, 6].
        J_t: Tangential friction Jacobians, shape [B, 2C, 6].
        J_j_p: Joint Jacobians for parent bodies, shape [D, 5, 6].
        J_j_c: Joint Jacobians for child bodies, shape [D, 5, 6].

    Returns:
        Tuple containing:
            - ∂res/∂body_vel: Derivative w.r.t. body velocities, shape [B, 6, 6].
            - ∂res/∂lambda_n: Derivative w.r.t. normal impulses, shape [B, 6, C].
            - ∂res/∂lambda_t: Derivative w.r.t. tangential impulses, shape [B, 6, 2C].
            - ∂res/∂lambda_j: Derivative w.r.t. joint impulses, shape [B, 6, 5D].
    """
    B = mass_matrix.shape[0]
    D = joint_parent.shape[0]

    dres_dbody_vel = mass_matrix
    dres_dlambda_n = -J_n.transpose(1, 2)
    dres_dlambda_t = -J_t.transpose(1, 2)
    dres_dlambda_j = torch.zeros(B, 6, 5 * D, device=mass_matrix.device)

    J_j_p_reshaped = J_j_p.transpose(1, 2)  # [D, 6, 5]
    J_j_c_reshaped = J_j_c.transpose(1, 2)  # [D, 6, 5]
    block_indices = (
            torch.arange(5, device=mass_matrix.device).view(1, 1, 5)
            + 5 * torch.arange(D, device=mass_matrix.device).view(D, 1, 1)
    ).expand(D, 6, 5)
    contrib_p = torch.zeros(D, 6, 5 * D, device=mass_matrix.device).scatter_(
        2, block_indices, -J_j_p_reshaped
    )
    contrib_c = torch.zeros(D, 6, 5 * D, device=mass_matrix.device).scatter_(
        2, block_indices, -J_j_c_reshaped
    )
    dres_dlambda_j.index_add_(0, joint_parent, contrib_p)
    dres_dlambda_j.index_add_(0, joint_child, contrib_c)

    return dres_dbody_vel, dres_dlambda_n, dres_dlambda_t, dres_dlambda_j


# Existing Warp kernels remain unchanged
@wp.kernel
def calculate_base_residual_kernel(
        mass_matrix: wp.array(dtype=wp.spatial_matrix, ndim=1),  # [B]
        g_accel: wp.array(dtype=wp.spatial_vector, ndim=1),  # Constant spatial vector (length 1 array)
        body_vel: wp.array(dtype=wp.spatial_vector, ndim=1),  # [B]
        body_vel_prev: wp.array(dtype=wp.spatial_vector, ndim=1),  # [B]
        body_f: wp.array(dtype=wp.spatial_vector, ndim=1),  # [B]
        dt: wp.float32,
        res: wp.array(dtype=wp.spatial_vector, ndim=1),  # Output: [B]
):
    b = wp.tid()  # Body index

    # Calculate residual excluding impulses
    vel_diff = body_vel[b] - body_vel_prev[b]
    res_b = mass_matrix[b] * vel_diff
    res_b = res_b - body_f[b] * dt
    # Note: g_accel is a length-1 array, access with index 0
    res_b = res_b - (mass_matrix[b] * g_accel[0]) * dt

    # Initialize the residual for body b
    res[b] = res_b


@wp.kernel
def accumulate_contact_impulse_kernel(
        lambda_n: wp.array(dtype=wp.float32, ndim=2),  # [B, C]
        J_n: wp.array(dtype=wp.spatial_vector, ndim=2),  # [B, C] spatial_vectors
        res: wp.array(dtype=wp.spatial_vector, ndim=1),  # Residuals: [B] (atomic add)
):
    b, c = wp.tid()
    contact_impulse_contrib = J_n[b, c] * lambda_n[b, c]
    wp.atomic_add(res, b, -contact_impulse_contrib)


@wp.kernel
def accumulate_friction_impulse_kernel(
        lambda_t: wp.array(dtype=wp.float32, ndim=2),  # [B, 2C]
        J_t: wp.array(dtype=wp.spatial_vector, ndim=2),  # [B, 2C] spatial_vectors
        res: wp.array(dtype=wp.spatial_vector, ndim=1),  # Residuals: [B] (atomic add)
):
    b, t = wp.tid() # Body index and tangential impulse index
    friction_impulse_contrib = J_t[b, t] * lambda_t[b, t]
    wp.atomic_add(res, b, -friction_impulse_contrib)


@wp.kernel
def accumulate_joint_impulse_kernel(
        joint_parent: wp.array(dtype=wp.int32, ndim=1),  # [D]
        joint_child: wp.array(dtype=wp.int32, ndim=1),  # [D]
        lambda_j: wp.array(dtype=wp.float32, ndim=2),  # [D, 5] - input impulse values
        J_j_p: wp.array(dtype=wp.spatial_vector, ndim=2),  # [D, 5] - parent Jacobians
        J_j_c: wp.array(dtype=wp.spatial_vector, ndim=2),  # [D, 5] - child Jacobians
        res: wp.array(dtype=wp.spatial_vector, ndim=1),  # Residuals: [B] (atomic add)
):
    d, k = wp.tid() # Joint index and impulse component index
    impulse_contrib_p = J_j_p[d, k] * lambda_j[d, k]
    impulse_contrib_c = J_j_c[d, k] * lambda_j[d, k]
    wp.atomic_add(res, joint_parent[d], -impulse_contrib_p)
    wp.atomic_add(res, joint_child[d], -impulse_contrib_c)


def launch_dynamics_kernels(
        mass_matrix: wp.array(dtype=wp.spatial_matrix, ndim=1),
        g_accel: wp.array(dtype=wp.spatial_vector, ndim=1),
        joint_parent: wp.array(dtype=wp.int32, ndim=1),
        joint_child: wp.array(dtype=wp.int32, ndim=1),
        body_vel: wp.array(dtype=wp.spatial_vector, ndim=1),
        body_vel_prev: wp.array(dtype=wp.spatial_vector, ndim=1),
        lambda_n: wp.array(dtype=wp.float32, ndim=2),
        lambda_t: wp.array(dtype=wp.float32, ndim=2),
        lambda_j: wp.array(dtype=wp.float32, ndim=2),  # [D, 5]
        body_f: wp.array(dtype=wp.spatial_vector, ndim=1),
        J_n: wp.array(dtype=wp.spatial_vector, ndim=2),  # [B, C] spatial_vectors
        J_t: wp.array(dtype=wp.spatial_vector, ndim=2),  # [B, 2C] spatial_vectors
        J_j_p: wp.array(dtype=wp.spatial_vector, ndim=2),  # [D, 5] spatial_vectors
        J_j_c: wp.array(dtype=wp.spatial_vector, ndim=2),  # [D, 5] spatial_vectors
        dt: float,
        # Output:
        res: wp.array(dtype=wp.spatial_vector, ndim=1),  # Output residual array
        # ---
        device: wp.context.Device
):
    """
    Launches the sequence of Warp kernels to compute the dynamics residual,
    assuming all inputs are already in Warp array format.

    Args:
        mass_matrix: Mass matrix of bodies in Warp array format.
        g_accel: Gravitational acceleration in Warp array format.
        joint_parent: Parent body indices for joints in Warp array format.
        joint_child: Child body indices for joints in Warp array format.
        body_vel: Current body velocities in Warp array format.
        body_vel_prev: Previous body velocities in Warp array format.
        lambda_n: Normal contact impulses in Warp array format.
        lambda_t: Tangential friction impulses in Warp array format.
        lambda_j: Joint impulses in Warp array format ([D, 5]).
        body_f: External forces on bodies in Warp array format.
        J_n: Normal contact Jacobians in Warp array format.
        J_t: Tangential friction Jacobians in Warp array format.
        J_j_p: Joint Jacobians for parent bodies in Warp array format.
        J_j_c: Joint Jacobians for child bodies in Warp array format.
        dt: Time step (scalar float).
        res: Output residual array (must be pre-allocated and zeroed).
        device: The Warp device to use for launching kernels.
    """
    B = mass_matrix.shape[0]
    C = 0 if lambda_n.ndim < 2 else lambda_n.shape[1]  # Handle case with no contacts
    D = joint_parent.shape[0]

    # --- Clear the output residual array before launching kernels ---
    res.zero_()

    # --- Launch the Warp kernels ---

    # 1. Calculate the base residual (excluding impulses) - Parallelized per Body
    wp.launch(
        kernel=calculate_base_residual_kernel,
        dim=B,
        inputs=[
            mass_matrix, g_accel, body_vel, body_vel_prev,
            body_f, float(dt), res,
        ],
        device=device,
    )

    # 2. Accumulate Contact Impulses - Parallelized per Body*Contact
    if C > 0:
        wp.launch(
            kernel=accumulate_contact_impulse_kernel,
            dim=(B, C),
            inputs=[lambda_n, J_n, res],
            device=device,
        )

    # 3. Accumulate Friction Impulses - Parallelized per Body*2C
    if C > 0:
        wp.launch(
            kernel=accumulate_friction_impulse_kernel,
            dim=(B, 2 * C),
            inputs=[lambda_t, J_t, res],
            device=device,
        )

    # 4. Accumulate Joint Impulses - Parallelized per Joint * 5 (scalar impulse components)
    if D > 0:
        wp.launch(
            kernel=accumulate_joint_impulse_kernel,
            dim=(D, 5),
            inputs=[
                joint_parent, joint_child, lambda_j, J_j_p, J_j_c, res
            ],
            device=device,
        )

@wp.kernel
def build_dres_dlambda_n_kernel(
        J_n: wp.array(dtype=wp.spatial_vector, ndim=2), # [B, C]
        output: wp.array(dtype=wp.float32, ndim=3) # [B, 6, C]
):
    b, i, c = wp.tid() # body, spatial component [0-6), contact index [0-C)

    # d(res[b][i])/d(lambda_n[b, c]) = -J_n[b, c][i] - Corrected access
    output[b, i, c] = -J_n[b, c][i]
    
@wp.kernel
def build_dres_dlambda_t_kernel(
        J_t: wp.array(dtype=wp.spatial_vector, ndim=2), # [B, 2C]
        output: wp.array(dtype=wp.float32, ndim=3) # [B, 6, 2C]
):
    b, i, t = wp.tid() # body, spatial component [0-6), tangential index [0-2C)

    # d(res[b][i])/d(lambda_t[b, t]) = -J_t[b, t][i] - Corrected access
    output[b, i, t] = -J_t[b, t][i]

@wp.kernel
def build_dres_dlambda_j_kernel(
        joint_parent: wp.array(dtype=wp.int32, ndim=1), # [D]
        joint_child: wp.array(dtype=wp.int32, ndim=1), # [D]
        J_j_p: wp.array(dtype=wp.spatial_vector, ndim=2), # [D, 5]
        J_j_c: wp.array(dtype=wp.spatial_vector, ndim=2), # [D, 5]
        output: wp.array(dtype=wp.float32, ndim=3) # [B, 6, 5D]
):
    b, i, flat_joint_impulse_idx = wp.tid() # body [0-B), spatial component [0-6), flattened joint impulse index [0-5D)

    D = joint_parent.shape[0]
    impulse_scalar_dim = 5

    # Map the flat joint impulse index back to joint index (d) and scalar component index (k)
    d = flat_joint_impulse_idx // impulse_scalar_dim
    k = flat_joint_impulse_idx % impulse_scalar_dim

    value = 0.0 # Derivative is zero by default for bodies not involved in this impulse

    # Ensure we don't access out of bounds for J_j_p/c if D is 0 (though kernel shouldn't launch if D=0)
    if d < D: # Add this check for safety, even if kernel dim should handle it
        parent_body = joint_parent[d]
        child_body = joint_child[d]

        if b == parent_body:
            # Derivative of residual of parent body w.r.t. lambda_j[d, k] is -J_j_p[d, k][i] - Corrected access
            value = -J_j_p[d, k][i]
        elif b == child_body:
             # Derivative of residual of child body w.r.t. lambda_j[d, k] is -J_j_c[d, k][i] - Corrected access
            value = -J_j_c[d, k][i]

    output[b, i, flat_joint_impulse_idx] = value



def launch_dynamics_derivative_kernels(
        mass_matrix: wp.array(dtype=wp.spatial_matrix, ndim=1),  # [B]
        J_n: wp.array(dtype=wp.spatial_vector, ndim=2),  # [B, C]
        J_t: wp.array(dtype=wp.spatial_vector, ndim=2),  # [B, 2C]
        joint_parent: wp.array(dtype=wp.int32, ndim=1),  # [D]
        joint_child: wp.array(dtype=wp.int32, ndim=1),  # [D]
        J_j_p: wp.array(dtype=wp.spatial_vector, ndim=2),  # [D, 5]
        J_j_c: wp.array(dtype=wp.spatial_vector, ndim=2),  # [D, 5]
        # Outputs:
        dres_dbody_vel: wp.array(dtype=wp.spatial_matrix, ndim=1),  # [B]
        dres_dlambda_n: wp.array(dtype=wp.float32, ndim=3),  # [B, 6, C]
        dres_dlambda_t: wp.array(dtype=wp.float32, ndim=3),  # [B, 6, 2C]
        dres_dlambda_j: wp.array(dtype=wp.float32, ndim=3),  # [B, 6, 5D]
        device: wp.context.Device
):
    """
    Launches the sequence of Warp kernels to compute the derivatives of the
    dynamics residual, assuming all inputs are already in Warp array format.

    Args:
        mass_matrix: Mass matrix of bodies in Warp array format ([B]).
        J_n: Normal contact Jacobians in Warp array format ([B, C]).
        J_t: Tangential friction Jacobians in Warp array format ([B, 2C]).
        joint_parent: Parent body indices for joints in Warp array format ([D]).
        joint_child: Child body indices for joints in Warp array format ([D]).
        J_j_p: Joint Jacobians for parent bodies in Warp array format ([D, 5]).
        J_j_c: Joint Jacobians for child bodies in Warp array format ([D, 5]).
        dres_dbody_vel: Output array for ∂res/∂body_vel ([B, 6, 6] as spatial_matrix).
        dres_dlambda_n: Output array for ∂res/∂lambda_n ([B, 6, C]).
        dres_dlambda_t: Output array for ∂res/∂lambda_t ([B, 6, 2C]).
        dres_dlambda_j: Output array for ∂res/∂lambda_j ([B, 6, 5D]).
        device: The Warp device to use for launching kernels.
    """
    B = mass_matrix.shape[0]
    C = J_n.shape[1] if J_n.ndim > 1 else 0
    D = joint_parent.shape[0]
    flat_joint_impulse_dim = 5 * D

    # ∂res/∂body_vel = mass_matrix (simple copy)
    # This is implicitly handled if dres_dbody_vel is just the mass_matrix array view.
    # If it's a separate array, a kernel would be needed:
    # for b in range(B): dres_dbody_vel[b] = mass_matrix[b] -> wp.copy(dres_dbody_vel, mass_matrix)
    # Assuming dres_dbody_vel is separate storage and needs to be filled:
    wp.copy(dres_dbody_vel, mass_matrix)


    # ∂res/∂lambda_n = -J_n.transpose(1, 2)
    # This requires a kernel to build the 3D tensor from spatial vectors
    if C > 0:
        wp.launch(
            kernel=build_dres_dlambda_n_kernel,
            dim=(B, 6, C), # Iterate over B, spatial dim, and C
            inputs=[J_n, dres_dlambda_n],
            device=device
        )
    # If C is 0, dres_dlambda_n remains its initialized value (likely zeros) which is correct.

    # ∂res/∂lambda_t = -J_t.transpose(1, 2)
    if C > 0: # J_t dimension depends on C
        wp.launch(
            kernel=build_dres_dlambda_t_kernel,
            dim=(B, 6, 2 * C), # Iterate over B, spatial dim, and 2C
            inputs=[J_t, dres_dlambda_t],
            device=device
        )
    # If C is 0, dres_dlambda_t remains its initialized value (likely zeros) which is correct.

    # ∂res/∂lambda_j
    # Requires a kernel due to the scatter/index_add logic
    if D > 0:
         wp.launch(
            kernel=build_dres_dlambda_j_kernel,
            dim=(B, 6, flat_joint_impulse_dim), # Iterate over B, spatial dim, and 5D
            inputs=[joint_parent, joint_child, J_j_p, J_j_c, dres_dlambda_j],
            device=device
         )
    # If D is 0, dres_dlambda_j remains its initialized value (likely zeros) which is correct.


# --- Test and Benchmarking Functions ---
def setup_test_case_data(B: int, C: int, D: int, device: torch.device):
    """Generates random data for a given test case configuration."""
    # Use more stable data generation where appropriate (e.g., positive mass)
    g_accel = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -9.81], device=device).view(6, 1)

    # Generate a symmetric positive definite mass matrix for stability if needed, randomize otherwise
    mass_matrix_sym_rand = torch.randn(B, 6, 6, device=device)
    mass_matrix = torch.matmul(mass_matrix_sym_rand, mass_matrix_sym_rand.transpose(1,2)) + torch.eye(6, device=device).unsqueeze(0) # Adding identity to ensure positive definite
    # mass_matrix = torch.randn(B, 6, 6, device=device) # Original random example

    # Ensure D joints have distinct parent and child indices within bounds [0, B-1]
    if B <= 1 and D > 0:
         raise ValueError("Cannot create a joint test case with B <= 1 and D > 0")

    joint_parent = torch.empty(D, dtype=torch.int32, device=device)
    joint_child = torch.empty(D, dtype=torch.int32, device=device)
    for i in range(D):
        parent = torch.randint(0, B, (1,), device=device).item()
        child = torch.randint(0, B, (1,), device=device).item()
        while parent == child:
             child = torch.randint(0, B, (1,), device=device).item()
        joint_parent[i] = parent
        joint_child[i] = child


    body_vel = torch.randn(B, 6, 1, device=device, requires_grad=True)
    body_vel_prev = torch.randn(B, 6, 1, device=device)
    lambda_n = torch.randn(B, C, 1, device=device, requires_grad=True) if C > 0 else torch.empty(B, 0, 1, device=device) # No grad for empty tensors
    lambda_t = torch.randn(B, 2 * C, 1, device=device, requires_grad=True) if C > 0 else torch.empty(B, 0, 1, device=device) # No grad for empty tensors
    lambda_j = torch.randn(5 * D, 1, device=device, requires_grad=True) if D > 0 else torch.empty(0, 1, device=device) # No grad for empty tensors
    body_f = torch.randn(B, 6, 1, device=device)
    J_n = torch.randn(B, C, 6, device=device) if C > 0 else torch.empty(B, 0, 6, device=device)
    J_t = torch.randn(B, 2 * C, 6, device=device) if C > 0 else torch.empty(B, 0, 6, device=device)
    J_j_p = torch.randn(D, 5, 6, device=device) if D > 0 else torch.empty(0, 5, 6, device=device)
    J_j_c = torch.randn(D, 5, 6, device=device) if D > 0 else torch.empty(0, 5, 6, device=device)
    dt = 0.01

    # Set requires_grad=False for inputs that analytical derivatives are not taken w.r.t.
    mass_matrix.requires_grad_(False)
    g_accel.requires_grad_(False)
    body_vel_prev.requires_grad_(False)
    body_f.requires_grad_(False)
    J_n.requires_grad_(False)
    J_t.requires_grad_(False)
    J_j_p.requires_grad_(False)
    J_j_c.requires_grad_(False)
    # dt is a scalar, no grad needed

    return {
        "mass_matrix": mass_matrix,
        "g_accel": g_accel,
        "joint_parent": joint_parent,
        "joint_child": joint_child,
        "body_vel": body_vel,
        "body_vel_prev": body_vel_prev,
        "lambda_n": lambda_n,
        "lambda_t": lambda_t,
        "lambda_j": lambda_j,
        "body_f": body_f,
        "J_n": J_n,
        "J_t": J_t,
        "J_j_p": J_j_p,
        "J_j_c": J_j_c,
        "dt": dt,
    }


def run_comparison_test(test_name: str, B: int, C: int, D: int, device: torch.device):
    """
    Runs a comparison test for both residual and derivative functions.
    Compares PyTorch numerical and analytical derivatives against Warp analytical derivatives.
    """
    print(f"Running Comparison Test '{test_name}' (B={B}, C={C}, D={D})")
    data = setup_test_case_data(B, C, D, device)
    dt = data["dt"]
    wp_device = wp.get_device()

    # --- Residual Comparison ---
    # Convert data to Warp arrays for the Warp residual launch
    # Ensure contiguity for efficient transfer
    wp_mass_matrix = wp.from_torch(data['mass_matrix'].contiguous(), dtype=wp.spatial_matrix)
    wp_g_accel = wp.from_torch(data['g_accel'].squeeze().contiguous(), dtype=wp.spatial_vector)
    wp_joint_parent = wp.from_torch(data['joint_parent'].contiguous().to(torch.int32), dtype=wp.int32) # Ensure int32
    wp_joint_child = wp.from_torch(data['joint_child'].contiguous().to(torch.int32), dtype=wp.int32)   # Ensure int32
    wp_body_vel = wp.from_torch(data['body_vel'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)
    wp_body_vel_prev = wp.from_torch(data['body_vel_prev'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)
    wp_body_f = wp.from_torch(data['body_f'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)

    C_actual = data['lambda_n'].shape[1] if data['lambda_n'].ndim > 1 else 0
    D_actual = data['joint_parent'].shape[0]

    wp_lambda_n = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C_actual == 0 else wp.from_torch(
        data['lambda_n'].squeeze(-1).contiguous(), dtype=wp.float32)
    wp_lambda_t = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C_actual == 0 else wp.from_torch(
        data['lambda_t'].squeeze(-1).contiguous(), dtype=wp.float32)
    wp_J_n = wp.empty(shape=(B, 0), dtype=wp.spatial_vector, device=wp_device) if C_actual == 0 else wp.from_torch(
        data['J_n'].contiguous(), dtype=wp.spatial_vector)
    wp_J_t = wp.empty(shape=(B, 0), dtype=wp.spatial_vector, device=wp_device) if C_actual == 0 else wp.from_torch(
        data['J_t'].contiguous(), dtype=wp.spatial_vector)

    wp_J_j_p = wp.empty(shape=(0, 5), dtype=wp.spatial_vector, device=wp_device) if D_actual == 0 else wp.from_torch(
        data['J_j_p'].contiguous(), dtype=wp.spatial_vector)
    wp_J_j_c = wp.empty(shape=(0, 5), dtype=wp.spatial_vector, device=wp_device) if D_actual == 0 else wp.from_torch(
        data['J_j_c'].contiguous(), dtype=wp.spatial_vector)
    wp_lambda_j = wp.empty(shape=(0, 5), dtype=wp.float32, device=wp_device) if D_actual == 0 else wp.from_torch(
        data['lambda_j'].view(D_actual, 5).contiguous(), dtype=wp.float32)

    wp_res = wp.zeros(shape=B, dtype=wp.spatial_vector, device=wp_device, requires_grad=False)

    # Run the Warp residual kernels
    launch_dynamics_kernels(
        wp_mass_matrix, wp_g_accel, wp_joint_parent, wp_joint_child,
        wp_body_vel, wp_body_vel_prev, wp_lambda_n, wp_lambda_t,
        wp_lambda_j, wp_body_f, wp_J_n, wp_J_t, wp_J_j_p, wp_J_j_c,
        dt, wp_res, device=wp_device
    )

    # Convert Warp residual result back to PyTorch for comparison
    res_warp = wp.to_torch(wp_res).clone().view(B, 6).unsqueeze(-1)
    res_pytorch = dynamics_residual(**data)

    assert torch.allclose(res_pytorch, res_warp, atol=1e-5), f"{test_name}: Residuals do not match"
    print(f"{test_name}: Residual comparison passed.")

    # --- Derivative Comparison ---
    # PyTorch analytical derivatives
    dres_dbody_vel_pt_ana, dres_dlambda_n_pt_ana, dres_dlambda_t_pt_ana, dres_dlambda_j_pt_ana = dynamics_derivatives(
        data['mass_matrix'], data['joint_parent'], data['joint_child'],
        data['J_n'], data['J_t'], data['J_j_p'], data['J_j_c'],
    )

    # PyTorch automatic derivatives (for validation)
    res_pytorch_grad = dynamics_residual(**data)

    # Compute gradients of the residual w.r.t each input
    # Need a placeholder in the .grad calls for inputs that don't have requires_grad=True
    grad_outputs_placeholder = torch.ones_like(res_pytorch_grad)

    # d(res)/d(body_vel)
    grad_body_vel = torch.autograd.grad(res_pytorch_grad, data['body_vel'], grad_outputs=grad_outputs_placeholder, create_graph=False)[0]
    dres_dbody_vel_pt_auto = grad_body_vel # Shape [B, 6, 1]. Need to match analytical shape [B, 6, 6]
    # To match [B, 6, 6], we'd need to compute the Jacobian explicitly with torch.autograd.functional.jacobian
    # However, comparing component-wise derivatives is sufficient for validation.
    # For d(res[b][i][0])/d(body_vel[b][j][0]), this is mass_matrix[b][i][j].
    # So, grad_body_vel[b][i][0] should equal sum_j(mass_matrix[b][i][j] * grad_outputs_placeholder[b][j][0])
    # Since grad_outputs_placeholder is ones, grad_body_vel[b][i][0] = sum_j(mass_matrix[b][i][j])
    # A more direct comparison is needed for the full derivative matrix.

    # Let's compute the full Jacobian for PyTorch for direct comparison with analytical
    from torch.autograd.functional import jacobian as torch_jacobian

    def residual_flat(
        body_vel_flat, lambda_n_flat, lambda_t_flat, lambda_j_flat,
        mass_matrix, g_accel, joint_parent, joint_child, body_vel_prev,
        body_f, J_n, J_t, J_j_p, J_j_c, dt):

        B = mass_matrix.shape[0]
        C_actual = J_n.shape[1] if J_n.ndim > 1 else 0
        D_actual = joint_parent.shape[0]

        body_vel_reshaped = body_vel_flat.view(B, 6, 1)
        lambda_n_reshaped = lambda_n_flat.view(B, C_actual, 1) if C_actual > 0 else torch.empty(B, 0, 1, device=body_vel_flat.device)
        lambda_t_reshaped = lambda_t_flat.view(B, 2 * C_actual, 1) if C_actual > 0 else torch.empty(B, 0, 1, device=body_vel_flat.device)
        lambda_j_reshaped = lambda_j_flat.view(D_actual * 5, 1) if D_actual > 0 else torch.empty(0, 1, device=body_vel_flat.device)

        return dynamics_residual(
            mass_matrix, g_accel, joint_parent, joint_child, body_vel_reshaped,
            body_vel_prev, lambda_n_reshaped, lambda_t_reshaped, lambda_j_reshaped,
            body_f, J_n, J_t, J_j_p, J_j_c, dt
            ).view(-1) # Flatten the output residual


    # Only compute gradients w.r.t. inputs that `dynamics_derivatives` returns
    inputs_to_grad = (data['body_vel'].view(-1), data['lambda_n'].view(-1), data['lambda_t'].view(-1), data['lambda_j'].view(-1),)

    # Pad inputs with non-grad tensors
    other_inputs = (
        data['mass_matrix'], data['g_accel'], data['joint_parent'], data['joint_child'], data['body_vel_prev'],
        data['body_f'], data['J_n'], data['J_t'], data['J_j_p'], data['J_j_c'], dt
    )

    # Compute overall Jacobian [B*6, B*6 + B*C + B*2C + 5*D]
    # J_overall = torch_jacobian(residual_flat, inputs_to_grad, args=other_inputs)

    # Computing each part of the Jacobian separately for clarity and direct comparison
    dres_dbody_vel_pt_auto_matrix = torch_jacobian(lambda x: residual_flat(x, inputs_to_grad[1], inputs_to_grad[2], inputs_to_grad[3], *other_inputs), inputs_to_grad[0])
    dres_dbody_vel_pt_auto_matrix = dres_dbody_vel_pt_auto_matrix.view(B, 6, B, 6).permute(0, 2, 1, 3) # Shape [B, B, 6, 6]
    # The derivative of res[b'][i'] w.r.t body_vel[b][j] is zero if b' != b.
    # Extract the block diagonal: dres[b][i] / dbody_vel[b][j]
    dres_dbody_vel_pt_auto = torch.zeros_like(data['mass_matrix'])
    for b_idx in range(B):
        dres_dbody_vel_pt_auto[b_idx, :, :] = dres_dbody_vel_pt_auto_matrix[b_idx, b_idx, :, :]

    dres_dlambda_n_pt_auto_matrix = torch_jacobian(lambda x: residual_flat(inputs_to_grad[0], x, inputs_to_grad[2], inputs_to_grad[3], *other_inputs), inputs_to_grad[1])
    dres_dlambda_n_pt_auto_matrix = dres_dlambda_n_pt_auto_matrix.view(B, 6, B, C_actual).permute(0, 2, 1, 3) # Shape [B, B, 6, C]
     # The derivative of res[b'][i'] w.r.t lambda_n[b][c] is zero if b' != b.
    dres_dlambda_n_pt_auto = torch.zeros_like(data['J_n'].transpose(1,2)) # Shape [B, 6, C]
    if C_actual > 0:
        for b_idx in range(B):
            dres_dlambda_n_pt_auto[b_idx, :, :] = dres_dlambda_n_pt_auto_matrix[b_idx, b_idx, :, :]

    dres_dlambda_t_pt_auto_matrix = torch_jacobian(lambda x: residual_flat(inputs_to_grad[0], inputs_to_grad[1], x, inputs_to_grad[3], *other_inputs), inputs_to_grad[2])
    dres_dlambda_t_pt_auto_matrix = dres_dlambda_t_pt_auto_matrix.view(B, 6, B, 2*C_actual).permute(0, 2, 1, 3) # Shape [B, B, 6, 2C]
    # The derivative of res[b'][i'] w.r.t lambda_t[b][t] is zero if b' != b.
    dres_dlambda_t_pt_auto = torch.zeros_like(data['J_t'].transpose(1,2)) # Shape [B, 6, 2C]
    if C_actual > 0:
       for b_idx in range(B):
            dres_dlambda_t_pt_auto[b_idx, :, :] = dres_dlambda_t_pt_auto_matrix[b_idx, b_idx, :, :]

    dres_dlambda_j_pt_auto_matrix = torch_jacobian(lambda x: residual_flat(inputs_to_grad[0], inputs_to_grad[1], inputs_to_grad[2], x, *other_inputs), inputs_to_grad[3])
    dres_dlambda_j_pt_auto_matrix = dres_dlambda_j_pt_auto_matrix.view(B, 6, D_actual * 5) # Shape [B, 6, 5D]


    # Convert data to Warp arrays for the Warp derivative launch
    # Outputs need to be pre-allocated
    wp_dres_dbody_vel = wp.zeros(shape=B, dtype=wp.spatial_matrix, device=wp_device) # [B, 6, 6] as spatial_matrix
    wp_dres_dlambda_n = wp.zeros(shape=(B, 6, C_actual), dtype=wp.float32, device=wp_device)  # [B, 6, C]
    wp_dres_dlambda_t = wp.zeros(shape=(B, 6, 2 * C_actual), dtype=wp.float32, device=wp_device)  # [B, 6, 2C]
    wp_dres_dlambda_j = wp.zeros(shape=(B, 6, 5 * D_actual), dtype=wp.float32, device=wp_device)  # [B, 6, 5D]

    # Run the Warp derivative kernels
    launch_dynamics_derivative_kernels(
        wp_mass_matrix, wp_J_n, wp_J_t, wp_joint_parent, wp_joint_child, wp_J_j_p, wp_J_j_c,
        wp_dres_dbody_vel, wp_dres_dlambda_n, wp_dres_dlambda_t, wp_dres_dlambda_j, device=wp_device
    )

    # Convert Warp derivative results back to PyTorch for comparison
    dres_dbody_vel_warp = wp.to_torch(wp_dres_dbody_vel).clone().view(B, 6, 6)
    dres_dlambda_n_warp = wp.to_torch(wp_dres_dlambda_n).clone()
    dres_dlambda_t_warp = wp.to_torch(wp_dres_dlambda_t).clone()
    dres_dlambda_j_warp = wp.to_torch(wp_dres_dlambda_j).clone()

    # Compare Warp analytical results with PyTorch analytical/automatic
    assert torch.allclose(dres_dbody_vel_pt_ana, dres_dbody_vel_warp, atol=1e-5), f"{test_name}: dres/dbody_vel mismatch"
    print(f"{test_name}: dres/dbody_vel comparison passed.")

    if C_actual > 0:
        assert torch.allclose(dres_dlambda_n_pt_ana, dres_dlambda_n_warp, atol=1e-5), f"{test_name}: dres/dlambda_n mismatch"
        print(f"{test_name}: dres/dlambda_n comparison passed.")

        assert torch.allclose(dres_dlambda_t_pt_ana, dres_dlambda_t_warp, atol=1e-5), f"{test_name}: dres/dlambda_t mismatch"
        print(f"{test_name}: dres/dlambda_t comparison passed.")
    else:
        # Ensure empty tensors match or are both zero
        assert dres_dlambda_n_warp.shape == torch.Size([B, 6, 0])
        assert dres_dlambda_t_warp.shape == torch.Size([B, 6, 0])
        print(f"{test_name}: dres/dlambda_n (empty) comparison passed.")
        print(f"{test_name}: dres/dlambda_t (empty) comparison passed.")


    if D_actual > 0:
         assert torch.allclose(dres_dlambda_j_pt_ana, dres_dlambda_j_warp, atol=1e-5), f"{test_name}: dres/dlambda_j mismatch"
         print(f"{test_name}: dres/dlambda_j comparison passed.")
    else:
        # Ensure empty tensors match or are both zero
        assert dres_dlambda_j_warp.shape == torch.Size([B, 6, 0])
        print(f"{test_name}: dres/dlambda_j (empty) comparison passed.")


    # Validate PyTorch analytical with PyTorch automatic (optional, but good for debugging analytical)
    assert torch.allclose(dres_dbody_vel_pt_ana, dres_dbody_vel_pt_auto, atol=1e-5), f"{test_name}: PT Analytical vs Auto dres/dbody_vel mismatch"
    print(f"{test_name}: PT Analytical vs Auto dres/dbody_vel validation passed.")

    if C_actual > 0:
         assert torch.allclose(dres_dlambda_n_pt_ana, dres_dlambda_n_pt_auto, atol=1e-5), f"{test_name}: PT Analytical vs Auto dres/dlambda_n mismatch"
         print(f"{test_name}: PT Analytical vs Auto dres/dlambda_n validation passed.")

         assert torch.allclose(dres_dlambda_t_pt_ana, dres_dlambda_t_pt_auto, atol=1e-5), f"{test_name}: PT Analytical vs Auto dres/dlambda_t mismatch"
         print(f"{test_name}: PT Analytical vs Auto dres/dlambda_t validation passed.")

    if D_actual > 0:
        assert torch.allclose(dres_dlambda_j_pt_ana, dres_dlambda_j_pt_auto_matrix, atol=1e-5), f"{test_name}: PT Analytical vs Auto dres/dlambda_j mismatch"
        print(f"{test_name}: PT Analytical vs Auto dres/dlambda_j validation passed.")

    print(f"{test_name} passed all comparisons.")


def run_performance_benchmark(B: int, C: int, D: int, device: torch.device, num_runs: int = 100):
    """Runs a performance benchmark for PyTorch and Warp (with graph)."""
    print(f"\nRunning Performance Test (B={B}, C={C}, D={D})")
    data = setup_test_case_data(B, C, D, device)
    dt = data["dt"]

    # PyTorch performance (Residual + Analytical Derivatives)
    # Warm-up
    res_pytorch = dynamics_residual(**data)
    dres_dbody_vel_pt_ana, dres_dlambda_n_pt_ana, dres_dlambda_t_pt_ana, dres_dlambda_j_pt_ana = dynamics_derivatives(
        data['mass_matrix'], data['joint_parent'], data['joint_child'],
        data['J_n'], data['J_t'], data['J_j_p'], data['J_j_c'],
    )
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_runs):
        res_pytorch = dynamics_residual(**data)
        dres_dbody_vel_pt_ana, dres_dlambda_n_pt_ana, dres_dlambda_t_pt_ana, dres_dlambda_j_pt_ana = dynamics_derivatives(
            data['mass_matrix'], data['joint_parent'], data['joint_child'],
            data['J_n'], data['J_t'], data['J_j_p'], data['J_j_c'],
        )
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_runs
    print(f"PyTorch average time per call (Residual + Analytical Derivatives): {pytorch_time:.6f} seconds")

    # Warp performance (Residual + Analytical Derivatives with graph)
    wp.init() # Ensure Warp is initialized
    wp_device = wp.get_device()

    # Convert data to Warp arrays for graph capture
    wp_mass_matrix = wp.from_torch(data['mass_matrix'].contiguous(), dtype=wp.spatial_matrix)
    wp_g_accel = wp.from_torch(data['g_accel'].squeeze().contiguous(), dtype=wp.spatial_vector)
    wp_joint_parent = wp.from_torch(data['joint_parent'].contiguous().to(torch.int32), dtype=wp.int32)
    wp_joint_child = wp.from_torch(data['joint_child'].contiguous().to(torch.int32), dtype=wp.int32)
    wp_body_vel = wp.from_torch(data['body_vel'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)
    wp_body_vel_prev = wp.from_torch(data['body_vel_prev'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)
    wp_body_f = wp.from_torch(data['body_f'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)

    C_actual = data['lambda_n'].shape[1] if data['lambda_n'].ndim > 1 else 0
    D_actual = data['joint_parent'].shape[0]
    flat_joint_impulse_dim = 5 * D_actual

    wp_lambda_n = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C_actual == 0 else wp.from_torch(
        data['lambda_n'].squeeze(-1).contiguous(), dtype=wp.float32)
    wp_lambda_t = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C_actual == 0 else wp.from_torch(
        data['lambda_t'].squeeze(-1).contiguous(), dtype=wp.float32)
    wp_J_n = wp.empty(shape=(B, 0), dtype=wp.spatial_vector) if C_actual == 0 else wp.from_torch(
        data['J_n'].contiguous(), dtype=wp.spatial_vector)
    wp_J_t = wp.empty(shape=(B, 0), dtype=wp.spatial_vector) if C_actual == 0 else wp.from_torch(
        data['J_t'].contiguous(), dtype=wp.spatial_vector)

    wp_J_j_p = wp.empty(shape=(0, 5), dtype=wp.spatial_vector) if D_actual == 0 else wp.from_torch(
        data['J_j_p'].contiguous(), dtype=wp.spatial_vector)
    wp_J_j_c = wp.empty(shape=(0, 5), dtype=wp.spatial_vector) if D_actual == 0 else wp.from_torch(
        data['J_j_c'].contiguous(), dtype=wp.spatial_vector)
    wp_lambda_j = wp.empty(shape=(0, 5), dtype=wp.float32) if D_actual == 0 else wp.from_torch(
        data['lambda_j'].view(D_actual, 5).contiguous(), dtype=wp.float32)

    wp_output_res = wp.zeros(shape=B, dtype=wp.spatial_vector, device=wp_device, requires_grad=False)
    wp_dres_dbody_vel = wp.zeros(shape=B, dtype=wp.spatial_matrix, device=wp_device, requires_grad=False)
    wp_dres_dlambda_n = wp.zeros(shape=(B, 6, C_actual), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_dres_dlambda_t = wp.zeros(shape=(B, 6, 2 * C_actual), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_dres_dlambda_j = wp.zeros(shape=(B, 6, flat_joint_impulse_dim), dtype=wp.float32, device=wp_device, requires_grad=False)


    # Capture the graph
    graph = None
    with wp.ScopedCapture() as capture:
        wp_output_res.zero_()  # Clear the output tensor within capture
        wp_dres_dbody_vel.zero_()
        wp_dres_dlambda_n.zero_()
        wp_dres_dlambda_t.zero_()
        wp_dres_dlambda_j.zero_()

        # Launch residual kernels
        launch_dynamics_kernels(
            wp_mass_matrix, wp_g_accel, wp_joint_parent,
            wp_joint_child, wp_body_vel, wp_body_vel_prev,
            wp_lambda_n, wp_lambda_t, wp_lambda_j,
            wp_body_f, wp_J_n, wp_J_t,
            wp_J_j_p, wp_J_j_c, dt, wp_output_res, device=wp_device
        )

        # Launch derivative kernels
        launch_dynamics_derivative_kernels(
             wp_mass_matrix, wp_J_n, wp_J_t, wp_joint_parent,
             wp_joint_child, wp_J_j_p, wp_J_j_c,
             wp_dres_dbody_vel, wp_dres_dlambda_n, wp_dres_dlambda_t, wp_dres_dlambda_j, device=wp_device
        )

    graph = capture.graph

    # Warm-up the graph
    wp.capture_launch(graph)
    torch.cuda.synchronize()

    # Graph execution timing
    start_time = time.time()
    for _ in range(num_runs):
        wp.capture_launch(graph)
    torch.cuda.synchronize()
    warp_graph_execution_time = (time.time() - start_time) / num_runs
    print(f"Warp with Graph (execution only) average time per call (Residual + Derivatives): {warp_graph_execution_time:.6f} seconds")
    print(f"Warp is {pytorch_time / warp_graph_execution_time:.2f}x faster than PyTorch for this test case.")


def run_all_tests():
    """Runs all the defined test cases and performance benchmarks."""
    wp.init() # WARP initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Comparison Tests (including derivatives)
    try:
        run_comparison_test("Test Case 1: No Contacts, No Joints", B=1, C=0, D=0, device=device)
    except ValueError as e:
        print(f"Skipping Test Case 1 due to configuration error: {e}")
    try:
        run_comparison_test("Test Case 2: With Single Contact, No Joints", B=1, C=1, D=0, device=device)
    except ValueError as e:
         print(f"Skipping Test Case 2 due to configuration error: {e}")
    try:
        run_comparison_test("Test Case 3: No Contacts, With Single Joint", B=2, C=0, D=1, device=device)
    except ValueError as e:
         print(f"Skipping Test Case 3 due to configuration error: {e}")
    try:
        run_comparison_test("Test Case 4: With Contacts and Joints", B=2, C=1, D=1, device=device)
    except ValueError as e:
        print(f"Skipping Test Case 4 due to configuration error: {e}")
    try:
        run_comparison_test("Test Case 5: Larger Scale", B=10, C=5, D=3, device=device)
    except ValueError as e:
        print(f"Skipping Test Case 5 due to configuration error: {e}")


    # Performance Benchmark (Residual + Derivatives)
    # Choose a representative scale for benchmarking
    try:
        run_performance_benchmark(B=20, C=36, D=10, device=device, num_runs=100)
    except ValueError as e:
        print(f"Skipping Performance Benchmark due to configuration error: {e}")


    print("\nAll tests finished.")


if __name__ == "__main__":
    run_all_tests()
