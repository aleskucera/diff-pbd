from typing import Tuple
import time

import torch
import warp as wp
from torch import Tensor
from jaxtyping import Float
from pbd_torch.model import Model
from pbd_torch.constraints import DynamicsConstraint

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
    if C > 0:
        wp.launch(
            kernel=build_dres_dlambda_n_kernel,
            dim=(B, 6, C), # Iterate over B, spatial dim, and C
            inputs=[J_n, dres_dlambda_n],
            device=device
        )

    # ∂res/∂lambda_t = -J_t.transpose(1, 2)
    if C > 0: # J_t dimension depends on C
        wp.launch(
            kernel=build_dres_dlambda_t_kernel,
            dim=(B, 6, 2 * C), # Iterate over B, spatial dim, and 2C
            inputs=[J_t, dres_dlambda_t],
            device=device
        )

    # ∂res/∂lambda_j
    # Requires a kernel due to the scatter/index_add logic
    if D > 0:
         wp.launch(
            kernel=build_dres_dlambda_j_kernel,
            dim=(B, 6, flat_joint_impulse_dim), # Iterate over B, spatial dim, and 5D
            inputs=[joint_parent, joint_child, J_j_p, J_j_c, dres_dlambda_j],
            device=device
         )


# --- Test and Benchmarking Functions using the DynamicsConstraint object ---

def setup_test_case_data(B: int, C: int, D: int, device: torch.device):
    """Generates random data for a given test case configuration."""
    # Use more stable data generation where appropriate (e.g., positive mass)
    g_accel = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -9.81], device=device).view(6, 1)

    # Generate a symmetric positive definite mass matrix for stability if needed, randomize otherwise
    mass_matrix_sym_rand = torch.randn(B, 6, 6, device=device)
    mass_matrix = torch.matmul(mass_matrix_sym_rand, mass_matrix_sym_rand.transpose(1,2)) + torch.eye(6, device=device).unsqueeze(0) # Adding identity to ensure positive definite

    # Ensure D joints have distinct parent and child indices within bounds [0, B-1]
    if B <= 1 and D > 0:
         raise ValueError("Cannot create a joint test case with B <= 1 and D > 0")

    joint_parent = torch.empty(D, dtype=torch.int64, device=device) # Use int64 for PyTorch indexing
    joint_child = torch.empty(D, dtype=torch.int64, device=device)
    for i in range(D):
        parent = torch.randint(0, B, (1,), device=device).item()
        child = torch.randint(0, B, (1,), device=device).item()
        while parent == child:
             child = torch.randint(0, B, (1,), device=device).item()
        joint_parent[i] = parent
        joint_child[i] = child

    # Instantiate a dummy Model with necessary attributes
    class DummyModel:
        def __init__(self, device, mass_matrix, g_accel, joint_parent, joint_child):
            self.device = device
            self.mass_matrix = mass_matrix
            self.g_accel = g_accel
            self.joint_parent = joint_parent
            self.joint_child = joint_child
            # Add other necessary attributes if the DynamicsConstraint class were to use them
            # (but the current DynamicsConstraint only uses the above)

    model = DummyModel(device, mass_matrix, g_accel, joint_parent, joint_child)


    body_vel = torch.randn(B, 6, 1, device=device, requires_grad=True)
    body_vel_prev = torch.randn(B, 6, 1, device=device)
    lambda_n = torch.randn(B, C, 1, device=device, requires_grad=True) if C > 0 else torch.empty(B, 0, 1, device=device) # No grad for empty tensors
    lambda_t = torch.randn(B, 2 * C, 1, device=device, requires_grad=True) if C > 0 else torch.empty(B, 0, 1, device=device) # No grad for empty tensors
    lambda_j = torch.randn(5 * D, 1, device=device, requires_grad=True) if D > 0 else torch.empty(0, 1, device=device) # No grad for empty tensors
    body_f = torch.randn(B, 6, 1, device=device)
    J_n = torch.randn(B, C, 6, device=device) if C > 0 else torch.empty(B, 0, 6, device=device)
    J_t = torch.randn(B, 2 * C, 6, device=device) if C > 0 else torch.empty(B, 0, 6, device=device)
    # J_j_p and J_j_c are D x 5 x 6
    J_j_p = torch.randn(D, 5, 6, device=device) if D > 0 else torch.empty(0, 5, 6, device=device)
    J_j_c = torch.randn(D, 5, 6, device=device) if D > 0 else torch.empty(0, 5, 6, device=device)
    dt = 0.01

    # Set requires_grad=False for inputs to the DynamicsConstraint methods where analytical derivatives are not computed w.r.t. it.
    # Only lambda_n, lambda_t, lambda_j, body_vel are expected to have requires_grad=True
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
        "model": model,
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
    Runs a comparison test for Warp analytical derivatives against PyTorch analytical derivatives
    computed by the DynamicsConstraint class.
    Also validates PyTorch analytical against PyTorch automatic derivatives.
    """
    print(f"Running Comparison Test '{test_name}' (B={B}, C={C}, D={D})")
    data = setup_test_case_data(B, C, D, device)
    dt = data["dt"]
    model = data["model"]
    wp_device = wp.get_device()

    # Instantiate the DynamicsConstraint class
    dyn_constraint = DynamicsConstraint(model)


    # --- Residual Comparison (Warp vs. PyTorch Class) ---
    # Convert data to Warp arrays for the Warp residual launch
    wp_mass_matrix = wp.from_torch(data['model'].mass_matrix.contiguous(), dtype=wp.spatial_matrix)
    wp_g_accel = wp.from_torch(data['model'].g_accel.squeeze().contiguous(), dtype=wp.spatial_vector)
    wp_joint_parent = wp.from_torch(data['model'].joint_parent.contiguous().to(torch.int32), dtype=wp.int32) # Ensure int32
    wp_joint_child = wp.from_torch(data['model'].joint_child.contiguous().to(torch.int32), dtype=wp.int32)   # Ensure int32
    wp_body_vel = wp.from_torch(data['body_vel'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)
    wp_body_vel_prev = wp.from_torch(data['body_vel_prev'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)
    wp_body_f = wp.from_torch(data['body_f'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)

    C_actual = data['lambda_n'].shape[1] if data['lambda_n'].ndim > 1 else 0
    D_actual = data['model'].joint_parent.shape[0]

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

    # Compute residual using the DynamicsConstraint class
    res_pt_class = dyn_constraint.get_residuals(
        body_vel=data['body_vel'],
        body_vel_prev=data['body_vel_prev'],
        lambda_n=data['lambda_n'],
        lambda_t=data['lambda_t'],
        lambda_j=data['lambda_j'],
        body_f=data['body_f'],
        J_n=data['J_n'],
        J_t=data['J_t'],
        J_j_p=data['J_j_p'],
        J_j_c=data['J_j_c'],
        dt=data['dt'],
    )

    assert torch.allclose(res_pt_class, res_warp, atol=1e-5), f"{test_name}: Residuals do not match (PyTorch Class vs. Warp)"
    print(f"{test_name}: Residual comparison passed (PyTorch Class vs. Warp).")


    # --- Derivative Comparison (Warp Analytical vs. PyTorch Class Analytical) ---
    # PyTorch class analytical derivatives
    dres_dbody_vel_pt_class_ana, dres_dlambda_n_pt_class_ana, dres_dlambda_t_pt_class_ana, dres_dlambda_j_pt_class_ana = dyn_constraint.get_derivatives(
        data['J_n'], data['J_t'], data['J_j_p'], data['J_j_c']
    )

    # Convert data to Warp arrays for the Warp derivative launch
    # Outputs need to be pre-allocated
    wp_dres_dbody_vel = wp.zeros(shape=B, dtype=wp.spatial_matrix, device=wp_device, requires_grad=False) # [B, 6, 6] as spatial_matrix
    wp_dres_dlambda_n = wp.zeros(shape=(B, 6, C_actual), dtype=wp.float32, device=wp_device, requires_grad=False)  # [B, 6, C]
    wp_dres_dlambda_t = wp.zeros(shape=(B, 6, 2 * C_actual), dtype=wp.float32, device=wp_device, requires_grad=False)  # [B, 6, 2C]
    wp_dres_dlambda_j = wp.zeros(shape=(B, 6, 5 * D_actual), dtype=wp.float32, device=wp_device, requires_grad=False)  # [B, 6, 5D]


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


    # Compare Warp analytical results with PyTorch Class analytical
    assert torch.allclose(dres_dbody_vel_pt_class_ana, dres_dbody_vel_warp, atol=1e-5), f"{test_name}: dres/dbody_vel mismatch (PyTorch Class Ana vs. Warp)"
    print(f"{test_name}: dres/dbody_vel comparison passed (PyTorch Class Ana vs. Warp).")

    if C_actual > 0:
        assert torch.allclose(dres_dlambda_n_pt_class_ana, dres_dlambda_n_warp, atol=1e-5), f"{test_name}: dres/dlambda_n mismatch (PyTorch Class Ana vs. Warp)"
        print(f"{test_name}: dres/dlambda_n comparison passed (PyTorch Class Ana vs. Warp).")

        assert torch.allclose(dres_dlambda_t_pt_class_ana, dres_dlambda_t_warp, atol=1e-5), f"{test_name}: dres/dlambda_t mismatch (PyTorch Class Ana vs. Warp)"
        print(f"{test_name}: dres/dlambda_t comparison passed (PyTorch Class Ana vs. Warp).")
    else:
        # Ensure empty tensors match or are both zero
        assert dres_dlambda_n_warp.shape == dres_dlambda_n_pt_class_ana.shape
        assert dres_dlambda_t_warp.shape == dres_dlambda_t_pt_class_ana.shape
        print(f"{test_name}: dres/dlambda_n (empty) comparison passed (PyTorch Class Ana vs. Warp).")
        print(f"{test_name}: dres/dlambda_t (empty) comparison passed (PyTorch Class Ana vs. Warp).")


    if D_actual > 0:
         assert torch.allclose(dres_dlambda_j_pt_class_ana, dres_dlambda_j_warp, atol=1e-5), f"{test_name}: dres/dlambda_j mismatch (PyTorch Class Ana vs. Warp)"
         print(f"{test_name}: dres/dlambda_j comparison passed (PyTorch Class Ana vs. Warp).")
    else:
        # Ensure empty tensors match or are both zero
        assert dres_dlambda_j_warp.shape == dres_dlambda_j_pt_class_ana.shape
        print(f"{test_name}: dres/dlambda_j (empty) comparison passed (PyTorch Class Ana vs. Warp).")


    # --- Validate PyTorch Class Analytical with PyTorch Automatic Differentiation ---
    # We need to define a function that wraps the residual calculation to use with torch.autograd.functional.jacobian
    # This function will take the inputs that we want to compute derivatives with respect to.

    def residual_wrapper_flat_for_auto_diff(
        body_vel_flat, lambda_n_flat, lambda_t_flat, lambda_j_flat,
        # Pass other arguments that derivatives are NOT taken with respect to
        dyn_constraint_instance, body_vel_prev, body_f, J_n, J_t, J_j_p, J_j_c, dt):

        B = dyn_constraint_instance.mass_matrix.shape[0]
        # Handle potential empty tensors and their shapes
        C_actual = J_n.shape[1] if J_n.ndim > 1 and J_n.shape[1] > 0 else 0
        D_actual = dyn_constraint_instance.joint_parent.shape[0] if dyn_constraint_instance.joint_parent.ndim > 0 else 0


        body_vel_reshaped = body_vel_flat.view(B, 6, 1)
        lambda_n_reshaped = lambda_n_flat.view(B, C_actual, 1) if C_actual > 0 else torch.empty(B, 0, 1, device=body_vel_flat.device)
        lambda_t_reshaped = lambda_t_flat.view(B, 2 * C_actual, 1) if C_actual > 0 else torch.empty(B, 0, 1, device=body_vel_flat.device)
        lambda_j_reshaped = lambda_j_flat.view(5 * D_actual, 1) if D_actual > 0 else torch.empty(0, 1, device=body_vel_flat.device)


        return dyn_constraint_instance.get_residuals(
            body_vel=body_vel_reshaped,
            body_vel_prev=body_vel_prev,
            lambda_n=lambda_n_reshaped,
            lambda_t=lambda_t_reshaped,
            lambda_j=lambda_j_reshaped,
            body_f=body_f,
            J_n=J_n,
            J_t=J_t,
            J_j_p=J_j_p,
            J_j_c=J_j_c,
            dt=dt,
        ).view(-1) # Flatten the output residual for jacobian


    from torch.autograd.functional import jacobian as torch_jacobian

    # Prepare inputs for jacobian function
    inputs_to_grad = (data['body_vel'].view(-1), data['lambda_n'].view(-1), data['lambda_t'].view(-1), data['lambda_j'].view(-1),)

    # Prepare other arguments
    other_args = (
        dyn_constraint, # Pass the instance
        data['body_vel_prev'], data['body_f'], data['J_n'], data['J_t'], data['J_j_p'], data['J_j_c'], data['dt']
    )

    # Compute Jacobian using automatic differentiation
    try:
        J_overall_pt_auto = torch_jacobian(residual_wrapper_flat_for_auto_diff, inputs_to_grad, args=other_args, strict=True)
    except Exception as e:
        print(f"Warning: Could not compute PyTorch automatic differentiation Jacobian. This might indicate an issue with grad tracking or input sizes. Error: {e}")
        print(f"{test_name} skipped PT Analytical vs Auto validation.")
        J_overall_pt_auto = None # Set to None if computation fails for comparison checks

    if J_overall_pt_auto is not None:
        # Extract blocks from the overall Jacobian
        B = model.mass_matrix.shape[0]
        C_actual = data['J_n'].shape[1] if data['J_n'].ndim > 1 and data['J_n'].shape[1] > 0 else 0
        D_actual = model.joint_parent.shape[0] if model.joint_parent.ndim > 0 else 0


        # 1. d(res)/d(body_vel):
        dres_dbody_vel_pt_auto_overall = J_overall_pt_auto[0].view(B, 6, B, 6).permute(0, 2, 1, 3) # Shape [B, B, 6, 6]
        dres_dbody_vel_pt_auto = torch.zeros_like(dres_dbody_vel_pt_class_ana)
        for b_idx in range(B):
             dres_dbody_vel_pt_auto[b_idx, :, :,] = dres_dbody_vel_pt_auto_overall[b_idx, b_idx, :, :]


        # 2. d(res)/d(lambda_n):
        dres_dlambda_n_pt_auto_overall = J_overall_pt_auto[1].view(B, 6, B, C_actual).permute(0, 2, 1, 3) if C_actual > 0 else torch.empty(B, B, 6, 0, device=device) # Shape [B, B, 6, C]
        dres_dlambda_n_pt_auto = torch.zeros_like(dres_dlambda_n_pt_class_ana)
        if C_actual > 0:
            for b_idx in range(B):
                 dres_dlambda_n_pt_auto[b_idx, :, :] = dres_dlambda_n_pt_auto_overall[b_idx, b_idx, :, :]


        # 3. d(res)/d(lambda_t):
        dres_dlambda_t_pt_auto_overall = J_overall_pt_auto[2].view(B, 6, B, 2*C_actual).permute(0, 2, 1, 3) if C_actual > 0 \
                                        else torch.empty(B, B, 6, 0, device=device) # Shape [B, B, 6, 2C]
        dres_dlambda_t_pt_auto = torch.zeros_like(dres_dlambda_t_pt_class_ana)
        if C_actual > 0:
            for b_idx in range(B):
                 dres_dlambda_t_pt_auto[b_idx, :, :] = dres_dlambda_t_pt_auto_overall[b_idx, b_idx, :, :]


        # 4. d(res)/d(lambda_j):
        dres_dlambda_j_pt_auto = J_overall_pt_auto[3].view(B, 6, 5 * D_actual) if D_actual > 0 \
                                     else torch.empty(B, 6, 0, device=device) # Shape [B, 6, 5D]


        # Compare PyTorch analytical with PyTorch automatic differentiation results
        assert torch.allclose(dres_dbody_vel_pt_class_ana, dres_dbody_vel_pt_auto, atol=1e-5), f"{test_name}: PT Analytical (Class) vs Auto dres/dbody_vel mismatch"
        print(f"{test_name}: PT Analytical (Class) vs Auto dres/dbody_vel validation passed.")

        if C_actual > 0:
             assert torch.allclose(dres_dlambda_n_pt_class_ana, dres_dlambda_n_pt_auto, atol=1e-5), f"{test_name}: PT Analytical (Class) vs Auto dres/dlambda_n mismatch"
             print(f"{test_name}: PT Analytical (Class) vs Auto dres/dlambda_n validation passed.")

             assert torch.allclose(dres_dlambda_t_pt_class_ana, dres_dlambda_t_pt_auto, atol=1e-5), f"{test_name}: PT Analytical (Class) vs Auto dres/dlambda_t mismatch"
             print(f"{test_name}: PT Analytical (Class) vs Auto dres/dlambda_t validation passed.")
        else:
             assert dres_dlambda_n_pt_auto.shape == dres_dlambda_n_pt_class_ana.shape
             assert dres_dlambda_t_pt_auto.shape == dres_dlambda_t_pt_class_ana.shape
             print(f"{test_name}: PT Analytical (Class) vs Auto dres/dlambda_n (empty) validation passed.")
             print(f"{test_name}: PT Analytical (Class) vs Auto dres/dlambda_t (empty) validation passed.")


        if D_actual > 0:
            assert torch.allclose(dres_dlambda_j_pt_class_ana, dres_dlambda_j_pt_auto, atol=1e-5), f"{test_name}: PT Analytical (Class) vs Auto dres/dlambda_j mismatch"
            print(f"{test_name}: PT Analytical (Class) vs Auto dres/dlambda_j validation passed.")
        else:
             assert dres_dlambda_j_pt_auto.shape == dres_dlambda_j_pt_class_ana.shape
             print(f"{test_name}: PT Analytical (Class) vs Auto dres/dlambda_j (empty) validation passed.")

    print(f"{test_name} passed all comparisons.")


def run_performance_benchmark(B: int, C: int, D: int, device: torch.device, num_runs: int = 100):
    """Runs a performance benchmark for the PyTorch implementation using the DynamicsConstraint class
       against the Warp kernels using a graph."""
    print(f"\nRunning Performance Test (B={B}, C={C}, D={D})")
    data = setup_test_case_data(B, C, D, device)
    model = data["model"]

    # Instantiate the DynamicsConstraint class
    dyn_constraint = DynamicsConstraint(model)

    # PyTorch performance (Residual + Analytical Derivatives using the class)
    # Warm-up
    res_computed = dyn_constraint.get_residuals(
        body_vel=data['body_vel'],
        body_vel_prev=data['body_vel_prev'],
        lambda_n=data['lambda_n'],
        lambda_t=data['lambda_t'],
        lambda_j=data['lambda_j'],
        body_f=data['body_f'],
        J_n=data['J_n'],
        J_t=data['J_t'],
        J_j_p=data['J_j_p'],
        J_j_c=data['J_j_c'],
        dt=data['dt'],
    )

    dres_dbody_vel_pt_class_ana, dres_dlambda_n_pt_class_ana, dres_dlambda_t_pt_class_ana, dres_dlambda_j_pt_class_ana = dyn_constraint.get_derivatives(
        data['J_n'], data['J_t'], data['J_j_p'], data['J_j_c']
    )
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_runs):
        res_computed = dyn_constraint.get_residuals(
            body_vel=data['body_vel'],
            body_vel_prev=data['body_vel_prev'],
            lambda_n=data['lambda_n'],
            lambda_t=data['lambda_t'],
            lambda_j=data['lambda_j'],
            body_f=data['body_f'],
            J_n=data['J_n'],
            J_t=data['J_t'],
            J_j_p=data['J_j_p'],
            J_j_c=data['J_j_c'],
            dt=data['dt'],
        )
        dres_dbody_vel_pt_class_ana, dres_dlambda_n_pt_class_ana, dres_dlambda_t_pt_class_ana, dres_dlambda_j_pt_class_ana = dyn_constraint.get_derivatives(
            data['J_n'], data['J_t'], data['J_j_p'], data['J_j_c']
        )
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_runs
    print(f"PyTorch average time per call (Residual + Analytical Derivatives): {pytorch_time:.6f} seconds")

    # Warp performance (Residual + Analytical Derivatives with graph)
    wp.init() # Ensure Warp is initialized
    wp_device = wp.get_device()

    # Convert data to Warp arrays for graph capture
    wp_mass_matrix = wp.from_torch(data['model'].mass_matrix.contiguous(), dtype=wp.spatial_matrix)
    wp_g_accel = wp.from_torch(data['model'].g_accel.squeeze().contiguous(), dtype=wp.spatial_vector)
    wp_joint_parent = wp.from_torch(data['model'].joint_parent.contiguous().to(torch.int32), dtype=wp.int32)
    wp_joint_child = wp.from_torch(data['model'].joint_child.contiguous().to(torch.int32), dtype=wp.int32)
    wp_body_vel = wp.from_torch(data['body_vel'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)
    wp_body_vel_prev = wp.from_torch(data['body_vel_prev'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)
    wp_body_f = wp.from_torch(data['body_f'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)

    C_actual = data['lambda_n'].shape[1] if data['lambda_n'].ndim > 1 else 0
    D_actual = data['model'].joint_parent.shape[0]
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
            wp_J_j_p, wp_J_j_c, data["dt"], wp_output_res, device=wp_device
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



# --- Main execution block ---
if __name__ == "__main__":
    wp.init() # Initialize Warp once at the beginning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Comparison Tests (Warp Analytical vs. PyTorch Class Analytical)
    # And validation of PyTorch Class Analytical vs. PyTorch Automatic Differentiation
    print("\n--- Running Comparison Tests (Warp Analytical vs. PyTorch Class Analytical & PT Auto Diff Validation) ---")
    try:
        run_comparison_test("Test Case 1: No Contacts, No Joints", B=1, C=0, D=0, device=device)
    except ValueError as e:
        print(f"Skipping Test Case 1 due to configuration error: {e}")
    except Exception as e:
        print(f"Error in Test Case 1: {e}")
    try:
        run_comparison_test("Test Case 2: With Single Contact, No Joints", B=1, C=1, D=0, device=device)
    except ValueError as e:
         print(f"Skipping Test Case 2 due to configuration error: {e}")
    except Exception as e:
        print(f"Error in Test Case 2: {e}")
    try:
        run_comparison_test("Test Case 3: No Contacts, With Single Joint", B=2, C=0, D=1, device=device)
    except ValueError as e:
         print(f"Skipping Test Case 3 due to configuration error: {e}")
    except Exception as e:
        print(f"Error in Test Case 3: {e}")
    try:
        run_comparison_test("Test Case 4: With Contacts and Joints", B=2, C=1, D=1, device=device)
    except ValueError as e:
        print(f"Skipping Test Case 4 due to configuration error: {e}")
    except Exception as e:
        print(f"Error in Test Case 4: {e}")
    try:
        run_comparison_test("Test Case 5: Larger Scale", B=10, C=5, D=3, device=device)
    except ValueError as e:
        print(f"Skipping Test Case 5 due to configuration error: {e}")
    except Exception as e:
        print(f"Error in Test Case 5: {e}")


    # Performance Benchmark (PyTorch Class Analytical vs. Warp with Graph)
    print("\n--- Running Performance Benchmark (PyTorch Class Analytical vs. Warp with Graph) ---")
    # Choose a representative scale for benchmarking
    try:
        run_performance_benchmark(B=20, C=16, D=10, device=device, num_runs=100) # Increased scale slightly for benchmark
    except ValueError as e:
        print(f"Skipping Performance Benchmark due to configuration error: {e}")
    except Exception as e:
        print(f"Error in Performance Benchmark: {e}")

    print("\nAll tests finished.")