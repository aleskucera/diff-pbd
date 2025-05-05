# contact_constraint_warp.py
import time
from typing import Tuple

import torch
import warp as wp
from jaxtyping import Float, Bool
from pbd_torch.ncp import ScaledFisherBurmeister
from pbd_torch.constraints import ContactConstraint
from pbd_torch.utils import swap_quaternion_real_part

@wp.func
def scaled_fisher_burmeister_evaluate(
    a: wp.float32,
    b: wp.float32,
    alpha: wp.float32,
    beta: wp.float32,
    epsilon: wp.float32
) -> wp.float32:
    """Warp kernel-side evaluation of the Scaled Fisher-Burmeister function."""
    scaled_a = alpha * a
    scaled_b = beta * b
    norm = wp.sqrt(scaled_a**2.0 + scaled_b**2.0 + epsilon)
    return scaled_a + scaled_b - norm

@wp.func
def scaled_fisher_burmeister_derivatives(
    a: wp.float32,
    b: wp.float32,
    alpha: wp.float32,
    beta: wp.float32,
    epsilon: wp.float32
) -> tuple[wp.float32, wp.float32]:
    """Warp kernel-side derivatives of the Scaled Fisher-Burmeister function."""
    scaled_a = alpha * a
    scaled_b = beta * b
    norm = wp.sqrt(scaled_a**2.0 + scaled_b**2.0 + epsilon)

    da = alpha * (1.0 - scaled_a / norm)
    db = beta * (1.0 - scaled_b / norm)

    return da, db


# --- Warp Kernels for Contact Constraint ---

@wp.kernel
def compute_contact_jacobians_kernel(
    body_trans: wp.array(dtype=wp.transform),  # [B] (position + quat(xyzw))
    contact_points_local: wp.array(dtype=wp.vec3, ndim=2), # [B, C] local contact points
    contact_normals_world: wp.array(dtype=wp.vec3, ndim=2),# [B, C] world contact normals
    contact_mask: wp.array(dtype=wp.float32, ndim=2),      # [B, C] float mask (1.0 or 0.0)
    J_n: wp.array(dtype=wp.spatial_vector, ndim=2)       # Output: [B, C] spatial_vectors (J_n)
):
    b, c = wp.tid() # Body index [0, B), contact index [0, C)

    mask = contact_mask[b, c] # Get the float mask value

    # Get body rotation (xyzw)
    body_rot = wp.transform_get_rotation(body_trans[b])

    # Rotate local contact point to world frame
    r = wp.quat_rotate(body_rot, contact_points_local[b, c])

    # Compute r × n for the rotational component of the Jacobian
    # Multiply the result by the `mask`
    r_cross_n = wp.cross(r, contact_normals_world[b, c])

    # Normal Jacobian J_n = [r x n, n]
    J_n[b, c] = wp.spatial_vector(
        r_cross_n * mask,
        contact_normals_world[b, c] * mask,
    )


@wp.kernel
def compute_penetration_depths_kernel(
    body_trans: wp.array(dtype=wp.transform),  # [B]
    contact_points_local: wp.array(dtype=wp.vec3, ndim=2), # [B, C] local contact points
    ground_points_world: wp.array(dtype=wp.vec3, ndim=2), # [B, C] world ground points
    contact_normals_world: wp.array(dtype=wp.vec3, ndim=2),# [B, C] world contact normals
    penetration_depth: wp.array(dtype=wp.float32, ndim=2) # Output: [B, C]
):
     b, c = wp.tid() # Body index [0, B), contact index [0, C)

     # Transform local contact point to world frame
     body_transform = body_trans[b]
     point_world = wp.transform_point(body_transform, contact_points_local[b, c])

     # Calculate penetration depth as projection of (ground_point - contact_point) onto normal
     diff = ground_points_world[b, c] - point_world
     depth = wp.dot(diff, contact_normals_world[b, c])

     penetration_depth[b, c] = depth


@wp.kernel
def get_contact_residuals_kernel(
    body_vel: wp.array(dtype=wp.spatial_vector), # [B]
    body_vel_prev: wp.array(dtype=wp.spatial_vector), # [B]
    lambda_n: wp.array(dtype=wp.float32, ndim=2), # [B, C] normal impulses
    J_n: wp.array(dtype=wp.spatial_vector, ndim=2),  # [B, C] spatial_vectors (J_n)
    penetration_depth: wp.array(dtype=wp.float32, ndim=2), # [B, C]
    contact_mask: wp.array(dtype=wp.float32, ndim=2), # [B, C] float mask (1.0 or 0.0)
    contact_weight: wp.array(dtype=wp.float32, ndim=2), # [B, C]
    restitution: wp.array(dtype=wp.float32, ndim=1), # [B]
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    fb_epsilon: wp.float32,
    res: wp.array(dtype=wp.float32, ndim=2)  # Output: [B, C]
):
    b, c = wp.tid() # Body index [0, B), contact index [0, C)

    active_mask = contact_mask[b, c] # Get the float mask value
    inactive_mask = 1.0 - active_mask
    weight = contact_weight[b, c]

    v_n = wp.dot(J_n[b, c], body_vel[b])
    v_n_prev = wp.dot(J_n[b, c], body_vel_prev[b])
    e = restitution[b]

    b_err = -(stabilization_factor / dt) * penetration_depth[b, c]
    b_rest = e * v_n_prev

    # Active constraint residual contribution (weighted)
    # This is phi(lambda_n, v_n + bias) * weight
    res_act_weighted = scaled_fisher_burmeister_evaluate(
        lambda_n[b, c],
        v_n + b_err + b_rest,
        fb_alpha, fb_beta, fb_epsilon
    ) * weight

    # Inactive constraint residual contribution
    res_inact_weighted = -lambda_n[b, c]

    # Combine contributions using the mask
    res[b, c] = res_act_weighted * active_mask + res_inact_weighted * inactive_mask


@wp.kernel
def get_contact_derivatives_kernel(
    body_vel: wp.array(dtype=wp.spatial_vector), # [B]
    body_vel_prev: wp.array(dtype=wp.spatial_vector), # [B]
    lambda_n: wp.array(dtype=wp.float32, ndim=2), # [B, C] normal impulses
    J_n: wp.array(dtype=wp.spatial_vector, ndim=2),  # [B, C] spatial_vectors (J_n)
    penetration_depth: wp.array(dtype=wp.float32, ndim=2), # [B, C]
    contact_mask: wp.array(dtype=wp.float32, ndim=2), # [B, C] float mask (1.0 or 0.0)
    contact_weight: wp.array(dtype=wp.float32, ndim=2), # [B, C]
    restitution: wp.array(dtype=wp.float32, ndim=1), # [B]
    dt: wp.float32,
    stabilization_factor: wp.float32,
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    fb_epsilon: wp.float32,
    dres_dbody_vel: wp.array(dtype=wp.float32, ndim=3), # Output: [B, C, 6]
    dres_dlambda_n: wp.array(dtype=wp.float32, ndim=2)  # Output: [B, C] (diagonal of ∂res/∂λ_n)
):
    b, c = wp.tid() # Body index [0, B), contact index [0, C)

    active_mask = contact_mask[b, c] # Get the float mask value
    inactive_mask = 1.0 - active_mask
    weight = contact_weight[b, c]

    # Compute terms needed for derivatives regardless of mask state
    v_n = wp.dot(J_n[b, c], body_vel[b])
    v_n_prev = wp.dot(J_n[b, c], body_vel_prev[b])
    e = restitution[b]

    b_err = -(stabilization_factor / dt) * penetration_depth[b, c]
    b_rest = e * v_n_prev

    # Derivatives for active constraint
    da_act, db_act = scaled_fisher_burmeister_derivatives(
        lambda_n[b, c],
        v_n + b_err + b_rest,
        fb_alpha, fb_beta, fb_epsilon
    )

    # Derivatives for inactive constraint
    # ∂(-λ_n)/∂λ_n = -1
    # ∂(-λ_n)/∂vel = 0
    da_inact = -1.0
    db_inact = 0.0

    # Combine derivatives using the mask
    # da combines based on whether the constraint is active or inactive
    da = da_act * active_mask * weight + da_inact * inactive_mask

    # db combines based on whether the constraint is active or inactive
    db = db_act * active_mask * weight + db_inact * inactive_mask

    # Store the derivatives for body_vel (reshaped from [B, C, 6])
    # Apply weight and db (which now handles active/inactive)
    for j in range(wp.static(6)):
        # ∂res_bc / ∂body_vel_b_j
        # For active: db_act * J_n_bc_j * weight
        # For inactive: db_inact * J_n_bc_j * weight = 0.0 * J_n_bc_j * weight = 0.0
        # Multiplying (db * J_n[b, c][j]) by weight gives the correct weighted derivative for both cases
        dres_dbody_vel[b, c, wp.static(j)] = db * J_n[b, c][wp.static(j)]

    # Store the diagonal terms of ∂res/∂lambda_n
    # If active: da_act * weight
    # If inactive: da_inact * weight = -1.0 * weight
    # Multiplying `da` (which now handles active/inactive) by weight gives the correct weighted diagonal derivative
    dres_dlambda_n[b, c] = da

# --- Warp Functions (Wrappers for Kernels) ---
def compute_contact_jacobians_warp(
    body_trans: wp.array, # [B] wp.transform (pos + quat xyzw)
    contact_points_local: wp.array, # [B, C] wp.vec3
    contact_normals_world: wp.array, # [B, C] wp.vec3
    contact_mask: wp.array, # [B, C] wp.uint8
    device: wp.context.Device
) -> wp.array: # [B, C] wp.spatial_vector
    B = body_trans.shape[0]
    C = contact_mask.shape[1] if len(contact_mask.shape) > 1 else 0 # Handle C = 0 case
    if C == 0:
        return wp.empty(shape=(B, 0), dtype=wp.spatial_vector, device=device) # Return empty if no contacts

    J_n = wp.zeros(shape=(B, C), dtype=wp.spatial_vector, device=device)
    wp.launch(
        kernel=compute_contact_jacobians_kernel,
        dim=(B, C),
        inputs=[
            body_trans, contact_points_local, contact_normals_world,
            contact_mask, J_n
        ],
        device=device
    )
    return J_n

def get_penetration_depths_warp(
     body_trans: wp.array, # [B] wp.transform (pos + quat xyzw)
     contact_points_local: wp.array, # [B, C] wp.vec3
     ground_points_world: wp.array, # [B, C] wp.vec3
     contact_normals_world: wp.array, # [B, C] wp.vec3
     device: wp.context.Device
) -> wp.array: # [B, C] wp.float32
    B = body_trans.shape[0]
    C = contact_points_local.shape[1] if len(contact_points_local.shape) > 1 else 0 # Handle C = 0 case
    if C == 0:
        return wp.empty(shape=(B, 0), dtype=wp.float32, device=device) # Return empty if no contacts

    penetration_depth = wp.zeros(shape=(B, C), dtype=wp.float32, device=device)
    wp.launch(
        kernel=compute_penetration_depths_kernel,
        dim=(B, C),
        inputs=[
            body_trans, contact_points_local, ground_points_world,
            contact_normals_world, penetration_depth
        ],
        device=device
    )
    return penetration_depth


def get_contact_residuals_warp(
    body_vel: wp.array, # [B] wp.spatial_vector
    body_vel_prev: wp.array, # [B] wp.spatial_vector
    lambda_n: wp.array, # [B, C] wp.float32
    J_n: wp.array, # [B, C] wp.spatial_vector
    penetration_depth: wp.array, # [B, C] wp.float32
    contact_mask: wp.array, # [B, C] wp.uint8
    contact_weight: wp.array, # [B, C] wp.float32
    restitution: wp.array, # [B] wp.float32
    dt: float,
    stabilization_factor: float,
    fb_alpha: float,
    fb_beta: float,
    fb_epsilon: float,
    device: wp.context.Device
) -> wp.array: # [B, C] wp.float32
    B = body_vel.shape[0]
    C = lambda_n.shape[1] if len(lambda_n.shape) > 1 else 0 # Handle C = 0 case
    if C == 0:
        return wp.empty(shape=(B, 0), dtype=wp.float32, device=device) # Return empty if no contacts

    res = wp.zeros(shape=(B, C), dtype=wp.float32, device=device)
    wp.launch(
        kernel=get_contact_residuals_kernel,
        dim=(B, C),
        inputs=[
            body_vel, body_vel_prev, lambda_n, J_n, penetration_depth,
            contact_mask, contact_weight, restitution, dt, stabilization_factor,
            fb_alpha, fb_beta, fb_epsilon, res
        ],
        device=device
    )
    return res


def get_contact_derivatives_warp(
    body_vel: wp.array, # [B] wp.spatial_vector
    body_vel_prev: wp.array, # [B] wp.spatial_vector
    lambda_n: wp.array, # [B, C] wp.float32
    J_n: wp.array, # [B, C] wp.spatial_vector
    penetration_depth: wp.array, # [B, C] wp.float32
    contact_mask: wp.array, # [B, C] wp.uint8
    contact_weight: wp.array, # [B, C] wp.float32
    restitution: wp.array, # [B] wp.float32
    dt: float,
    stabilization_factor: float,
    fb_alpha: float,
    fb_beta: float,
    fb_epsilon: float,
    device: wp.context.Device
) -> Tuple[wp.array, wp.array]: # ∂res/∂body_vel [B, C, 6], ∂res/∂lambda_n [B, C] (diagonal)
    B = body_vel.shape[0]
    C = lambda_n.shape[1] if len(lambda_n.shape) > 1 else 0 # Handle C = 0 case
    if C == 0:
         return (wp.empty(shape=(B, 0, 6), dtype=wp.float32, device=device),
                 wp.empty(shape=(B, 0), dtype=wp.float32, device=device)) # Return empty if no contacts


    dres_dbody_vel = wp.zeros(shape=(B, C, 6), dtype=wp.float32, device=device)
    dres_dlambda_n = wp.zeros(shape=(B, C), dtype=wp.float32, device=device) # Diagonal terms

    wp.launch(
        kernel=get_contact_derivatives_kernel,
        dim=(B, C),
        inputs=[
            body_vel, body_vel_prev, lambda_n, J_n, penetration_depth,
            contact_mask, contact_weight, restitution, dt, stabilization_factor,
            fb_alpha, fb_beta, fb_epsilon, dres_dbody_vel, dres_dlambda_n
        ],
        device=device
    )
    return dres_dbody_vel, dres_dlambda_n

# --- Helper for Quaternion format conversion (WXYZ to XYZW for Warp transform) ---
# This is already present in joint_constraint.py, assuming it's in a common utility file
# from joint_constraint import swap_quaternion_real_part

# --- Test Data Setup ---
def setup_contact_test_case_data(B: int, C: int, device: torch.device):
    """Generates random data for a contact constraint test case."""
    if C > 0 and B < 1:
        raise ValueError("Cannot create contacts with B < 1 if C > 0")

    # Random body transforms (pos + quat [w, x, y, z])
    body_trans = torch.randn(B, 7, 1, device=device) if B > 0 else torch.empty(0, 7, 1, device=device)
    if B > 0:
         # Normalize body_trans quaternions (wxyz)
         body_trans_quats = body_trans[:, 3:, 0] # Shape [B, 4]
         body_trans_quat_norms = torch.linalg.norm(body_trans_quats, dim=-1, keepdim=True) # Shape [B, 1]
         body_trans_quat_norms[body_trans_quat_norms == 0] = 1.0
         body_trans[:, 3:, 0] = body_trans_quats / body_trans_quat_norms
         body_trans[:, 3:, :] = body_trans[:, 3:, 0].unsqueeze(-1) # Restore shape

    # Random local contact points, world ground points, world contact normals
    contact_points_local = torch.randn(B, C, 3, 1, device=device) if C > 0 else torch.empty(B, 0, 3, 1, device=device)
    ground_points_world = torch.randn(B, C, 3, 1, device=device) if C > 0 else torch.empty(B, 0, 3, 1, device=device)
    contact_normals_world = torch.randn(B, C, 3, 1, device=device) if C > 0 else torch.empty(B, 0, 3, 1, device=device)

    if C > 0:
        # Normalize contact normals
        contact_normal_norms = torch.linalg.norm(contact_normals_world, dim=2, keepdim=True) # Shape [B, C, 1, 1]
        contact_normal_norms[contact_normal_norms == 0] = 1.0
        contact_normals_world = contact_normals_world / contact_normal_norms

    # Random body velocities, normal impulses, contact mask, weight, restitution
    body_vel = torch.randn(B, 6, 1, device=device, requires_grad=True) if B > 0 else torch.empty(0, 6, 1, device=device)
    body_vel_prev = torch.randn(B, 6, 1, device=device) if B > 0 else torch.empty(0, 6, 1, device=device)
    lambda_n = torch.randn(B, C, 1, device=device, requires_grad=True) if C > 0 else torch.empty(B, 0, 1, device=device)
    contact_mask = (torch.rand(B, C, device=device) > 0.2).bool() if C > 0 else torch.empty(B, 0, device=device, dtype=torch.bool)
    contact_weight = torch.rand(B, C, device=device) + 0.1 if C > 0 else torch.empty(B, 0, device=device) # Add 0.1 to avoid zero weights
    restitution = torch.rand(B, 1, device=device) if B > 0 else torch.empty(0, 1, device=device)
    dt = 0.01

    # Scaled Fisher-Burmeister parameters (matching the original torch class)
    fb_alpha = 0.3
    fb_beta = 0.3
    fb_epsilon = 1e-12
    stabilization_factor = 0.2

    # Set requires_grad=False for inputs that analytical derivatives are not taken w.r.t.
    body_trans.requires_grad_(False)
    contact_points_local.requires_grad_(False)
    ground_points_world.requires_grad_(False)
    contact_normals_world.requires_grad_(False)
    body_vel_prev.requires_grad_(False)
    contact_mask.requires_grad_(False)
    contact_weight.requires_grad_(False)
    restitution.requires_grad_(False)
    # dt, fb_alpha, fb_beta, fb_epsilon are scalars, no grad needed

    return {
        "body_trans": body_trans,
        "contact_points_local": contact_points_local,
        "ground_points_world": ground_points_world,
        "contact_normals_world": contact_normals_world,
        "body_vel": body_vel,
        "body_vel_prev": body_vel_prev,
        "lambda_n": lambda_n,
        "contact_mask": contact_mask,
        "contact_weight": contact_weight,
        "restitution": restitution,
        "dt": dt,
        "fb_alpha": fb_alpha,
        "fb_beta": fb_beta,
        "fb_epsilon": fb_epsilon,
        "stabilization_factor": stabilization_factor,
    }

# --- Comparison Test ---
def run_contact_comparison_test(test_name: str, B: int, C: int, device: torch.device):
    print(f"\nRunning Contact Constraint Comparison Test '{test_name}' (B={B}, C={C})")
    try:
        data = setup_contact_test_case_data(B, C, device)
    except ValueError as e:
        print(f"Skipping test due to setup error: {e}")
        return

    dt = data["dt"]
    wp_device = wp.get_device()

    # Need to adjust PyTorch input shapes for some functions
    body_trans_pt_in = data["body_trans"]
    contact_points_pt_in = data["contact_points_local"]
    ground_points_pt_in = data["ground_points_world"]
    contact_normals_pt_in = data["contact_normals_world"]
    body_vel_pt_in = data["body_vel"]
    body_vel_prev_pt_in = data["body_vel_prev"]
    lambda_n_pt_in = data["lambda_n"]
    contact_mask_pt_in = data["contact_mask"]
    contact_weight_pt_in = data["contact_weight"]
    restitution_pt_in = data["restitution"]
    fb_alpha = data["fb_alpha"]
    fb_beta = data["fb_beta"]
    fb_epsilon = data["fb_epsilon"]
    stabilization_factor = data["stabilization_factor"]

    # --- PyTorch computations ---
    # Instantiate the original PyTorch class with dummy device (device is passed later)
    constraint_pt = ContactConstraint(device=device,
                                      fb_alpha=fb_alpha,
                                      fb_beta=fb_beta,
                                      fb_epsilon=fb_epsilon,
                                      stabilization_factor=stabilization_factor)


    if C > 0:
        # Penetration depths
        penetration_depth_pt = constraint_pt.get_penetration_depths(
            body_trans_pt_in, contact_points_pt_in, ground_points_pt_in, contact_normals_pt_in
        )

        # Jacobians
        J_n_pt = constraint_pt.compute_contact_jacobians(
            body_trans_pt_in, contact_points_pt_in, contact_normals_pt_in, contact_mask_pt_in
        ) # [B, C, 6]

        # Residuals
        res_pt = constraint_pt.get_residuals(
            body_vel_pt_in, body_vel_prev_pt_in, lambda_n_pt_in, J_n_pt,
            penetration_depth_pt, contact_mask_pt_in, contact_weight_pt_in,
            restitution_pt_in, dt
        ) # [B, C, 1]

        # Derivatives
        dres_dbody_vel_pt, dres_dlambda_n_pt_diag = constraint_pt.get_derivatives(
             body_vel_pt_in, body_vel_prev_pt_in, lambda_n_pt_in, J_n_pt,
             penetration_depth_pt, contact_mask_pt_in, contact_weight_pt_in,
             restitution_pt_in, dt
        ) # dres_dbody_vel_pt [B, C, 6], dres_dlambda_n_pt_diag [B, C, C] (diagonal matrix)
    else:
        # Handle C=0 case for PyTorch outputs
        penetration_depth_pt = torch.empty(B, 0, 1, device=device)
        J_n_pt = torch.empty(B, 0, 6, device=device)
        res_pt = torch.empty(B, 0, 1, device=device)
        dres_dbody_vel_pt = torch.empty(B, 0, 6, device=device)
        dres_dlambda_n_pt_diag = torch.empty(B, 0, 0, device=device)


    # --- Warp computations ---
    # Convert data to Warp arrays
    # Quaternions need to be XYZW for wp.transform
    wp_body_trans = wp.from_torch(swap_quaternion_real_part(data['body_trans']).squeeze(-1).contiguous(), dtype=wp.transform)
    wp_contact_points_local = wp.from_torch(data['contact_points_local'].squeeze(-1).contiguous(), dtype=wp.vec3)
    wp_ground_points_world = wp.from_torch(data['ground_points_world'].squeeze(-1).contiguous(), dtype=wp.vec3)
    wp_contact_normals_world = wp.from_torch(data['contact_normals_world'].squeeze(-1).contiguous(), dtype=wp.vec3)
    wp_body_vel = wp.from_torch(data['body_vel'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)
    wp_body_vel_prev = wp.from_torch(data['body_vel_prev'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)

    C_actual = data["lambda_n"].shape[1] if data["lambda_n"].ndim > 1 else 0
    wp_lambda_n = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C_actual == 0 else wp.from_torch(data['lambda_n'].squeeze(-1).contiguous(), dtype=wp.float32)
    wp_contact_mask = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C_actual == 0 else wp.from_torch(data['contact_mask'].float().contiguous(), dtype=wp.float32)
    wp_contact_weight = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C_actual == 0 else wp.from_torch(data['contact_weight'].contiguous(), dtype=wp.float32)
    wp_restitution = wp.from_torch(data['restitution'].squeeze(-1).contiguous(), dtype=wp.float32)

    if C_actual > 0:
        # Penetration depths
        penetration_depth_wp = get_penetration_depths_warp(
            wp_body_trans, wp_contact_points_local, wp_ground_points_world, wp_contact_normals_world, device=wp_device
        )

        # Jacobians
        J_n_wp = compute_contact_jacobians_warp(
            wp_body_trans, wp_contact_points_local, wp_contact_normals_world, wp_contact_mask, device=wp_device
        )

        # Residuals
        res_wp = get_contact_residuals_warp(
            wp_body_vel, wp_body_vel_prev, wp_lambda_n, J_n_wp,
            penetration_depth_wp, wp_contact_mask, wp_contact_weight,
            wp_restitution, dt, data["stabilization_factor"],
            data["fb_alpha"], data["fb_beta"], data["fb_epsilon"], device=wp_device
        )

        # Derivatives
        dres_dbody_vel_wp, dres_dlambda_n_wp_diag = get_contact_derivatives_warp(
            wp_body_vel, wp_body_vel_prev, wp_lambda_n, J_n_wp,
            penetration_depth_wp, wp_contact_mask, wp_contact_weight,
            wp_restitution, dt, data["stabilization_factor"],
            data["fb_alpha"], data["fb_beta"], data["fb_epsilon"], device=wp_device
        )
    else:
        # Handle C=0 case for Warp outputs
        penetration_depth_wp = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device)
        J_n_wp = wp.empty(shape=(B, 0), dtype=wp.spatial_vector, device=wp_device)
        res_wp = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device)
        dres_dbody_vel_wp = wp.empty(shape=(B, 0, 6), dtype=wp.float32, device=wp_device)
        dres_dlambda_n_wp_diag = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device)


    # Convert Warp results back to PyTorch
    # Adjust shapes to match PyTorch outputs
    penetration_depth_wp_torch = wp.to_torch(penetration_depth_wp).clone().unsqueeze(-1)
    J_n_wp_torch = wp.to_torch(J_n_wp).clone().view(B, C_actual, 6) # Reshape spatial_vector
    res_wp_torch = wp.to_torch(res_wp).clone().unsqueeze(-1) # Add last dimension
    dres_dbody_vel_wp_torch = wp.to_torch(dres_dbody_vel_wp).clone()
    dres_dlambda_n_wp_torch_diag = wp.to_torch(dres_dlambda_n_wp_diag).clone() # This IS the diagonal

    # Compare
    print(f"Comparing results for '{test_name}'")
    assert torch.allclose(penetration_depth_pt, penetration_depth_wp_torch, atol=1e-5), f"{test_name}: Penetration Depth mismatch"
    assert torch.allclose(J_n_pt, J_n_wp_torch, atol=1e-5), f"{test_name}: J_n mismatch"
    assert torch.allclose(res_pt, res_wp_torch, atol=1e-5), f"{test_name}: Residuals mismatch"
    assert torch.allclose(dres_dbody_vel_pt, dres_dbody_vel_wp_torch, atol=1e-5), f"{test_name}: dres/dbody_vel mismatch"
    assert torch.allclose(dres_dlambda_n_pt_diag.diagonal(dim1=1, dim2=2), dres_dlambda_n_wp_torch_diag, atol=1e-5), f"{test_name}: dres/dlambda_n (diagonal) mismatch"

    print(f"{test_name}: All comparisons passed.")


# --- Performance Benchmark ---
def run_contact_performance_benchmark(B: int, C: int, device: torch.device, num_runs: int = 100):
    """Runs a performance benchmark for PyTorch and Warp (with graph) for contact constraints."""
    print(f"\nRunning Contact Constraint Performance Test (B={B}, C={C})")
    try:
        data = setup_contact_test_case_data(B, C, device)
    except ValueError as e:
        print(f"Skipping benchmark: {e}")
        return

    dt = data["dt"]

    # PyTorch performance (All functions: Penetration, Jacobians, Residuals, Derivatives)
    constraint_pt = ContactConstraint(device=device)
    # Warm-up
    if C > 0:
        penetration_depth_pt = constraint_pt.get_penetration_depths(data["body_trans"], data["contact_points_local"], data["ground_points_world"], data["contact_normals_world"])
        J_n_pt = constraint_pt.compute_contact_jacobians(data["body_trans"], data["contact_points_local"], data["contact_normals_world"], data["contact_mask"])
        res_pt = constraint_pt.get_residuals(data["body_vel"], data["body_vel_prev"], data["lambda_n"], J_n_pt, penetration_depth_pt, data["contact_mask"], data["contact_weight"], data["restitution"], dt)
        dres_dbody_vel_pt, dres_dlambda_n_pt_diag = constraint_pt.get_derivatives(data["body_vel"], data["body_vel_prev"], data["lambda_n"], J_n_pt, penetration_depth_pt, data["contact_mask"], data["contact_weight"], data["restitution"], dt)
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_runs):
         if C > 0:
            penetration_depth_pt = constraint_pt.get_penetration_depths(data["body_trans"], data["contact_points_local"], data["ground_points_world"], data["contact_normals_world"])
            J_n_pt = constraint_pt.compute_contact_jacobians(data["body_trans"], data["contact_points_local"], data["contact_normals_world"], data["contact_mask"])
            res_pt = constraint_pt.get_residuals(data["body_vel"], data["body_vel_prev"], data["lambda_n"], J_n_pt, penetration_depth_pt, data["contact_mask"], data["contact_weight"], data["restitution"], dt)
            dres_dbody_vel_pt, dres_dlambda_n_pt_diag = constraint_pt.get_derivatives(data["body_vel"], data["body_vel_prev"], data["lambda_n"], J_n_pt, penetration_depth_pt, data["contact_mask"], data["contact_weight"], data["restitution"], dt)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_runs
    print(f"PyTorch average time per call (All functions): {pytorch_time:.6f} seconds")

    # Warp performance (All functions with graph)
    wp.init()
    wp_device = wp.get_device()

    # Convert data to Warp arrays for graph capture (ensure contiguity)
    wp_body_trans = wp.from_torch(swap_quaternion_real_part(data['body_trans']).squeeze(-1).contiguous(), dtype=wp.transform)
    wp_contact_points_local = wp.from_torch(data['contact_points_local'].squeeze(-1).contiguous(), dtype=wp.vec3)
    wp_ground_points_world = wp.from_torch(data['ground_points_world'].squeeze(-1).contiguous(), dtype=wp.vec3)
    wp_contact_normals_world = wp.from_torch(data['contact_normals_world'].squeeze(-1).contiguous(), dtype=wp.vec3)
    wp_body_vel = wp.from_torch(data['body_vel'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)
    wp_body_vel_prev = wp.from_torch(data['body_vel_prev'].squeeze(-1).contiguous(), dtype=wp.spatial_vector)

    C_actual = data["lambda_n"].shape[1] if data["lambda_n"].ndim > 1 else 0
    wp_lambda_n = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C_actual == 0 else wp.from_torch(data['lambda_n'].squeeze(-1).contiguous(), dtype=wp.float32)
    wp_contact_mask = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C_actual == 0 else wp.from_torch(data['contact_mask'].float().contiguous(), dtype=wp.float32)
    wp_contact_weight = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C_actual == 0 else wp.from_torch(data['contact_weight'].contiguous(), dtype=wp.float32)
    wp_restitution = wp.from_torch(data['restitution'].squeeze(-1).contiguous(), dtype=wp.float32)

    # Pre-allocate output arrays for Warp
    wp_penetration_depth = wp.zeros(shape=(B, C_actual), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_J_n = wp.zeros(shape=(B, C_actual), dtype=wp.spatial_vector, device=wp_device, requires_grad=False)
    wp_res = wp.zeros(shape=(B, C_actual), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_dres_dbody_vel = wp.zeros(shape=(B, C_actual, 6), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_dres_dlambda_n_diag = wp.zeros(shape=(B, C_actual), dtype=wp.float32, device=wp_device, requires_grad=False)

    # Capture the graph
    graph = None
    with wp.ScopedCapture() as capture:
        # Clear output tensors before use in the captured graph
        wp_penetration_depth.zero_()
        wp_J_n.zero_()
        wp_res.zero_()
        wp_dres_dbody_vel.zero_()
        wp_dres_dlambda_n_diag.zero_()

        if C_actual > 0:
             # Launch kernels using wp.launch
             wp.launch(
                 kernel=compute_penetration_depths_kernel,
                 dim=(B, C_actual),
                 inputs=[
                    wp_body_trans, wp_contact_points_local, wp_ground_points_world,
                    wp_contact_normals_world, wp_penetration_depth
                 ],
                 device=wp_device
             )

             wp.launch(
                 kernel=compute_contact_jacobians_kernel,
                 dim=(B, C_actual),
                 inputs=[
                     wp_body_trans, wp_contact_points_local, wp_contact_normals_world,
                     wp_contact_mask, wp_J_n
                 ],
                 device=wp_device
             )

             wp.launch(
                 kernel=get_contact_residuals_kernel,
                 dim=(B, C_actual),
                 inputs=[
                     wp_body_vel, wp_body_vel_prev, wp_lambda_n, wp_J_n, wp_penetration_depth,
                     wp_contact_mask, wp_contact_weight, wp_restitution, dt,
                         data["stabilization_factor"], data["fb_alpha"], data["fb_beta"], data["fb_epsilon"],
                     wp_res
                 ],
                 device=wp_device
             )

             wp.launch(
                 kernel=get_contact_derivatives_kernel,
                 dim=(B, C_actual),
                 inputs=[
                     wp_body_vel, wp_body_vel_prev, wp_lambda_n, wp_J_n, wp_penetration_depth,
                     wp_contact_mask, wp_contact_weight, wp_restitution, dt,
                         data["stabilization_factor"], data["fb_alpha"], data["fb_beta"], data["fb_epsilon"],
                     wp_dres_dbody_vel, wp_dres_dlambda_n_diag
                 ],
                 device=wp_device
             )

    graph = capture.graph

    # Warm-up the graph
    wp.capture_launch(graph)
    torch.cuda.synchronize()  # Synchronize after Warp graph execution

    # Graph execution timing
    start_time = time.time()
    for _ in range(num_runs):
        wp.capture_launch(graph)
    torch.cuda.synchronize() # Synchronize after all graph executions
    warp_graph_execution_time = (time.time() - start_time) / num_runs
    print(f"Warp with Graph (execution only) average time per call (All functions): {warp_graph_execution_time:.6f} seconds")

    if warp_graph_execution_time > 1e-9: # Avoid division by zero if time is negligible
         print(f"Warp is {pytorch_time / warp_graph_execution_time:.2f}x faster than PyTorch for this test case.")
    else:
        print("Warp time is negligible.")


# --- Main execution block ---
if __name__ == "__main__":
    wp.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Comparison Tests
    run_contact_comparison_test("Test Case: No Contacts", B=5, C=0, device=device)
    run_contact_comparison_test("Test Case: Single Contact", B=1, C=1, device=device)
    run_contact_comparison_test("Test Case: Multiple Contacts per Body", B=2, C=3, device=device)
    run_contact_comparison_test("Test Case: Larger Scale", B=10, C=8, device=device)

    # Performance Benchmark
    # Choose a representative scale
    run_contact_performance_benchmark(B=20, C=16, device=device, num_runs=100)

    print("\nAll Contact Constraint tests finished.")

# 12.2 ms
# 0.75 ms
# 16x better performance

# Only writing the jacobian takes around 2 ms

