# friction_constraint_warp.py
import time
from typing import Tuple

import torch
import warp as wp
from jaxtyping import Float, Bool
from pbd_torch.ncp import ScaledFisherBurmeister
from pbd_torch.constraints import FrictionConstraint
from pbd_torch.utils import swap_quaternion_real_part

@wp.func
def scaled_fisher_burmeister_evaluate(
    a: wp.float32, b: wp.float32, alpha: wp.float32, beta: wp.float32, epsilon: wp.float32
) -> wp.float32:
    """Warp kernel-side evaluation of the Scaled Fisher-Burmeister function."""
    scaled_a = alpha * a
    scaled_b = beta * b
    norm = wp.sqrt(scaled_a**2.0 + scaled_b**2.0 + epsilon)
    return scaled_a + scaled_b - norm

@wp.func
def scaled_fisher_burmeister_derivatives(
    a: wp.float32, b: wp.float32, alpha: wp.float32, beta: wp.float32, epsilon: wp.float32
) -> tuple[wp.float32, wp.float32]:
    """Warp kernel-side derivatives of the Scaled Fisher-Burmeister function."""
    scaled_a = alpha * a
    scaled_b = beta * b
    norm = wp.sqrt(scaled_a**2.0 + scaled_b**2.0 + epsilon)
    da = alpha * (1.0 - scaled_a / norm)
    db = beta * (1.0 - scaled_b / norm)
    return da, db

# --- Warp Kernels for Friction Constraint ---

@wp.kernel
def compute_tangential_basis_kernel(
    contact_normals_world: wp.array(dtype=wp.vec3, ndim=2), # [B, C] world contact normals
    t1: wp.array(dtype=wp.vec3, ndim=2),                   # Output: [B, C] first tangent vectors
    t2: wp.array(dtype=wp.vec3, ndim=2)                    # Output: [B, C] second tangent vectors
):
    b, c = wp.tid() # Body index [0, B), contact index [0, C)

    n = contact_normals_world[b, c]
    ref = wp.vec3(1.0, 0.0, 0.0)

    # Check if normal is nearly parallel to ref [1, 0, 0]
    dot = wp.dot(n, ref)
    if wp.abs(dot) > 0.99:
        ref = wp.vec3(0.0, 1.0, 0.0)

    # Compute first tangent: t1 = n × ref
    t1_vec = wp.cross(n, ref)
    t1_norm = wp.length(t1_vec)
    if t1_norm > 1e-6:
        t1_vec = t1_vec / t1_norm
    else:
        t1_vec = wp.vec3(0.0, 0.0, 0.0)

    # Compute second tangent: t2 = n × t1
    t2_vec = wp.cross(n, t1_vec)
    t2_norm = wp.length(t2_vec)
    if t2_norm > 1e-6:
        t2_vec = t2_vec / t2_norm
    else:
        t2_vec = wp.vec3(0.0, 0.0, 0.0)

    t1[b, c] = t1_vec
    t2[b, c] = t2_vec

@wp.kernel
def compute_tangential_jacobians_kernel(
    body_trans: wp.array(dtype=wp.transform),              # [B] (position + quat(xyzw))
    contact_points_local: wp.array(dtype=wp.vec3, ndim=2), # [B, C] local contact points
    t1: wp.array(dtype=wp.vec3, ndim=2),                   # [B, C] first tangent vectors
    t2: wp.array(dtype=wp.vec3, ndim=2),                   # [B, C] second tangent vectors
    contact_mask: wp.array(dtype=wp.float32, ndim=2),      # [B, C] float mask (1.0 or 0.0)
    J_t: wp.array(dtype=wp.float32, ndim=3)                # Output: [B, 2C, 6]
):
    b, c = wp.tid() # Body index [0, B), contact index [0, C)

    mask = contact_mask[b, c]
    body_rot = wp.transform_get_rotation(body_trans[b])
    r = wp.quat_rotate(body_rot, contact_points_local[b, c])

    # First tangent direction (t1)
    t1_vec = t1[b, c]
    r_cross_t1 = wp.cross(r, t1_vec)
    for j in range(3):
        J_t[b, c, j] = -r_cross_t1[j] * mask
        J_t[b, c, j + 3] = -t1_vec[j] * mask

    # Second tangent direction (t2)
    t2_vec = t2[b, c]
    r_cross_t2 = wp.cross(r, t2_vec)
    for j in range(3):
        J_t[b, c + contact_points_local.shape[1], j] = -r_cross_t2[j] * mask
        J_t[b, c + contact_points_local.shape[1], j + 3] = -t2_vec[j] * mask

@wp.kernel
def get_friction_residuals_kernel(
    body_vel: wp.array(dtype=wp.spatial_vector),           # [B]
    lambda_n: wp.array(dtype=wp.float32, ndim=2),          # [B, C] normal impulses
    lambda_t: wp.array(dtype=wp.float32, ndim=2),          # [B, 2C] tangential impulses
    gamma: wp.array(dtype=wp.float32, ndim=2),             # [B, C] friction cone variables
    J_t: wp.array(dtype=wp.float32, ndim=3),               # [B, 2C, 6]
    contact_mask: wp.array(dtype=wp.float32, ndim=2),      # [B, C] float mask (1.0 or 0.0)
    friction_coeff: wp.array(dtype=wp.float32, ndim=1),    # [B]
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    fb_epsilon: wp.float32,
    res: wp.array(dtype=wp.float32, ndim=2)                # Output: [B, 3C]
):
    b, c = wp.tid() # Body index [0, B), contact index [0, C)

    active_mask = contact_mask[b, c]
    inactive_mask = 1.0 - active_mask
    C = contact_mask.shape[1]

    # Compute tangential velocities
    v_t1 = 0.0
    v_t2 = 0.0
    for j in range(6):
        v_t1 += J_t[b, c, j] * body_vel[b][j]
        v_t2 += J_t[b, c + C, j] * body_vel[b][j]

    lambda_t1 = lambda_t[b, c]
    lambda_t2 = lambda_t[b, c + C]
    gamma_val = gamma[b, c]
    mu = friction_coeff[b]
    lambda_n_val = lambda_n[b, c]

    # Compute friction impulse norm
    lambda_t_norm = wp.sqrt(lambda_t1**2.0 + lambda_t2**2.0 + fb_epsilon)

    # Residuals for friction directions (fr1, fr2)
    res_fr1_act = v_t1 + gamma_val * lambda_t1 / (lambda_t_norm + fb_epsilon)
    res_fr2_act = v_t2 + gamma_val * lambda_t2 / (lambda_t_norm + fb_epsilon)
    res_fr1_inact = -lambda_t1
    res_fr2_inact = -lambda_t2

    res[b, c] = res_fr1_act * active_mask + res_fr1_inact * inactive_mask
    res[b, c + C] = res_fr2_act * active_mask + res_fr2_inact * inactive_mask

    # Residual for friction cone (frc)
    res_frc_act = scaled_fisher_burmeister_evaluate(
        gamma_val, mu * lambda_n_val - lambda_t_norm, fb_alpha, fb_beta, fb_epsilon
    )
    res_frc_inact = -gamma_val
    res[b, c + 2 * C] = res_frc_act * active_mask + res_frc_inact * inactive_mask

@wp.kernel
def get_friction_derivatives_kernel(
    body_vel: wp.array(dtype=wp.spatial_vector),           # [B]
    lambda_n: wp.array(dtype=wp.float32, ndim=2),          # [B, C] normal impulses
    lambda_t: wp.array(dtype=wp.float32, ndim=2),          # [B, 2C] tangential impulses
    gamma: wp.array(dtype=wp.float32, ndim=2),             # [B, C] friction cone variables
    J_t: wp.array(dtype=wp.float32, ndim=3),               # [B, 2C, 6]
    contact_mask: wp.array(dtype=wp.float32, ndim=2),      # [B, C] float mask (1.0 or 0.0)
    friction_coeff: wp.array(dtype=wp.float32, ndim=1),    # [B]
    fb_alpha: wp.float32,
    fb_beta: wp.float32,
    fb_epsilon: wp.float32,
    dres_dbody_vel: wp.array(dtype=wp.float32, ndim=3),    # Output: [B, 3C, 6]
    dres_dlambda_n: wp.array(dtype=wp.float32, ndim=3),    # Output: [B, 3C, C]
    dres_dlambda_t: wp.array(dtype=wp.float32, ndim=3),    # Output: [B, 3C, 2C]
    dres_dgamma: wp.array(dtype=wp.float32, ndim=3)        # Output: [B, 3C, C]
):
    b, c = wp.tid() # Body index [0, B), contact index [0, C)

    active_mask = contact_mask[b, c]
    inactive_mask = 1.0 - active_mask
    C = contact_mask.shape[1]

    lambda_t1 = lambda_t[b, c]
    lambda_t2 = lambda_t[b, c + C]
    gamma_val = gamma[b, c]
    mu = friction_coeff[b]
    lambda_n_val = lambda_n[b, c]
    lambda_t_norm = wp.sqrt(lambda_t1**2.0 + lambda_t2**2.0 + fb_epsilon)

    # Derivatives for friction directions (fr1, fr2)
    # ∂res_fr1 / ∂body_vel
    for j in range(6):
        dres_dbody_vel[b, c, j] = J_t[b, c, j] * active_mask
        dres_dbody_vel[b, c + C, j] = J_t[b, c + C, j] * active_mask
        dres_dbody_vel[b, c + 2 * C, j] = 0.0

    # ∂res_fr1 / ∂lambda_t1
    dres_fr1_dlambda_t1_act = gamma_val * (lambda_t2**2.0 + fb_epsilon) / (lambda_t_norm**3.0 + fb_epsilon)
    dres_fr1_dlambda_t1_inact = -1.0
    dres_dlambda_t[b, c, c] = dres_fr1_dlambda_t1_act * active_mask + dres_fr1_dlambda_t1_inact * inactive_mask

    # ∂res_fr1 / ∂lambda_t2
    dres_fr1_dlambda_t2 = -gamma_val * lambda_t1 * lambda_t2 / (lambda_t_norm**3.0 + fb_epsilon) * active_mask
    dres_dlambda_t[b, c, c + C] = dres_fr1_dlambda_t2

    # ∂res_fr2 / ∂lambda_t2
    dres_fr2_dlambda_t2_act = gamma_val * (lambda_t1**2.0 + fb_epsilon) / (lambda_t_norm**3.0 + fb_epsilon)
    dres_fr2_dlambda_t2_inact = -1.0
    dres_dlambda_t[b, c + C, c + C] = dres_fr2_dlambda_t2_act * active_mask + dres_fr2_dlambda_t2_inact * inactive_mask

    # ∂res_fr2 / ∂lambda_t1
    dres_fr2_dlambda_t1 = -gamma_val * lambda_t1 * lambda_t2 / (lambda_t_norm**3.0 + fb_epsilon) * active_mask
    dres_dlambda_t[b, c + C, c] = dres_fr2_dlambda_t1

    # ∂res_fr1 / ∂gamma
    dres_fr1_dgamma = (lambda_t1 / (lambda_t_norm + fb_epsilon)) * active_mask
    dres_dgamma[b, c, c] = dres_fr1_dgamma

    # ∂res_fr2 / ∂gamma
    dres_fr2_dgamma = (lambda_t2 / (lambda_t_norm + fb_epsilon)) * active_mask
    dres_dgamma[b, c + C, c] = dres_fr2_dgamma

    # ∂res_fr / ∂lambda_n
    dres_dlambda_n[b, c, c] = 0.0
    dres_dlambda_n[b, c + C, c] = 0.0

    # Derivatives for friction cone (frc)
    da_frc_act, db_frc_act = scaled_fisher_burmeister_derivatives(
        gamma_val, mu * lambda_n_val - lambda_t_norm, fb_alpha, fb_beta, fb_epsilon
    )
    db_frc = db_frc_act * active_mask

    # ∂res_frc / ∂lambda_n
    dres_dlambda_n[b, c + 2 * C, c] = db_frc * mu

    # ∂res_frc / ∂lambda_t1
    dres_frc_dlambda_t1 = -db_frc * lambda_t1 / (lambda_t_norm + fb_epsilon)
    dres_dlambda_t[b, c + 2 * C, c] = dres_frc_dlambda_t1

    # ∂res_frc / ∂lambda_t2
    dres_frc_dlambda_t2 = -db_frc * lambda_t2 / (lambda_t_norm + fb_epsilon)
    dres_dlambda_t[b, c + 2 * C, c + C] = dres_frc_dlambda_t2

    # ∂res_frc / ∂gamma
    dres_frc_dgamma_inact = -1.0
    dres_dgamma[b, c + 2 * C, c] = da_frc_act * active_mask + dres_frc_dgamma_inact * inactive_mask

# --- Warp Functions (Wrappers for Kernels) ---

def compute_tangential_basis_warp(
    contact_normals_world: wp.array, # [B, C] wp.vec3
    device: wp.context.Device
) -> Tuple[wp.array, wp.array]: # t1, t2: [B, C] wp.vec3
    B = contact_normals_world.shape[0]
    C = contact_normals_world.shape[1] if len(contact_normals_world.shape) > 1 else 0
    if C == 0:
        return (wp.empty(shape=(B, 0), dtype=wp.vec3, device=device),
                wp.empty(shape=(B, 0), dtype=wp.vec3, device=device))

    t1 = wp.zeros(shape=(B, C), dtype=wp.vec3, device=device)
    t2 = wp.zeros(shape=(B, C), dtype=wp.vec3, device=device)
    wp.launch(
        kernel=compute_tangential_basis_kernel,
        dim=(B, C),
        inputs=[contact_normals_world, t1, t2],
        device=device
    )
    return t1, t2

def compute_tangential_jacobians_warp(
    body_trans: wp.array, # [B] wp.transform
    contact_points_local: wp.array, # [B, C] wp.vec3
    contact_normals_world: wp.array, # [B, C] wp.vec3
    contact_mask: wp.array, # [B, C] wp.uint8
    device: wp.context.Device
) -> wp.array: # [B, 2C, 6] wp.float32
    B = body_trans.shape[0]
    C = contact_points_local.shape[1] if len(contact_points_local.shape) > 1 else 0
    if C == 0:
        return wp.empty(shape=(B, 0, 6), dtype=wp.float32, device=device)

    # Compute tangent basis
    t1, t2 = compute_tangential_basis_warp(contact_normals_world, device)

    J_t = wp.zeros(shape=(B, 2 * C, 6), dtype=wp.float32, device=device)
    wp.launch(
        kernel=compute_tangential_jacobians_kernel,
        dim=(B, C),
        inputs=[body_trans, contact_points_local, t1, t2, contact_mask, J_t],
        device=device
    )
    return J_t

def get_friction_residuals_warp(
    body_vel: wp.array, # [B] wp.spatial_vector
    lambda_n: wp.array, # [B, C] wp.float32
    lambda_t: wp.array, # [B, 2C] wp.float32
    gamma: wp.array, # [B, C] wp.float32
    J_t: wp.array, # [B, 2C, 6] wp.float32
    contact_mask: wp.array, # [B, C] wp.uint8
    friction_coeff: wp.array, # [B] wp.float32
    fb_alpha: float,
    fb_beta: float,
    fb_epsilon: float,
    device: wp.context.Device
) -> wp.array: # [B, 3C] wp.float32
    B = body_vel.shape[0]
    C = lambda_n.shape[1] if len(lambda_n.shape) > 1 else 0
    if C == 0:
        return wp.empty(shape=(B, 0), dtype=wp.float32, device=device)

    res = wp.zeros(shape=(B, 3 * C), dtype=wp.float32, device=device)
    wp.launch(
        kernel=get_friction_residuals_kernel,
        dim=(B, C),
        inputs=[
            body_vel, lambda_n, lambda_t, gamma, J_t, contact_mask,
            friction_coeff, fb_alpha, fb_beta, fb_epsilon, res
        ],
        device=device
    )
    return res

def get_friction_derivatives_warp(
    body_vel: wp.array, # [B] wp.spatial_vector
    lambda_n: wp.array, # [B, C] wp.float32
    lambda_t: wp.array, # [B, 2C] wp.float32
    gamma: wp.array, # [B, C] wp.float32
    J_t: wp.array, # [B, 2C, 6] wp.float32
    contact_mask: wp.array, # [B, C] wp.uint8
    friction_coeff: wp.array, # [B] wp.float32
    fb_alpha: float,
    fb_beta: float,
    fb_epsilon: float,
    device: wp.context.Device
) -> Tuple[wp.array, wp.array, wp.array, wp.array]: # [B, 3C, 6], [B, 3C, C], [B, 3C, 2C], [B, 3C, C]
    B = body_vel.shape[0]
    C = lambda_n.shape[1] if len(lambda_n.shape) > 1 else 0
    if C == 0:
        return (
            wp.empty(shape=(B, 0, 6), dtype=wp.float32, device=device),
            wp.empty(shape=(B, 0, C), dtype=wp.float32, device=device),
            wp.empty(shape=(B, 0, 2 * C), dtype=wp.float32, device=device),
            wp.empty(shape=(B, 0, C), dtype=wp.float32, device=device)
        )

    dres_dbody_vel = wp.zeros(shape=(B, 3 * C, 6), dtype=wp.float32, device=device)
    dres_dlambda_n = wp.zeros(shape=(B, 3 * C, C), dtype=wp.float32, device=device)
    dres_dlambda_t = wp.zeros(shape=(B, 3 * C, 2 * C), dtype=wp.float32, device=device)
    dres_dgamma = wp.zeros(shape=(B, 3 * C, C), dtype=wp.float32, device=device)

    wp.launch(
        kernel=get_friction_derivatives_kernel,
        dim=(B, C),
        inputs=[
            body_vel, lambda_n, lambda_t, gamma, J_t, contact_mask,
            friction_coeff, fb_alpha, fb_beta, fb_epsilon,
            dres_dbody_vel, dres_dlambda_n, dres_dlambda_t, dres_dgamma
        ],
        device=device
    )
    return dres_dbody_vel, dres_dlambda_n, dres_dlambda_t, dres_dgamma

# --- Test Data Setup ---

def setup_friction_test_case_data(B: int, C: int, device: torch.device):
    """Generates random data for a friction constraint test case."""
    if C > 0 and B < 1:
        raise ValueError("Cannot create contacts with B < 1 if C > 0")

    # Random body transforms (pos + quat [w, x, y, z])
    body_trans = torch.randn(B, 7, 1, device=device) if B > 0 else torch.empty(0, 7, 1, device=device)
    if B > 0:
        # Normalize quaternions
        quats = body_trans[:, 3:, 0]
        norms = torch.linalg.norm(quats, dim=-1, keepdim=True)
        norms[norms == 0] = 1.0
        body_trans[:, 3:, 0] = quats / norms
        body_trans[:, 3:, :] = body_trans[:, 3:, 0].unsqueeze(-1)

    # Random contact points and normals
    contact_points_local = torch.randn(B, C, 3, 1, device=device) if C > 0 else torch.empty(B, 0, 3, 1, device=device)
    contact_normals_world = torch.randn(B, C, 3, 1, device=device) if C > 0 else torch.empty(B, 0, 3, 1, device=device)
    if C > 0:
        # Normalize normals
        norms = torch.linalg.norm(contact_normals_world, dim=2, keepdim=True)
        norms[norms == 0] = 1.0
        contact_normals_world = contact_normals_world / norms

    # Random inputs
    body_vel = torch.randn(B, 6, 1, device=device, requires_grad=True) if B > 0 else torch.empty(0, 6, 1, device=device)
    lambda_n = torch.randn(B, C, 1, device=device, requires_grad=True) if C > 0 else torch.empty(B, 0, 1, device=device)
    lambda_t = torch.randn(B, 2 * C, 1, device=device, requires_grad=True) if C > 0 else torch.empty(B, 0, 1, device=device)
    gamma = torch.randn(B, C, 1, device=device, requires_grad=True) if C > 0 else torch.empty(B, 0, 1, device=device)
    contact_mask = (torch.rand(B, C, device=device) > 0.2).bool() if C > 0 else torch.empty(B, 0, device=device, dtype=torch.bool)
    friction_coeff = torch.rand(B, 1, device=device) if B > 0 else torch.empty(0, 1, device=device)

    # Parameters
    fb_alpha = 0.3
    fb_beta = 0.3
    fb_epsilon = 1e-12

    # Set requires_grad=False for non-differentiated inputs
    body_trans.requires_grad_(False)
    contact_points_local.requires_grad_(False)
    contact_normals_world.requires_grad_(False)
    contact_mask.requires_grad_(False)
    friction_coeff.requires_grad_(False)

    return {
        "body_trans": body_trans,
        "contact_points_local": contact_points_local,
        "contact_normals_world": contact_normals_world,
        "body_vel": body_vel,
        "lambda_n": lambda_n,
        "lambda_t": lambda_t,
        "gamma": gamma,
        "contact_mask": contact_mask,
        "friction_coeff": friction_coeff,
        "fb_alpha": fb_alpha,
        "fb_beta": fb_beta,
        "fb_epsilon": fb_epsilon
    }

# --- Comparison Test ---

def run_friction_comparison_test(test_name: str, B: int, C: int, device: torch.device):
    print(f"\nRunning Friction Constraint Comparison Test '{test_name}' (B={B}, C={C})")
    try:
        data = setup_friction_test_case_data(B, C, device)
    except ValueError as e:
        print(f"Skipping test due to setup error: {e}")
        return

    wp_device = wp.get_device()

    # PyTorch inputs
    body_trans = data["body_trans"]
    contact_points = data["contact_points_local"]
    contact_normals = data["contact_normals_world"]
    body_vel = data["body_vel"]
    lambda_n = data["lambda_n"]
    lambda_t = data["lambda_t"]
    gamma = data["gamma"]
    contact_mask = data["contact_mask"]
    friction_coeff = data["friction_coeff"]
    fb_alpha = data["fb_alpha"]
    fb_beta = data["fb_beta"]
    fb_epsilon = data["fb_epsilon"]

    # PyTorch computations
    constraint_pt = FrictionConstraint(device=device)
    if C > 0:
        J_t_pt = constraint_pt.compute_tangential_jacobians(body_trans, contact_points, contact_normals, contact_mask)
        res_pt = constraint_pt.get_residuals(body_vel, lambda_n, lambda_t, gamma, J_t_pt, contact_mask, friction_coeff)
        dres_dbody_vel_pt, dres_dlambda_n_pt, dres_dlambda_t_pt, dres_dgamma_pt = constraint_pt.get_derivatives(
            body_vel, lambda_n, lambda_t, gamma, J_t_pt, contact_mask, friction_coeff
        )
    else:
        J_t_pt = torch.empty(B, 0, 6, device=device)
        res_pt = torch.empty(B, 0, 1, device=device)
        dres_dbody_vel_pt = torch.empty(B, 0, 6, device=device)
        dres_dlambda_n_pt = torch.empty(B, 0, 0, device=device)
        dres_dlambda_t_pt = torch.empty(B, 0, 0, device=device)
        dres_dgamma_pt = torch.empty(B, 0, 0, device=device)

    # Warp computations
    wp_body_trans = wp.from_torch(swap_quaternion_real_part(body_trans).squeeze(-1).contiguous(), dtype=wp.transform)
    wp_contact_points = wp.from_torch(contact_points.squeeze(-1).contiguous(), dtype=wp.vec3)
    wp_contact_normals = wp.from_torch(contact_normals.squeeze(-1).contiguous(), dtype=wp.vec3)
    wp_body_vel = wp.from_torch(body_vel.squeeze(-1).contiguous(), dtype=wp.spatial_vector)
    wp_lambda_n = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C == 0 else wp.from_torch(lambda_n.squeeze(-1).contiguous(), dtype=wp.float32)
    wp_lambda_t = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C == 0 else wp.from_torch(lambda_t.squeeze(-1).contiguous(), dtype=wp.float32)
    wp_gamma = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C == 0 else wp.from_torch(gamma.squeeze(-1).contiguous(), dtype=wp.float32)
    wp_contact_mask = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C == 0 else wp.from_torch(contact_mask.float().contiguous(), dtype=wp.float32)
    wp_friction_coeff = wp.from_torch(friction_coeff.squeeze(-1).contiguous(), dtype=wp.float32)

    if C > 0:
        J_t_wp = compute_tangential_jacobians_warp(
            wp_body_trans, wp_contact_points, wp_contact_normals, wp_contact_mask, wp_device
        )
        res_wp = get_friction_residuals_warp(
            wp_body_vel, wp_lambda_n, wp_lambda_t, wp_gamma, J_t_wp, wp_contact_mask,
            wp_friction_coeff, fb_alpha, fb_beta, fb_epsilon, wp_device
        )
        dres_dbody_vel_wp, dres_dlambda_n_wp, dres_dlambda_t_wp, dres_dgamma_wp = get_friction_derivatives_warp(
            wp_body_vel, wp_lambda_n, wp_lambda_t, wp_gamma, J_t_wp, wp_contact_mask,
            wp_friction_coeff, fb_alpha, fb_beta, fb_epsilon, wp_device
        )
    else:
        J_t_wp = wp.empty(shape=(B, 0, 6), dtype=wp.float32, device=wp_device)
        res_wp = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device)
        dres_dbody_vel_wp = wp.empty(shape=(B, 0, 6), dtype=wp.float32, device=wp_device)
        dres_dlambda_n_wp = wp.empty(shape=(B, 0, 0), dtype=wp.float32, device=wp_device)
        dres_dlambda_t_wp = wp.empty(shape=(B, 0, 0), dtype=wp.float32, device=wp_device)
        dres_dgamma_wp = wp.empty(shape=(B, 0, 0), dtype=wp.float32, device=wp_device)

    # Convert Warp results to PyTorch
    J_t_wp_torch = wp.to_torch(J_t_wp).clone()
    res_wp_torch = wp.to_torch(res_wp).clone().unsqueeze(-1)
    dres_dbody_vel_wp_torch = wp.to_torch(dres_dbody_vel_wp).clone()
    dres_dlambda_n_wp_torch = wp.to_torch(dres_dlambda_n_wp).clone()
    dres_dlambda_t_wp_torch = wp.to_torch(dres_dlambda_t_wp).clone()
    dres_dgamma_wp_torch = wp.to_torch(dres_dgamma_wp).clone()

    # Compare
    print(f"Comparing results for '{test_name}'")
    assert torch.allclose(J_t_pt, J_t_wp_torch, atol=1e-5), f"{test_name}: J_t mismatch"
    assert torch.allclose(res_pt, res_wp_torch, atol=1e-5), f"{test_name}: Residuals mismatch"
    assert torch.allclose(dres_dbody_vel_pt, dres_dbody_vel_wp_torch, atol=1e-5), f"{test_name}: dres_dbody_vel mismatch"
    assert torch.allclose(dres_dlambda_n_pt, dres_dlambda_n_wp_torch, atol=1e-5), f"{test_name}: dres_dlambda_n mismatch"
    assert torch.allclose(dres_dlambda_t_pt, dres_dlambda_t_wp_torch, atol=1e-5), f"{test_name}: dres_dlambda_t mismatch"
    assert torch.allclose(dres_dgamma_pt, dres_dgamma_wp_torch, atol=1e-5), f"{test_name}: dres_dgamma mismatch"
    print(f"{test_name}: All comparisons passed.")

# --- Performance Benchmark ---

def run_friction_performance_benchmark(B: int, C: int, device: torch.device, num_runs: int = 100):
    print(f"\nRunning Friction Constraint Performance Test (B={B}, C={C})")
    try:
        data = setup_friction_test_case_data(B, C, device)
    except ValueError as e:
        print(f"Skipping benchmark: {e}")
        return

    # PyTorch performance
    constraint_pt = FrictionConstraint(device=device)
    if C > 0:
        J_t_pt = constraint_pt.compute_tangential_jacobians(
            data["body_trans"], data["contact_points_local"], data["contact_normals_world"], data["contact_mask"]
        )
        res_pt = constraint_pt.get_residuals(
            data["body_vel"], data["lambda_n"], data["lambda_t"], data["gamma"], J_t_pt,
            data["contact_mask"], data["friction_coeff"]
        )
        dres_dbody_vel_pt, dres_dlambda_n_pt, dres_dlambda_t_pt, dres_dgamma_pt = constraint_pt.get_derivatives(
            data["body_vel"], data["lambda_n"], data["lambda_t"], data["gamma"], J_t_pt,
            data["contact_mask"], data["friction_coeff"]
        )
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_runs):
        if C > 0:
            J_t_pt = constraint_pt.compute_tangential_jacobians(
                data["body_trans"], data["contact_points_local"], data["contact_normals_world"], data["contact_mask"]
            )
            res_pt = constraint_pt.get_residuals(
                data["body_vel"], data["lambda_n"], data["lambda_t"], data["gamma"], J_t_pt,
                data["contact_mask"], data["friction_coeff"]
            )
            dres_dbody_vel_pt, dres_dlambda_n_pt, dres_dlambda_t_pt, dres_dgamma_pt = constraint_pt.get_derivatives(
                data["body_vel"], data["lambda_n"], data["lambda_t"], data["gamma"], J_t_pt,
                data["contact_mask"], data["friction_coeff"]
            )
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_runs
    print(f"PyTorch average time per call (All functions): {pytorch_time:.6f} seconds")

    # Warp performance
    wp.init()
    wp_device = wp.get_device()

    # Convert data to Warp arrays
    wp_body_trans = wp.from_torch(swap_quaternion_real_part(data["body_trans"]).squeeze(-1).contiguous(), dtype=wp.transform)
    wp_contact_points = wp.from_torch(data["contact_points_local"].squeeze(-1).contiguous(), dtype=wp.vec3)
    wp_contact_normals = wp.from_torch(data["contact_normals_world"].squeeze(-1).contiguous(), dtype=wp.vec3)
    wp_body_vel = wp.from_torch(data["body_vel"].squeeze(-1).contiguous(), dtype=wp.spatial_vector)
    wp_lambda_n = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C == 0 else wp.from_torch(data["lambda_n"].squeeze(-1).contiguous(), dtype=wp.float32)
    wp_lambda_t = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C == 0 else wp.from_torch(data["lambda_t"].squeeze(-1).contiguous(), dtype=wp.float32)
    wp_gamma = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C == 0 else wp.from_torch(data["gamma"].squeeze(-1).contiguous(), dtype=wp.float32)
    wp_contact_mask = wp.empty(shape=(B, 0), dtype=wp.float32, device=wp_device) if C == 0 else wp.from_torch(data["contact_mask"].float().contiguous(), dtype=wp.float32)
    wp_friction_coeff = wp.from_torch(data["friction_coeff"].squeeze(-1).contiguous(), dtype=wp.float32)

    # Pre-allocate output arrays
    wp_J_t = wp.zeros(shape=(B, 2 * C, 6), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_res = wp.zeros(shape=(B, 3 * C), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_dres_dbody_vel = wp.zeros(shape=(B, 3 * C, 6), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_dres_dlambda_n = wp.zeros(shape=(B, 3 * C, C), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_dres_dlambda_t = wp.zeros(shape=(B, 3 * C, 2 * C), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_dres_dgamma = wp.zeros(shape=(B, 3 * C, C), dtype=wp.float32, device=wp_device, requires_grad=False)

    # Capture the graph
    graph = None
    with wp.ScopedCapture() as capture:
        wp_J_t.zero_()
        wp_res.zero_()
        wp_dres_dbody_vel.zero_()
        wp_dres_dlambda_n.zero_()
        wp_dres_dlambda_t.zero_()
        wp_dres_dgamma.zero_()

        if C > 0:
            t1, t2 = compute_tangential_basis_warp(wp_contact_normals, wp_device)
            wp.launch(
                kernel=compute_tangential_jacobians_kernel,
                dim=(B, C),
                inputs=[wp_body_trans, wp_contact_points, t1, t2, wp_contact_mask, wp_J_t],
                device=wp_device
            )
            wp.launch(
                kernel=get_friction_residuals_kernel,
                dim=(B, C),
                inputs=[
                    wp_body_vel, wp_lambda_n, wp_lambda_t, wp_gamma, wp_J_t, wp_contact_mask,
                    wp_friction_coeff, data["fb_alpha"], data["fb_beta"], data["fb_epsilon"], wp_res
                ],
                device=wp_device
            )
            wp.launch(
                kernel=get_friction_derivatives_kernel,
                dim=(B, C),
                inputs=[
                    wp_body_vel, wp_lambda_n, wp_lambda_t, wp_gamma, wp_J_t, wp_contact_mask,
                    wp_friction_coeff, data["fb_alpha"], data["fb_beta"], data["fb_epsilon"],
                    wp_dres_dbody_vel, wp_dres_dlambda_n, wp_dres_dlambda_t, wp_dres_dgamma
                ],
                device=wp_device
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
    print(f"Warp with Graph (execution only) average time per call (All functions): {warp_graph_execution_time:.6f} seconds")

    if warp_graph_execution_time > 1e-9:
        print(f"Warp is {pytorch_time / warp_graph_execution_time:.2f}x faster than PyTorch for this test case.")
    else:
        print("Warp time is negligible.")

# --- Main execution block ---

if __name__ == "__main__":
    wp.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Comparison Tests
    run_friction_comparison_test("Test Case: No Contacts", B=5, C=0, device=device)
    run_friction_comparison_test("Test Case: Single Contact", B=1, C=1, device=device)
    run_friction_comparison_test("Test Case: Multiple Contacts per Body", B=2, C=3, device=device)
    run_friction_comparison_test("Test Case: Larger Scale", B=10, C=8, device=device)

    # Performance Benchmark
    run_friction_performance_benchmark(B=20, C=16, device=device, num_runs=100)

    print("\nAll Friction Constraint tests finished.")