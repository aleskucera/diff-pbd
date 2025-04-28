import time
from typing import Tuple

import warp as wp
import torch
from jaxtyping import Float
from pbd_torch.model import Model  # Assuming Model is defined elsewhere
from pbd_torch.constraints import RevoluteConstraint
from pbd_torch.utils import swap_quaternion_real_part

# --- Helper Functions ---

def skew_symmetric_matrix_batch(
        vectors: Float[torch.Tensor, "D 3 1"]
) -> Float[torch.Tensor, "D 3 3"]:
    """
    Compute the skew-symmetric matrix of a batch of vectors.

    Args:
        vectors: Tensor of shape [D, 3, 1].

    Returns:
        Tensor of shape [D, 3, 3] with skew-symmetric matrices.
    """
    D = vectors.shape[0]
    skew = torch.zeros(D, 3, 3, device=vectors.device, dtype=vectors.dtype)
    skew[..., 0, 1] = -vectors[..., 2, 0]
    skew[..., 0, 2] = vectors[..., 1, 0]
    skew[..., 1, 0] = vectors[..., 2, 0]
    skew[..., 1, 2] = -vectors[..., 0, 0]
    skew[..., 2, 0] = -vectors[..., 1, 0]
    skew[..., 2, 1] = vectors[..., 0, 0]
    return skew

# --- Warp Kernels ---

@wp.kernel
def compute_joint_jacobians_kernel(
    body_trans: wp.array(dtype=wp.transform),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    joint_trans_parent: wp.array(dtype=wp.transform),
    joint_trans_child: wp.array(dtype=wp.transform),
    J_p: wp.array(dtype=wp.float32, ndim=3),  # [D, 5, 6]
    J_c: wp.array(dtype=wp.float32, ndim=3)   # [D, 5, 6]
):
    d = wp.tid()
    parent = joint_parent[d]
    child = joint_child[d]

    X_p = wp.transform_multiply(body_trans[parent], joint_trans_parent[d])

    X_p_pos = wp.transform_get_translation(X_p)
    X_p_rot = wp.transform_get_rotation(X_p)

    # Compute X_c
    X_c = wp.transform_multiply(body_trans[child], joint_trans_child[d])
    X_c_pos = wp.transform_get_translation(X_c)
    X_c_rot = wp.transform_get_rotation(X_c)

    # Compute r_p and r_c
    r_p_vec = X_p_pos - wp.transform_get_translation(body_trans[parent])
    r_c_vec = X_c_pos - wp.transform_get_translation(body_trans[child])

    # Translational Jacobians
    # Jt_p[:, :, 0:3] = skew(r_p), Jt_p[:, :, 3:6] = -eye(3)
    J_p[d, 0, 0] = 0.0
    J_p[d, 0, 1] = -r_p_vec[2]
    J_p[d, 0, 2] = r_p_vec[1]
    J_p[d, 1, 0] = r_p_vec[2]
    J_p[d, 1, 1] = 0.0
    J_p[d, 1, 2] = -r_p_vec[0]
    J_p[d, 2, 0] = -r_p_vec[1]
    J_p[d, 2, 1] = r_p_vec[0]
    J_p[d, 2, 2] = 0.0
    J_p[d, 0, 3] = -1.0
    J_p[d, 0, 4] = 0.0
    J_p[d, 0, 5] = 0.0
    J_p[d, 1, 3] = 0.0
    J_p[d, 1, 4] = -1.0
    J_p[d, 1, 5] = 0.0
    J_p[d, 2, 3] = 0.0
    J_p[d, 2, 4] = 0.0
    J_p[d, 2, 5] = -1.0

    # Jt_c[:, :, 0:3] = -skew(r_c), Jt_c[:, :, 3:6] = eye(3)
    J_c[d, 0, 0] = 0.0
    J_c[d, 0, 1] = r_c_vec[2]
    J_c[d, 0, 2] = -r_c_vec[1]
    J_c[d, 1, 0] = -r_c_vec[2]
    J_c[d, 1, 1] = 0.0
    J_c[d, 1, 2] = r_c_vec[0]
    J_c[d, 2, 0] = r_c_vec[1]
    J_c[d, 2, 1] = -r_c_vec[0]
    J_c[d, 2, 2] = 0.0
    J_c[d, 0, 3] = 1.0
    J_c[d, 0, 4] = 0.0
    J_c[d, 0, 5] = 0.0
    J_c[d, 1, 3] = 0.0
    J_c[d, 1, 4] = 1.0
    J_c[d, 1, 5] = 0.0
    J_c[d, 2, 3] = 0.0
    J_c[d, 2, 4] = 0.0
    J_c[d, 2, 5] = 1.0

    # Rotational Jacobians
    x_axis = wp.vec3(1.0, 0.0, 0.0)
    y_axis = wp.vec3(0.0, 1.0, 0.0)
    z_axis = wp.vec3(0.0, 0.0, 1.0)
    x_axis_c = wp.quat_rotate(X_c_rot, x_axis)
    y_axis_c = wp.quat_rotate(X_c_rot, y_axis)
    z_axis_p = wp.quat_rotate(X_p_rot, z_axis)
    x_c_cross_z_p = wp.cross(x_axis_c, z_axis_p)
    y_c_cross_z_p = wp.cross(y_axis_c, z_axis_p)

    # Jr_p[d, 0, :] = [-x_c_cross_z_p, 0]
    J_p[d, 3, 0] = -x_c_cross_z_p[0]
    J_p[d, 3, 1] = -x_c_cross_z_p[1]
    J_p[d, 3, 2] = -x_c_cross_z_p[2]
    J_p[d, 3, 3] = 0.0
    J_p[d, 3, 4] = 0.0
    J_p[d, 3, 5] = 0.0
    # Jr_p[d, 1, :] = [-y_c_cross_z_p, 0]
    J_p[d, 4, 0] = -y_c_cross_z_p[0]
    J_p[d, 4, 1] = -y_c_cross_z_p[1]
    J_p[d, 4, 2] = -y_c_cross_z_p[2]
    J_p[d, 4, 3] = 0.0
    J_p[d, 4, 4] = 0.0
    J_p[d, 4, 5] = 0.0

    # Jr_c[d, 0, :] = [x_c_cross_z_p, 0]
    J_c[d, 3, 0] = x_c_cross_z_p[0]
    J_c[d, 3, 1] = x_c_cross_z_p[1]
    J_c[d, 3, 2] = x_c_cross_z_p[2]
    J_c[d, 3, 3] = 0.0
    J_c[d, 3, 4] = 0.0
    J_c[d, 3, 5] = 0.0
    # Jr_c[d, 1, :] = [y_c_cross_z_p, 0]
    J_c[d, 4, 0] = y_c_cross_z_p[0]
    J_c[d, 4, 1] = y_c_cross_z_p[1]
    J_c[d, 4, 2] = y_c_cross_z_p[2]
    J_c[d, 4, 3] = 0.0
    J_c[d, 4, 4] = 0.0
    J_c[d, 4, 5] = 0.0

@wp.kernel
def compute_joint_residuals_kernel(
    body_vel: wp.array(dtype=wp.spatial_vector, ndim=1),  # [B, 6]
    J_p: wp.array(dtype=wp.float32, ndim=3),  # [D, 5, 6]
    J_c: wp.array(dtype=wp.float32, ndim=3),  # [D, 5, 6]
    body_trans: wp.array(dtype=wp.transform),
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    joint_trans_parent: wp.array(dtype=wp.transform),
    joint_trans_child: wp.array(dtype=wp.transform),
    dt: wp.float32,
    stabilization_factor: wp.float32,
    res: wp.array(dtype=wp.float32, ndim=2)  # [D, 5]
):
    d, k = wp.tid()  # d=0 to D-1, k=0 to 4

    parent = joint_parent[d]
    child = joint_child[d]

    # Compute X_p
    X_p = wp.transform_multiply(body_trans[parent], joint_trans_parent[d])
    X_p_pos = wp.transform_get_translation(X_p)
    X_p_rot = wp.transform_get_rotation(X_p)

    # Compute X_c
    X_c = wp.transform_multiply(body_trans[child], joint_trans_child[d])
    X_c_pos = wp.transform_get_translation(X_c)
    X_c_rot = wp.transform_get_rotation(X_c)

    # Compute errors (only needed components for 'k')
    err_tx = X_c_pos[0] - X_p_pos[0]
    err_ty = X_c_pos[1] - X_p_pos[1]
    err_tz = X_c_pos[2] - X_p_pos[2]

    x_axis = wp.vec3(1.0, 0.0, 0.0)
    y_axis = wp.vec3(0.0, 1.0, 0.0)
    z_axis = wp.vec3(0.0, 0.0, 1.0)
    x_axis_c = wp.quat_rotate(X_c_rot, x_axis)
    y_axis_c = wp.quat_rotate(X_c_rot, y_axis)
    z_axis_p = wp.quat_rotate(X_p_rot, z_axis)
    err_rx = wp.dot(x_axis_c, z_axis_p)
    err_ry = wp.dot(y_axis_c, z_axis_p)

    body_vel_p = body_vel[parent]  # [6]
    body_vel_c = body_vel[child]  # [6]

    v_j_p_k = 0.0
    v_j_c_k = 0.0
    # Unroll the loop over j from 0 to 5
    for j in range(wp.static(6)):
        v_j_p_k += J_p[d, k, wp.static(j)] * body_vel_p[wp.static(j)]
        v_j_c_k += J_c[d, k, wp.static(j)] * body_vel_c[wp.static(j)]


    bias_k = 0.0
    if k == 0:
        bias_k = (stabilization_factor / dt) * err_tx
    elif k == 1:
        bias_k = (stabilization_factor / dt) * err_ty
    elif k == 2:
        bias_k = (stabilization_factor / dt) * err_tz
    elif k == 3:
        bias_k = (stabilization_factor / dt) * err_rx
    elif k == 4:
        bias_k = (stabilization_factor / dt) * err_ry

    res[d, k] = (v_j_p_k + v_j_c_k) + bias_k


def create_residuals_kernel(k_value):
    @wp.kernel
    def compute_joint_residuals_for_k(
            body_vel: wp.array(dtype=wp.spatial_vector, ndim=1),
            J_p: wp.array(dtype=wp.float32, ndim=3),
            J_c: wp.array(dtype=wp.float32, ndim=3),
            body_trans: wp.array(dtype=wp.transform),
            joint_parent: wp.array(dtype=wp.int32),
            joint_child: wp.array(dtype=wp.int32),
            joint_trans_parent: wp.array(dtype=wp.transform),
            joint_trans_child: wp.array(dtype=wp.transform),
            dt: wp.float32,
            stabilization_factor: wp.float32,
            res: wp.array(dtype=wp.float32, ndim=2)
    ):
        d = wp.tid()
        k = wp.static(k_value)  # Static k value

        parent = joint_parent[d]
        child = joint_child[d]

        # Compute X_p
        X_p = wp.transform_multiply(body_trans[parent], joint_trans_parent[d])
        X_p_pos = wp.transform_get_translation(X_p)
        X_p_rot = wp.transform_get_rotation(X_p)

        # Compute X_c
        X_c = wp.transform_multiply(body_trans[child], joint_trans_child[d])
        X_c_pos = wp.transform_get_translation(X_c)
        X_c_rot = wp.transform_get_rotation(X_c)

        # Compute errors (only needed for this specific k)
        bias_k = 0.0
        bias_factor = stabilization_factor / dt

        # Only compute the error needed for this specific k
        if wp.static(k == 0):
            err_tx = X_c_pos[0] - X_p_pos[0]
            bias_k = bias_factor * err_tx
        elif wp.static(k == 1):
            err_ty = X_c_pos[1] - X_p_pos[1]
            bias_k = bias_factor * err_ty
        elif wp.static(k == 2):
            err_tz = X_c_pos[2] - X_p_pos[2]
            bias_k = bias_factor * err_tz
        elif wp.static(k == 3):
            x_axis = wp.vec3(1.0, 0.0, 0.0)
            z_axis = wp.vec3(0.0, 0.0, 1.0)
            x_axis_c = wp.quat_rotate(X_c_rot, x_axis)
            z_axis_p = wp.quat_rotate(X_p_rot, z_axis)
            err_rx = wp.dot(x_axis_c, z_axis_p)
            bias_k = bias_factor * err_rx
        elif wp.static(k == 4):
            y_axis = wp.vec3(0.0, 1.0, 0.0)
            z_axis = wp.vec3(0.0, 0.0, 1.0)
            y_axis_c = wp.quat_rotate(X_c_rot, y_axis)
            z_axis_p = wp.quat_rotate(X_p_rot, z_axis)
            err_ry = wp.dot(y_axis_c, z_axis_p)
            bias_k = bias_factor * err_ry

        body_vel_p = body_vel[parent]
        body_vel_c = body_vel[child]

        # Unroll the loop for computing v_j_p_k and v_j_c_k
        v_j_p_k = 0.0
        v_j_c_k = 0.0

        for j in range(wp.static(6)):
            v_j_p_k += J_p[d, k, wp.static(j)] * body_vel_p[wp.static(j)]
            v_j_c_k += J_c[d, k, wp.static(j)] * body_vel_c[wp.static(j)]

        res[d, k] = (v_j_p_k + v_j_c_k) + bias_k

    return compute_joint_residuals_for_k


@wp.kernel
def compute_joint_derivatives_kernel(
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    J_p: wp.array(dtype=wp.float32, ndim=3),  # [D, 5, 6]
    J_c: wp.array(dtype=wp.float32, ndim=3),  # [D, 5, 6]
    dres_dbody_vel: wp.array(dtype=wp.float32, ndim=2)  # [5D, 6B]
):
    d, k, j = wp.tid()  # d=0 to D-1, k=0 to 4, j=0 to 5
    parent = joint_parent[d]
    child = joint_child[d]
    row = 5 * d + k
    col_p = 6 * parent + j
    col_c = 6 * child + j
    dres_dbody_vel[row, col_p] = J_p[d, k, j]
    dres_dbody_vel[row, col_c] = J_c[d, k, j]

# --- Warp Functions ---

def compute_joint_jacobians_warp(
    body_trans: wp.array,
    joint_parent: wp.array,
    joint_child: wp.array,
    joint_trans_parent: wp.array,
    joint_trans_child: wp.array,
    device: wp.context.Device
) -> Tuple[wp.array, wp.array]:
    D = joint_parent.shape[0]
    J_p = wp.zeros(shape=(D, 5, 6), dtype=wp.float32, device=device)
    J_c = wp.zeros(shape=(D, 5, 6), dtype=wp.float32, device=device)
    wp.launch(
        kernel=compute_joint_jacobians_kernel,
        dim=D,
        inputs=[
            body_trans, joint_parent, joint_child,
            joint_trans_parent, joint_trans_child,
            J_p, J_c
        ],
        device=device
    )
    return J_p, J_c

def compute_joint_residuals_warp(
    body_vel: wp.array,
    J_p: wp.array,
    J_c: wp.array,
    body_trans: wp.array,
    joint_parent: wp.array,
    joint_child: wp.array,
    joint_trans_parent: wp.array,
    joint_trans_child: wp.array,
    dt: float,
    stabilization_factor: float,
    device: wp.context.Device
) -> wp.array:
    D = joint_parent.shape[0]
    res = wp.zeros(shape=(D, 5), dtype=wp.float32, device=device)
    wp.launch(
        kernel=compute_joint_residuals_kernel,
        dim=(D, 5),
        inputs=[
            body_vel, J_p, J_c, body_trans, joint_parent, joint_child,
            joint_trans_parent, joint_trans_child,
            dt, stabilization_factor, res
        ],
        device=device
    )
    return res

def compute_joint_residuals_warp2(
        body_vel: wp.array,
        J_p: wp.array,
        J_c: wp.array,
        body_trans: wp.array,
        joint_parent: wp.array,
        joint_child: wp.array,
        joint_trans_parent: wp.array,
        joint_trans_child: wp.array,
        dt: float,
        stabilization_factor: float,
        device: wp.context.Device
) -> wp.array:
    D = joint_parent.shape[0]
    res = wp.zeros(shape=(D, 5), dtype=wp.float32, device=device)

    joint_kernels = [create_residuals_kernel(k) for k in range(5)]
    for k in range(5):
        wp.launch(
            kernel=joint_kernels[k],
            dim=(D,),
            inputs=[
                body_vel, J_p, J_c, body_trans, joint_parent, joint_child,
                joint_trans_parent, joint_trans_child,
                dt, stabilization_factor, res
            ],
            device=device
        )
    return res

def compute_joint_derivatives_warp(
    joint_parent: wp.array,
    joint_child: wp.array,
    J_p: wp.array,
    J_c: wp.array,
    B: int,
    device: wp.context.Device
) -> wp.array:
    D = joint_parent.shape[0]
    dres_dbody_vel = wp.zeros(shape=(5 * D, 6 * B), dtype=wp.float32, device=device)
    if D > 0:
        wp.launch(
            kernel=compute_joint_derivatives_kernel,
            dim=(D, 5, 6),
            inputs=[joint_parent, joint_child, J_p, J_c, dres_dbody_vel],
            device=device
        )
    return dres_dbody_vel

# --- Test Data Setup ---
def setup_joint_test_case_data(B: int, D: int, device: torch.device):
    """Generates random data for a joint constraint test case."""
    if D > 0 and B < 2:
        raise ValueError("Cannot create joints with B < 2")

    model = Model() # Placeholder
    model.device = device

    # Generate random parent and child indices, ensuring they are different
    if D > 0:
        model.joint_parent = torch.randint(0, B, (D,), device=device, dtype=torch.int32)
        model.joint_child = torch.randint(0, B, (D,), device=device, dtype=torch.int32)
        # Ensure no parent-child self-loops
        while torch.any(model.joint_parent == model.joint_child):
            model.joint_child = torch.randint(0, B, (D,), device=device, dtype=torch.int32)

        # Generate random joint transforms (pos + quat [w, x, y, z])
        model.joint_X_p = torch.randn(D, 7, 1, device=device)
        model.joint_X_c = torch.randn(D, 7, 1, device=device)

        # Normalize quaternions for joint transforms
        joint_p_quats = model.joint_X_p[:, 3:, 0] # Shape [D, 4]
        joint_p_quat_norms = torch.linalg.norm(joint_p_quats, dim=-1, keepdim=True) # Shape [D, 1]
        joint_p_quat_norms[joint_p_quat_norms == 0] = 1.0
        model.joint_X_p[:, 3:, 0] = joint_p_quats / joint_p_quat_norms
        model.joint_X_p[:, 3:, :] = model.joint_X_p[:, 3:, 0].unsqueeze(-1) # Restore shape

        joint_c_quats = model.joint_X_c[:, 3:, 0] # Shape [D, 4]
        joint_c_quat_norms = torch.linalg.norm(joint_c_quats, dim=-1, keepdim=True) # Shape [D, 1]
        joint_c_quat_norms[joint_c_quat_norms == 0] = 1.0
        model.joint_X_c[:, 3:, 0] = joint_c_quats / joint_c_quat_norms
        model.joint_X_c[:, 3:, :] = model.joint_X_c[:, 3:, 0].unsqueeze(-1) # Restore shape

    else:
        model.joint_parent = torch.empty(0, dtype=torch.int32, device=device)
        model.joint_child = torch.empty(0, dtype=torch.int32, device=device)
        model.joint_X_p = torch.empty(0, 7, 1, device=device)
        model.joint_X_c = torch.empty(0, 7, 1, device=device)


    # Generate random body transforms and velocities
    body_trans = torch.randn(B, 7, 1, device=device)
    body_vel = torch.randn(B, 6, 1, device=device)

    # Normalize body_trans quaternions (wxyz)
    if B > 0:
        body_trans_quats = body_trans[:, 3:, 0] # Shape [B, 4]
        body_trans_quat_norms = torch.linalg.norm(body_trans_quats, dim=-1, keepdim=True) # Shape [B, 1]
        body_trans_quat_norms[body_trans_quat_norms == 0] = 1.0
        body_trans[:, 3:, 0] = body_trans_quats / body_trans_quat_norms
        body_trans[:, 3:, :] = body_trans[:, 3:, 0].unsqueeze(-1) # Restore shape
    dt = 0.01

    # Set requires_grad=False for inputs if you are only testing analytical derivatives
    body_trans.requires_grad_(False)
    model.joint_X_p.requires_grad_(False)
    model.joint_X_c.requires_grad_(False)
    body_vel.requires_grad_(False) # Derivatives w.r.t. body_vel is in dynamics_constraint

    return model, body_trans, body_vel, dt


# --- Comparison Test ---
def run_joint_comparison_test(test_name: str, B: int, D: int, device: torch.device):
    print(f"\nRunning Joint Constraint Comparison Test '{test_name}' (B={B}, D={D})")
    model, body_trans, body_vel, dt = setup_joint_test_case_data(B, D, device)
    wp_device = wp.get_device()

    # --- PyTorch computations ---
    constraint = RevoluteConstraint(model)
    if D > 0:
        J_p_pt, J_c_pt = constraint.compute_jacobians(body_trans)
        res_pt = constraint.get_residuals(body_vel, body_trans, J_p_pt, J_c_pt, dt)
        dres_dbody_vel_pt = constraint.get_derivatives(body_vel, J_p_pt, J_c_pt)
    else:
        # Handle case with no joints - results should be empty tensors of correct shape
        J_p_pt = torch.empty(0, 5, 6, device=device)
        J_c_pt = torch.empty(0, 5, 6, device=device)
        res_pt = torch.empty(0, 1, device=device)
        dres_dbody_vel_pt = torch.empty(0, 6 * B, device=device)

    # --- Warp computations ---
    # Convert data to Warp arrays
    wp_body_trans = wp.from_torch(swap_quaternion_real_part(body_trans).squeeze(-1), dtype=wp.transform)
    wp_joint_parent = wp.from_torch(model.joint_parent, dtype=wp.int32)
    wp_joint_child = wp.from_torch(model.joint_child, dtype=wp.int32)
    wp_joint_trans_parent = wp.from_torch(swap_quaternion_real_part(model.joint_X_p).squeeze(-1), dtype=wp.transform)
    wp_joint_trans_child = wp.from_torch(swap_quaternion_real_part(model.joint_X_c).squeeze(-1), dtype=wp.transform)
    wp_body_vel = wp.from_torch(body_vel.view(B, 6), dtype=wp.spatial_vector) # View [B, 6] for wp.spatial_vector

    if D > 0:
        J_p_wp, J_c_wp = compute_joint_jacobians_warp(
            wp_body_trans, wp_joint_parent, wp_joint_child,
            wp_joint_trans_parent, wp_joint_trans_child,
            device=wp_device
        )
        res_wp = compute_joint_residuals_warp(
            wp_body_vel, J_p_wp, J_c_wp, wp_body_trans, wp_joint_parent, wp_joint_child,
            wp_joint_trans_parent, wp_joint_trans_child,
            dt, constraint.stabilization_factor, device=wp_device
        )
        dres_dbody_vel_wp = compute_joint_derivatives_warp(
            wp_joint_parent, wp_joint_child, J_p_wp, J_c_wp, B, device=wp_device
        )
    else:
        # Handle case with no joints in Warp
        J_p_wp = wp.empty(shape=(0, 5, 6), dtype=wp.float32, device=wp_device)
        J_c_wp = wp.empty(shape=(0, 5, 6), dtype=wp.float32, device=wp_device)
        res_wp = wp.empty(shape=(0, 5), dtype=wp.float32, device=wp_device)
        dres_dbody_vel_wp = wp.empty(shape=(0, 6 * B), dtype=wp.float32, device=wp_device)


    # Convert Warp results back to PyTorch
    J_p_wp_torch = wp.to_torch(J_p_wp).clone()
    J_c_wp_torch = wp.to_torch(J_c_wp).clone()
    res_wp_torch = wp.to_torch(res_wp).clone().view(res_pt.shape) # Reshape to match PT output [5*D, 1]
    dres_dbody_vel_wp_torch = wp.to_torch(dres_dbody_vel_wp).clone()


    # Compare
    print(f"Comparing results for '{test_name}'")
    assert torch.allclose(J_p_pt, J_p_wp_torch, atol=1e-5), f"{test_name}: J_p mismatch"
    assert torch.allclose(J_c_pt, J_c_wp_torch, atol=1e-5), f"{test_name}: J_c mismatch"
    assert torch.allclose(res_pt, res_wp_torch, atol=1e-5), f"{test_name}: Residuals mismatch"

    # Derivatives comparison needs careful indexing for the sparse dres_dbody_vel matrix
    # The PyTorch get_derivatives produces a sparse-like matrix [5*D, 6*B]
    # The Warp kernel also fills a [5*D, 6*B] matrix.
    # Direct comparison should work if the indices match.
    assert torch.allclose(dres_dbody_vel_pt, dres_dbody_vel_wp_torch, atol=1e-5), f"{test_name}: Derivatives mismatch"

    print(f"{test_name}: All comparisons passed.")


# --- Performance Benchmark ---
def run_joint_performance_benchmark(B: int, D: int, device: torch.device, num_runs: int = 100):
    """Runs a performance benchmark for PyTorch and Warp (with graph) for joint constraints."""
    print(f"\nRunning Joint Constraint Performance Test (B={B}, D={D})")
    if D > 0 and B < 2:
        print("Skipping benchmark: Cannot create joints with B < 2")
        return

    model, body_trans, body_vel, dt = setup_joint_test_case_data(B, D, device)

    # PyTorch performance (Jacobians, Residuals, Derivatives)
    constraint = RevoluteConstraint(model)
    # Warm-up
    if D > 0:
        J_p_pt, J_c_pt = constraint.compute_jacobians(body_trans)
        res_pt = constraint.get_residuals(body_vel, body_trans, J_p_pt, J_c_pt, dt)
        dres_dbody_vel_pt = constraint.get_derivatives(body_vel, J_p_pt, J_c_pt)
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_runs):
        if D > 0:
            J_p_pt, J_c_pt = constraint.compute_jacobians(body_trans)
            res_pt = constraint.get_residuals(body_vel, body_trans, J_p_pt, J_c_pt, dt)
            dres_dbody_vel_pt = constraint.get_derivatives(body_vel, J_p_pt, J_c_pt)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_runs
    print(f"PyTorch average time per call (Jacobians + Residuals + Derivatives): {pytorch_time:.6f} seconds")

    # Convert data to Warp arrays for graph capture
    wp.init()
    wp_device = wp.get_device()
    wp_body_trans = wp.from_torch(swap_quaternion_real_part(body_trans).squeeze(-1).contiguous(), dtype=wp.transform)
    wp_joint_parent = wp.from_torch(model.joint_parent.contiguous(), dtype=wp.int32)
    wp_joint_child = wp.from_torch(model.joint_child.contiguous(), dtype=wp.int32)
    wp_joint_trans_parent = wp.from_torch(swap_quaternion_real_part(model.joint_X_p).squeeze(-1).contiguous(), dtype=wp.transform)
    wp_joint_trans_child = wp.from_torch(swap_quaternion_real_part(model.joint_X_c).squeeze(-1).contiguous(), dtype=wp.transform)
    wp_body_vel = wp.from_torch(body_vel.view(B, 6).contiguous(), dtype=wp.spatial_vector)

    # Pre-allocate output arrays for Warp
    D_actual = model.joint_parent.shape[0]
    wp_J_p = wp.zeros(shape=(D_actual, 5, 6), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_J_c = wp.zeros(shape=(D_actual, 5, 6), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_res = wp.zeros(shape=(D_actual, 5), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_dres_dbody_vel = wp.zeros(shape=(5 * D_actual, 6 * B), dtype=wp.float32, device=wp_device, requires_grad=False)

    # Pre-allocate output arrays for Warp
    D_actual = model.joint_parent.shape[0]
    wp_J_p = wp.zeros(shape=(D_actual, 5, 6), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_J_c = wp.zeros(shape=(D_actual, 5, 6), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_res = wp.zeros(shape=(D_actual, 5), dtype=wp.float32, device=wp_device, requires_grad=False)
    wp_dres_dbody_vel = wp.zeros(shape=(5 * D_actual, 6 * B), dtype=wp.float32, device=wp_device, requires_grad=False)

    # Capture the graph
    graph = None
    joint_residual_kernels = [create_residuals_kernel(k) for k in range(5)]
    with wp.ScopedCapture() as capture:
        # Clear output tensors before use in the captured graph
        wp_J_p.zero_()
        wp_J_c.zero_()
        wp_res.zero_()
        wp_dres_dbody_vel.zero_()

        if D_actual > 0:
            # Launch Jacobian kernel using wp.launch
            wp.launch(
                kernel=compute_joint_jacobians_kernel,
                dim=D_actual,
                inputs=[
                    wp_body_trans, wp_joint_parent, wp_joint_child,
                    wp_joint_trans_parent, wp_joint_trans_child,
                    wp_J_p, wp_J_c
                ],
                device=wp_device
            )
            # # Launch Residual kernel using wp.launch
            # for k in range(5):
            #     wp.launch(
            #         kernel=joint_residual_kernels[k],
            #         dim=(D_actual,),
            #         inputs=[
            #             wp_body_vel, wp_J_p, wp_J_c, wp_body_trans, wp_joint_parent, wp_joint_child,
            #             wp_joint_trans_parent, wp_joint_trans_child,
            #             dt, constraint.stabilization_factor, wp_res
            #         ],
            #         device=wp_device
            #     )

            wp.launch(
                kernel=compute_joint_residuals_kernel,
                dim=(D_actual, 5),
                inputs=[
                    wp_body_vel, wp_J_p, wp_J_c, wp_body_trans, wp_joint_parent, wp_joint_child,
                    wp_joint_trans_parent, wp_joint_trans_child,
                    dt, constraint.stabilization_factor, wp_res
                ],
                device=wp_device
            )
            # Launch Derivative kernel using wp.launch
            wp.launch(
                kernel=compute_joint_derivatives_kernel,
                dim=(D_actual, 5, 6),
                inputs=[wp_joint_parent, wp_joint_child, wp_J_p, wp_J_c, wp_dres_dbody_vel],
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
    print(f"Warp with Graph (execution only) average time per call (Jacobians + Residuals + Derivatives): {warp_graph_execution_time:.6f} seconds")

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
    try:
        run_joint_comparison_test("Test Case: No Joints", B=5, D=0, device=device)
    except ValueError as e:
        print(f"Skipped test due to setup error: {e}")
    try:
        run_joint_comparison_test("Test Case: Single Joint", B=2, D=1, device=device)
    except ValueError as e:
        print(f"Skipped test due to setup error: {e}")
    try:
        run_joint_comparison_test("Test Case: Multiple Joints", B=10, D=5, device=device)
    except ValueError as e:
        print(f"Skipped test due to setup error: {e}")


    # Performance Benchmark
    # Choose a representative scale
    run_joint_performance_benchmark(B=500, D=20, device=device, num_runs=100)

    print("\nAll Joint Constraint tests finished.")