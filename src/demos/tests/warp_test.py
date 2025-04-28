import warp as wp
import torch
import numpy as np


@wp.kernel
def accumulate_joint_impulse_kernel(
    joint_parent: wp.array(dtype=wp.int32, ndim=1),         # [D]
    joint_child: wp.array(dtype=wp.int32, ndim=1),          # [D]
    lambda_j: wp.array(dtype=wp.float32, ndim=2),           # [D, 5] - input impulse values
    J_j_p: wp.array(dtype=wp.spatial_vector, ndim=2),       # [D, 5] - parent Jacobians
    J_j_c: wp.array(dtype=wp.spatial_vector, ndim=2),       # [D, 5] - child Jacobians
    res: wp.array(dtype=wp.spatial_vector, ndim=1),         # Residuals: [B] (atomic add)
):
    # This kernel is launched with dim = D * 5.
    flat_idx = wp.tid()
    D = joint_parent.shape[0]
    impulse_scalar_dim = 5

    # Integer division to get joint index
    d = flat_idx // impulse_scalar_dim
    # Modulo to get impulse component index
    k = flat_idx % impulse_scalar_dim  # k is 0 to 4

    # Get the spatial vector and scale by scalar
    impulse_contrib_p = J_j_p[d, k] * lambda_j[d, k]  # Should work if J_j_p is correctly defined
    impulse_contrib_c = J_j_c[d, k] * lambda_j[d, k]

    # Atomically add the negative contribution to residuals
    wp.atomic_add(res, joint_parent[d], -impulse_contrib_p)
    wp.atomic_add(res, joint_child[d], -impulse_contrib_c)

wp.init()

# Example parameters
D = 10  # Number of joints
B = 20  # Number of bodies
impulse_scalar_dim = 5

# Create example input tensors
joint_parent = np.random.randint(0, B, D)
joint_child = np.random.randint(0, B, D)
lambda_j = np.random.randn(D, impulse_scalar_dim).astype(np.float32)

# Jacobian tensors: [D, 5, 6, 1]
J_j_p_torch = torch.randn(D, impulse_scalar_dim, 6, 1, dtype=torch.float32, device="cuda")
J_j_c_torch = torch.randn(D, impulse_scalar_dim, 6, 1, dtype=torch.float32, device="cuda")

# Reshape to [D, 5, 6] for wp.spatial_vector
J_j_p_torch = J_j_p_torch.squeeze(-1)  # [D, 5, 6]
J_j_c_torch = J_j_c_torch.squeeze(-1)  # [D, 5, 6]

# Convert to Warp arrays
joint_parent_wp = wp.array(joint_parent, dtype=wp.int32, device="cuda")
joint_child_wp = wp.array(joint_child, dtype=wp.int32, device="cuda")
lambda_j_wp = wp.array(lambda_j, dtype=wp.float32, device="cuda")
J_j_p_wp = wp.array(J_j_p_torch, dtype=wp.spatial_vector, device="cuda")
J_j_c_wp = wp.array(J_j_c_torch, dtype=wp.spatial_vector, device="cuda")
res_wp = wp.zeros(B, dtype=wp.spatial_vector, device="cuda")

# Launch kernel
wp.launch(
    kernel=accumulate_joint_impulse_kernel,
    dim=D * impulse_scalar_dim,
    inputs=[joint_parent_wp, joint_child_wp, lambda_j_wp, J_j_p_wp, J_j_c_wp, res_wp],
    device="cuda"
)

# Synchronize and check results
wp.synchronize()
print("Residuals:", res_wp.numpy())