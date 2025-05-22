from typing import Tuple

import torch
from simview import BodyShapeType
from simview import SimulationScene
from simview import SimViewLauncher
from simview import SimViewBodyState
from pbd_torch.transform import transform_multiply_batch, normalize_quat_batch
from pbd_torch.transform import rotate_vectors_batch, quat_inv_batch, quat_mul_batch


def _create_orthogonal_basis(first_axis: torch.Tensor):
    assert len(first_axis.shape) == 1, "Start vector must be a 1D tensor"
    assert first_axis.shape[0] == 3, "Start vector must be a 3D vector"
    assert torch.norm(first_axis) > 0, "Start vector must be non-zero"

    first_axis = first_axis / torch.norm(first_axis)

    # Initialize basis list with the first normalized vector
    basis = [first_axis]

    rand_vector = torch.tensor([0.0, 0.0, -1.0], device=first_axis.device)
    if torch.linalg.cross(first_axis, rand_vector).norm() < 1e-6:
        rand_vector = torch.tensor([0.0, -1.0, 0.0], device=first_axis.device)

    # Generate a vector orthogonal to the first vector
    second_axis = torch.linalg.cross(first_axis, rand_vector)
    basis.append(second_axis)

    # Generate a vector orthogonal to the first two vectors
    third_axis = torch.linalg.cross(first_axis, second_axis)
    basis.append(third_axis / torch.norm(third_axis))

    return torch.stack(basis, dim=1)


def _generate_fibonacci_sphere_points(radius: float, n_points: int):
    indices = torch.arange(n_points, dtype=torch.float32)

    # Golden ratio constant
    phi = (torch.sqrt(torch.tensor(5.0)) - 1) / 2

    # Calculate z coordinates
    z = (2 * indices + 1) / n_points - 1

    # Calculate radius at each z
    radius_at_z = torch.sqrt(1 - z * z)

    # Calculate angles
    theta = 2 * torch.pi * indices * phi

    # Calculate x, y, z coordinates
    x = radius * radius_at_z * torch.cos(theta)
    y = radius * radius_at_z * torch.sin(theta)
    z = radius * z

    # Stack coordinates
    points = torch.stack([x, y, z], dim=1)

    return points


def _generate_fibonacci_surface_points(radius: float, height: float, n_points: int):
    """Generate uniformly distributed points on cylinder surface using Fibonacci spiral."""
    golden_ratio = (1 + torch.sqrt(torch.tensor(5.0))) / 2

    points = []
    for i in range(n_points):
        z = height * (i / (n_points - 1) - 0.5)
        theta = 2 * torch.pi * i / golden_ratio

        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        points.append([x, y, z])

    return torch.tensor(points)


def _generate_fibonacci_disk_points(radius: float, n_points: int):
    """Generate uniformly distributed points on circle using sunflower pattern."""
    golden_angle = torch.pi * (3 - torch.sqrt(torch.tensor(5.0)))

    indices = torch.arange(n_points, dtype=torch.float32)
    r = radius * torch.sqrt(indices / n_points)
    theta = indices * golden_angle

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    return torch.stack([x, y], dim=1)


def sphere_collision_points(
    radius: float, n_points: int, device: torch.device = torch.device("cpu")
):
    return _generate_fibonacci_sphere_points(radius, n_points).to(device)


def cylinder_collision_points(
    radius: float,
    height: float,
    n_base_points: int,
    n_surface_points: int,
    device: torch.device = torch.device("cpu"),
):
    base_points = _generate_fibonacci_disk_points(radius, n_base_points)
    surface_vertices = _generate_fibonacci_surface_points(
        radius, height, n_surface_points
    )

    # Create top and bottom vertices for the bases
    top_vertices = torch.cat(
        [base_points, torch.ones(len(base_points), 1) * height / 2], dim=1
    )

    bottom_vertices = torch.cat(
        [base_points, torch.ones(len(base_points), 1) * -height / 2], dim=1
    )

    # Combine all vertices
    vertices = torch.cat([surface_vertices, top_vertices, bottom_vertices], dim=0).type(
        torch.float32
    )

    return vertices.to(device)


def box_collision_points(
    hx: float,
    hy: float,
    hz: float,
    n_points: int,
    device: torch.device = torch.device("cpu"),
):
    points = []
    points_per_face = n_points // 6

    # Generate points for each face
    # Front and back faces (YZ planes)
    for x in [-hx, hx]:
        y = torch.linspace(-hy, hy, int(torch.sqrt(torch.tensor(points_per_face))))
        z = torch.linspace(-hz, hz, int(torch.sqrt(torch.tensor(points_per_face))))
        Y, Z = torch.meshgrid(y, z, indexing="ij")
        face_points = torch.stack(
            [torch.full_like(Y, x).flatten(), Y.flatten(), Z.flatten()], dim=1
        )
        points.append(face_points)

    # Left and right faces (XZ planes)
    for y in [-hy, hy]:
        x = torch.linspace(-hx, hx, int(torch.sqrt(torch.tensor(points_per_face))))
        z = torch.linspace(-hz, hz, int(torch.sqrt(torch.tensor(points_per_face))))
        X, Z = torch.meshgrid(x, z, indexing="ij")
        face_points = torch.stack(
            [X.flatten(), torch.full_like(X, y).flatten(), Z.flatten()], dim=1
        )
        points.append(face_points)

    # Top and bottom faces (XY planes)
    for z in [-hz, hz]:
        x = torch.linspace(-hx, hx, int(torch.sqrt(torch.tensor(points_per_face))))
        y = torch.linspace(-hy, hy, int(torch.sqrt(torch.tensor(points_per_face))))
        X, Y = torch.meshgrid(x, y, indexing="ij")
        face_points = torch.stack(
            [X.flatten(), Y.flatten(), torch.full_like(X, z).flatten()], dim=1
        )
        points.append(face_points)

    points = torch.cat(points, dim=0)
    points = torch.unique(points, dim=0)

    return points.to(device)


def sphere_inertia(radius: float, m: float, device: torch.device, requires_grad: bool):
    return torch.tensor(
        [
            [2 / 5 * m * radius, 0.0, 0.0],
            [0.0, 2 / 5 * m * radius, 0.0],
            [0.0, 0.0, 2 / 5 * m * radius],
        ],
        device=device,
        requires_grad=requires_grad,
    )


def box_inertia(
    hx: float, hy: float, hz: float, m: float, device: torch.device, requires_grad: bool
):
    return torch.tensor(
        [
            [m / 12 * (hy**2 + hz**2), 0.0, 0.0],
            [0.0, m / 12 * (hx**2 + hz**2), 0.0],
            [0.0, 0.0, m / 12 * (hx**2 + hy**2)],
        ],
        device=device,
        requires_grad=requires_grad,
    )


def cylinder_inertia(
    radius: float, height: float, m: float, device: torch.device, requires_grad: bool
):
    return torch.tensor(
        [
            [m * (3 * radius**2 + height**2) / 12, 0.0, 0.0],
            [0.0, m * (3 * radius**2 + height**2) / 12, 0.0],
            [0.0, 0.0, m * radius**2 / 2],
        ],
        device=device,
        requires_grad=requires_grad,
    )


def forces_from_joint_acts(
        joint_act: torch.Tensor,
        joint_q: torch.Tensor,
        joint_qd: torch.Tensor,
        joint_ke: torch.Tensor,
        joint_kd: torch.Tensor,
        body_trans: torch.Tensor,
        joint_parent: torch.Tensor,
        joint_child: torch.Tensor,
        joint_trans_parent: torch.Tensor,
        joint_trans_child: torch.Tensor,
) -> torch.Tensor:
    device = body_trans.device
    B = body_trans.shape[0]
    D = joint_parent.shape[0]

    body_f = torch.zeros((B, 6, 1), dtype=torch.float32, device=device)

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).repeat(D, 1).unsqueeze(-1) # [D, 3, 1]

    trans_parent = body_trans[joint_parent]  # [D, 7, 1]
    trans_child = body_trans[joint_child]  # [D, 7, 1]

    # Joint frame computed from the parent and child bodies
    X_p = transform_multiply_batch(trans_parent, joint_trans_parent)  # [D, 7, 1]
    X_c = transform_multiply_batch(trans_child, joint_trans_child) # [D, 7, 1]

    z_p = rotate_vectors_batch(z_axis, X_p[:, 3:]) # [D, 3, 1]
    z_c = rotate_vectors_batch(z_axis, X_c[:, 3:]) # [D, 3, 1]

    joint_force = eval_joint_force(
        joint_act,
        joint_q,
        joint_qd,
        joint_ke,
        joint_kd,
        mode=2) # [D, 1]

    torque_p = z_p * joint_force.unsqueeze(1) # [D, 3, 1]
    torque_c = z_c * joint_force.unsqueeze(1) # [D, 3, 1]

    force_p = torch.zeros((D, 6, 1), dtype=torch.float32, device=device)
    force_p[:, :3] = -torque_p # [D, 3, 1]

    force_c = torch.zeros((D, 6, 1), dtype=torch.float32, device=device)
    force_c[:, :3] = torque_c # [D, 3, 1]

    body_f.index_add_(0, joint_parent, force_p)
    body_f.index_add_(0, joint_child, force_c)

    return body_f

def eval_joint_force(
    joint_act: torch.Tensor,
    joint_q: torch.Tensor,
    joint_qd: torch.Tensor,
    ke: torch.Tensor,
    kd: torch.Tensor,
    mode: int
) -> torch.Tensor:
    """Evaluates joint force based on the joint mode.

    Args:
        q (float): Joint position
        qd (float): Joint velocity
        act (float): Joint actuation
        ke (float): Position gain
        kd (float): Velocity gain
        mode (int): Joint mode (FORCE, TARGET_POSITION, or TARGET_VELOCITY)

    Returns:
        float: Computed joint force
    """
    if mode == 0:
        return joint_act
    elif mode == 1:
        return ke * (joint_act - joint_q) - kd * joint_qd
    elif mode == 2:
        return ke * (joint_act - joint_qd)
    else:
        raise ValueError("Invalid joint mode")

def swap_quaternion_real_part(
        transforms: torch.Tensor
) -> torch.Tensor:
    """
    Swaps the real part of the quaternion from the first position to the last
    within transformation tensors of shape [N, 7, 1].

    Args:
        transforms: A torch.Tensor of shape [N, 7, 1], where transforms[:, :3, :]
                    are positions and transforms[:, 3:, :] are quaternions
                    in [w, x, y, z] format.

    Returns:
        A torch.Tensor of the same shape [N, 7, 1], with quaternions in
        [x, y, z, w] format.
    """
    if transforms.shape[-1] != 1 or transforms.shape[-2] != 7:
        raise ValueError("Input tensor must have shape [N, 7, 1]")

    N = transforms.shape[0]
    device = transforms.device
    dtype = transforms.dtype

    # Separate position and quaternion
    positions = transforms[:, :3, :]  # Shape [N, 3, 1]
    quaternions_wxyz = transforms[:, 3:, :] # Shape [N, 4, 1]

    # Swap quaternion components from [w, x, y, z] to [x, y, z, w]
    quaternions_xyzw = torch.cat([quaternions_wxyz[:, 1:, :], quaternions_wxyz[:, :1, :]], dim=1) # Shape [N, 4, 1]

    # Concatenate position and the swapped quaternion
    transformed_tensor = torch.cat([positions, quaternions_xyzw], dim=1) # Shape [N, 7, 1]

    return transformed_tensor

def compute_joint_coordinates(
    body_q: torch.Tensor,
    body_qd: torch.Tensor,
    joint_parent: torch.Tensor,
    joint_child: torch.Tensor,
    joint_X_p: torch.Tensor,
    joint_X_c: torch.Tensor,
    joint_axis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    q_parent = body_q[joint_parent]  # [D, 7, 1]
    q_child = body_q[joint_child]  # [D, 7, 1]

    # Joint frames computed from the parent and child bodies
    X_p_w = transform_multiply_batch(q_parent, joint_X_p)  # [D, 7, 1]
    X_c_w = transform_multiply_batch(q_child, joint_X_c)  # [D, 7, 1]

    rot_p = X_p_w[:, 3:] # [D, 4, 1]
    rot_c = X_c_w[:, 3:] # [D, 4, 1]

    rot_p_inv = quat_inv_batch(rot_p) # [D, 4, 1]
    rot_rel = quat_mul_batch(rot_p_inv, rot_c) # [D, 4, 1]
    rot_rel = normalize_quat_batch(rot_rel) # [D, 4, 1]
    joint_angle = 2 * torch.atan2(rot_rel[:, 3], rot_rel[:, 0]) # [D]
    joint_q = joint_angle.view(-1, 1) # [D, 1]

    # Compute joint velocities
    omega_p = body_qd[joint_parent, :3] # [D, 3, 1]
    omega_c = body_qd[joint_child, :3] # [D, 3, 1]

    joint_axis_w = rotate_vectors_batch(joint_axis, rot_p) # [D, 3, 1]
    rel_omega = omega_c - omega_p # [D, 3, 1]
    joint_omega = torch.matmul(joint_axis_w.transpose(2, 1), rel_omega) # [D, 1, 1]
    joint_qd = joint_omega.view(-1, 1) # [D, 1]

    return joint_q, joint_qd

def add_state_to_scene(
    sim_scene: SimulationScene,
    model,
    state,
    time: float,
    scalar_values: dict[str, torch.Tensor],
) -> None:
    """
    Adds a state to the simulation scene.

    Args:
        sim_scene (SimulationScene): The simulation scene to which the state will be added.
        body_states (list[SimViewBodyState]): List of body states to add.
        scalar_values (dict[str, torch.Tensor]): Dictionary of scalar values to add.
        time (float): The time at which the state is recorded.
    """
    body_states = []
    for body in range(model.num_bodies):
        body_states.appen(SimViewBodyState(
            body_name=model.body_name[body],
            position=state.body_q[body, :3].flatten(),
            orientation=state.body_q[body, 3:].flatten(),
        ))

    sim_scene.add_state(
        time=time,
        body_states=body_states,
        scalar_values=scalar_values,
    )

def create_scene_from_model(
        model,
        batch_size: int,
        scalar_names: list[str] = None,
):
 pass