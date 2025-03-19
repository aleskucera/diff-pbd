import torch


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
