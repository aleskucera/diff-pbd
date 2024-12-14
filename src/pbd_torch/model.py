from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import NewType

import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from pbd_torch.transform import *

matplotlib.use('TkAgg')

# Define new types for the simulation
Vector3 = NewType('Vector3', torch.Tensor)  # (3, ) tensor
Transform = NewType('Transform', torch.Tensor)  # (7, ) tensor
Matrix3x3 = NewType('Matrix3x3', torch.Tensor)  # (3, 3) tensor
Quaternion = NewType('Quaternion', torch.Tensor)  # (4, ) tensor

BOX = 0
SPHERE = 1
CYLINDER = 2

JOINT_MODE_FORCE = 0
JOINT_MODE_TARGET_POSITION = 1
JOINT_MODE_TARGET_VELOCITY = 2


@dataclass
class BodyVector:
    r: torch.Tensor
    body_q: torch.Tensor

    def to_world(self):
        return transform(self.r, self.body_q)


@dataclass
class WorldVector:
    v: torch.Tensor

    def to_body(self, body_q: torch.Tensor):
        return rotate_vectors(self.v, quat_inv(body_q))


@dataclass
class Shape:
    type: int

    def serialize(self):
        return asdict(self)


@dataclass
class Box(Shape):
    hx: float
    hy: float
    hz: float
    type: int = field(init=False, default=BOX)


@dataclass
class Sphere(Shape):
    radius: float
    type: int = field(init=False, default=SPHERE)


@dataclass
class Cylinder(Shape):
    radius: float
    height: float
    type: int = field(init=False, default=CYLINDER)


class State:

    def __init__(self):
        self.time = 0.0

        # Body-related attributes
        self.body_q = torch.zeros((0, 7), dtype=torch.float32)
        self.body_qd = torch.zeros((0, 6), dtype=torch.float32)
        self.body_f = torch.zeros((0, 6), dtype=torch.float32)

        # Joint-related attributes
        self.joint_q = torch.zeros((0, 7), dtype=torch.float32)
        self.joint_qd = torch.zeros((0, 6), dtype=torch.float32)
        self.joint_f = torch.zeros((0, ), dtype=torch.float32)
        self.joint_act = torch.zeros((0, ), dtype=torch.float32)

        # Robot-related attributes
        self.robot_q = torch.zeros((0, 7), dtype=torch.float32)
        self.robot_qd = torch.zeros((0, 6), dtype=torch.float32)
        self.robot_f = torch.zeros((0, 6), dtype=torch.float32)

        # Collision-related attributes
        self.contact_count = 0
        self.contact_body = torch.zeros((0, ), dtype=torch.int32)
        self.contact_point = torch.zeros((0, 3), dtype=torch.float32)
        self.contact_normal = torch.zeros((0, 3), dtype=torch.float32)

        # Correction-related attributes
        self.contact_deltas = torch.zeros((0, 7), dtype=torch.float32)
        self.restitution_deltas = torch.zeros((0, 6), dtype=torch.float32)
        self.static_friction_deltas = torch.zeros((0, 6), dtype=torch.float32)
        self.dynamic_friction_deltas = torch.zeros((0, 6), dtype=torch.float32)

        self.robot_contact_deltas = torch.zeros((0, 7), dtype=torch.float32)

    @property
    def body_count(self):
        assert self.body_q is not None, 'Body state is not initialized'
        return self.body_q.shape[0]

    @property
    def joint_count(self):
        assert self.joint_q is not None, 'Joint state is not initialized'
        return self.joint_q.shape[0]

    def add_contact_deltas(self):
        if self.contact_count == 0:
            return

        for b in range(self.body_count):
            body_indices = torch.where(self.contact_body == b)[0]
            delta_q = torch.sum(self.contact_deltas[body_indices],
                                dim=0) / len(body_indices)
            self.body_q[b, :3] += delta_q[:3]
            self.body_q[b,
                        3:] = normalize_quat(self.body_q[b, 3:] + delta_q[3:])

    def add_robot_contact_deltas(self):
        if self.contact_count == 0:
            return

        delta_q = torch.sum(self.robot_contact_deltas, dim=0)
        self.robot_q[:3] += delta_q[:3]
        self.robot_q[3:] = normalize_quat(self.robot_q[3:] + delta_q[3:])

    def serialize(self, model: 'Model'):

        def tensor_to_list(tensor):
            return tensor.detach().cpu().tolist() if tensor.numel() > 0 else []

        # list of tensors where each tensor is contact_points of certain body
        body_contact_points = [
            transform(self.contact_point[self.contact_body == i],
                      self.body_q[i]) for i in range(model.body_count)
        ]
        body_contact_normals_start = [
            body_contact_points[i] for i in range(model.body_count)
        ]
        body_contact_normals_end = [
            body_contact_points[i] +
            self.contact_normal[self.contact_body == i]
            for i in range(model.body_count)
        ]

        contact_normals = []
        for i in range(model.body_count):
            body_contact_normals = []
            for j in range(len(body_contact_normals_start[i])):
                body_contact_normals.append({
                    "start":
                    tensor_to_list(body_contact_normals_start[i][j]),
                    "end":
                    tensor_to_list(body_contact_normals_end[i][j])
                })
            contact_normals.append(body_contact_normals)

        data = {
            "time":
            self.time,
            "bodies": [
                {
                    "name": model.body_name[i],
                    "q": tensor_to_list(self.body_q[i]),
                    "qd": tensor_to_list(self.body_qd[i]),
                    "contact_points": tensor_to_list(body_contact_points[i]),
                    "contact_normals": contact_normals[i]
                    # "f": tensor_to_list(self.body_f[i])
                } for i in range(self.body_count)
            ],
            "joints": [
                {
                    "q": tensor_to_list(self.joint_q[i]),
                    "qd": tensor_to_list(self.joint_qd[i]),
                    # "f": tensor_to_list(self.joint_f[i]),
                    "act": self.joint_act[i].item()
                } for i in range(self.joint_count)
            ],
            "robot": {
                "q": tensor_to_list(self.robot_q[0]),
                "qd": tensor_to_list(self.robot_qd[0]),
                # "f": tensor_to_list(self.robot_f[0])
            },
            # "contacts": [{
            #     "body":
            #     self.contact_body[i].item(),
            #     "point":
            #     tensor_to_list(self.contact_point[i]),
            #     "normal":
            #     tensor_to_list(self.contact_normal[i]),
            #     "point_world":
            #     tensor_to_list(
            #         transform(self.contact_point[i],
            #                   self.body_q[self.contact_body[i]])),
            #     "normal_vector": {
            #         "start":
            #         tensor_to_list(
            #             transform(self.contact_point[i],
            #                       self.body_q[self.contact_body[i]])),
            #         "end":
            #         tensor_to_list(
            #             transform(self.contact_point[i], self.body_q[
            #                 self.contact_body[i]]) + self.contact_normal[i])
            #     }
            # } for i in range(self.contact_count)]
        }

        return data


class Control:

    def __init__(self):
        self.joint_act = None


class Model:

    def __init__(self):
        self.body_count = 0
        self.joint_count = 0

        device = torch.device('cpu')
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # ====================== Body-related attributes ======================
        self.body_name: List[str] = []
        self.body_shapes: List[Shape] = []

        self.body_q = torch.zeros((0, 7), dtype=torch.float32, device=device)
        self.body_qd = torch.zeros((0, 6), dtype=torch.float32, device=device)
        self.body_f = torch.zeros((0, 6), dtype=torch.float32, device=device)

        self.body_mass = torch.zeros((0, ), dtype=torch.float32, device=device)
        self.body_inv_mass = torch.zeros((0, ),
                                         dtype=torch.float32,
                                         device=device)
        self.body_restitution = torch.zeros((0, ),
                                            dtype=torch.float32,
                                            device=device)
        self.body_inertia = torch.zeros((0, 3, 3),
                                        dtype=torch.float32,
                                        device=device)
        self.body_inv_inertia = torch.zeros((0, 3, 3),
                                            dtype=torch.float32,
                                            device=device)
        self.body_collision_points: dict[int, torch.Tensor] = {}

        # ====================== Robot-related attributes ======================
        self.robot_name: str = ''

        self.robot_q = torch.zeros((7, ), dtype=torch.float32, device=device)
        self.robot_qd = torch.zeros((6, ), dtype=torch.float32, device=device)
        self.robot_f = torch.zeros((6, ), dtype=torch.float32, device=device)

        self.robot_mass = 0.0
        self.robot_inv_mass = 0.0

        self.robot_inertia = torch.zeros((3, 3),
                                         dtype=torch.float32,
                                         device=device)
        self.robot_inv_inertia = torch.zeros((3, 3),
                                             dtype=torch.float32,
                                             device=device)

        self.robot_body = torch.zeros((0, ), dtype=torch.int32, device=device)
        self.robot_joint = torch.zeros((0, ), dtype=torch.int32, device=device)

        # ====================== Joint-related attributes ======================
        self.joint_name = []
        self.joint_q = torch.zeros((0, ), dtype=torch.float32, device=device)
        self.joint_qd = torch.zeros((0, ), dtype=torch.float32, device=device)
        self.joint_act = torch.zeros((0, ), dtype=torch.float32, device=device)
        self.joint_axis_mode = torch.zeros((0, ),
                                           dtype=torch.int32,
                                           device=device)
        self.joint_ke = torch.zeros((0, ), dtype=torch.float32, device=device)
        self.joint_kd = torch.zeros((0, ), dtype=torch.float32, device=device)
        self.joint_parent = torch.zeros((0, ),
                                        dtype=torch.int32,
                                        device=device)
        self.joint_child = torch.zeros((0, ), dtype=torch.int32, device=device)

        self.joint_X_p = torch.zeros((0, 7),
                                     dtype=torch.float32,
                                     device=device)
        self.joint_X_c = torch.zeros((0, 7),
                                     dtype=torch.float32,
                                     device=device)
        self.joint_axis = torch.zeros((0, 3),
                                      dtype=torch.float32,
                                      device=device)
        self.joint_compliance = torch.zeros((0, ),
                                            dtype=torch.float32,
                                            device=device)

        # System properties
        self.ground = True
        self.gravity = torch.tensor([0.0, 0.0, -9.81],
                                    dtype=torch.float32,
                                    device=device)

    def add_box(self,
                m: float,
                hx: float = 1.0,
                hy: float = 1.0,
                hz: float = 1.0,
                name: str = None,
                pos: Vector3 = Vector3(torch.zeros(3)),
                rot: Quaternion = Quaternion(torch.tensor([1, 0, 0, 0])),
                restitution: float = 0.5,
                n_collision_points: int = 200) -> int:
        assert m > 0, 'Body mass must be positive'
        assert hx > 0, 'Box hx must be positive'
        assert hy > 0, 'Box hy must be positive'
        assert hz > 0, 'Box hz must be positive'

        self.body_count += 1

        self.body_shapes.append(Box(hx=hx, hy=hy, hz=hz))
        self.body_name.append(name or f'body_{self.body_count}')
        self.body_q = torch.cat(
            [self.body_q, torch.cat([pos, rot]).unsqueeze(0)])
        self.body_qd = torch.cat([self.body_qd, torch.zeros(1, 6)])
        self.body_f = torch.cat([self.body_f, torch.zeros(1, 6)])

        self.body_mass = torch.cat([self.body_mass, torch.tensor([m])])
        self.body_inv_mass = torch.cat(
            [self.body_inv_mass, torch.tensor([1 / m])])

        inertia = torch.Tensor([[m / 12 * (hy**2 + hz**2), 0.0, 0.0],
                                [0.0, m / 12 * (hx**2 + hz**2), 0.0],
                                [0.0, 0.0, m / 12 * (hx**2 + hy**2)]])
        self.body_inertia = torch.cat(
            [self.body_inertia, inertia.unsqueeze(0)])
        self.body_inv_inertia = torch.cat(
            [self.body_inv_inertia,
             torch.inverse(inertia).unsqueeze(0)])

        self.body_collision_points[self.body_count - 1] = box_collision_points(
            hx, hy, hz, n_collision_points)

        return self.body_count - 1

    def add_sphere(self,
                   m: float,
                   radius: float = 1.0,
                   name: str = None,
                   pos: Vector3 = Vector3(torch.zeros(3)),
                   rot: Quaternion = Quaternion(torch.tensor([1, 0, 0, 0])),
                   restitution: float = 0.5,
                   n_collision_points: int = 1000) -> int:
        assert m > 0, 'Body mass must be positive'
        assert radius > 0, 'Sphere radius must be positive'

        self.body_count += 1
        self.body_shapes.append(Sphere(radius=radius))

        self.body_name.append(name or f'body_{self.body_count}')
        self.body_q = torch.cat(
            [self.body_q, torch.cat([pos, rot]).unsqueeze(0)])
        self.body_qd = torch.cat([self.body_qd, torch.zeros(1, 6)])
        self.body_f = torch.cat([self.body_f, torch.zeros(1, 6)])

        self.body_mass = torch.cat([self.body_mass, torch.tensor([m])])
        self.body_inv_mass = torch.cat(
            [self.body_inv_mass, torch.tensor([1 / m])])

        inertia = torch.Tensor([[2 / 5 * m * radius, 0.0, 0.0],
                                [0.0, 2 / 5 * m * radius, 0.0],
                                [0.0, 0.0, 2 / 5 * m * radius]]).unsqueeze(0)

        self.body_inertia = torch.cat([self.body_inertia, inertia])
        self.body_inv_inertia = torch.cat(
            [self.body_inv_inertia,
             torch.inverse(inertia)])

        self.body_collision_points[self.body_count -
                                   1] = sphere_collision_points(
                                       radius, n_collision_points)

        return self.body_count - 1

    def add_cylinder(self,
                     m: float,
                     radius: float = 1.0,
                     height: float = 1.0,
                     name: str = None,
                     pos: Vector3 = Vector3(torch.zeros(3)),
                     rot: Quaternion = Quaternion(torch.tensor([1, 0, 0, 0])),
                     restitution: float = 0.5,
                     n_collision_points_base: int = 500,
                     n_collision_points_surface: int = 1000) -> int:
        assert m > 0, 'Body mass must be positive'
        assert radius > 0, 'Cylinder radius must be positive'
        assert height > 0, 'Cylinder height must be positive'

        self.body_count += 1
        self.body_shapes.append(Cylinder(radius=radius, height=height))

        self.body_name.append(name or f'body_{self.body_count}')
        self.body_q = torch.cat(
            [self.body_q, torch.cat([pos, rot]).unsqueeze(0)])
        self.body_qd = torch.cat([self.body_qd, torch.zeros(1, 6)])
        self.body_f = torch.cat([self.body_f, torch.zeros(1, 6)])

        self.body_mass = torch.cat([self.body_mass, torch.tensor([m])])
        self.body_inv_mass = torch.cat(
            [self.body_inv_mass, torch.tensor([1 / m])])

        inertia = torch.Tensor(
            [[m * (3 * radius**2 + height**2) / 12, 0.0, 0.0],
             [0.0, m * (3 * radius**2 + height**2) / 12, 0.0],
             [0.0, 0.0, m * radius**2 / 2]])
        self.body_inertia = torch.cat(
            [self.body_inertia, inertia.unsqueeze(0)])
        self.body_inv_inertia = torch.cat(
            [self.body_inv_inertia,
             torch.inverse(inertia).unsqueeze(0)])

        self.body_collision_points[self.body_count -
                                   1] = cylinder_collision_points(
                                       radius, height, n_collision_points_base,
                                       n_collision_points_surface)

        return self.body_count - 1

    def add_hinge_joint(self,
                        parent: int,
                        child: int,
                        axis: Vector3,
                        name: str = None,
                        q: float = 0.0,
                        qd: float = 0.0,
                        act: float = 0.0,
                        parent_xform: Transform = None,
                        child_xform: Transform = None,
                        compliance: float = 0.0) -> int:
        assert parent >= -1 and parent < self.body_count, 'Parent index out of bounds'
        assert child >= 0 and child < self.body_count, 'Child index out of bounds'

        self.joint_count += 1

        # Add joint name
        self.joint_name.append(name or f'joint_{self.joint_count}')

        # Add joint properties
        self.joint_q = torch.cat(
            [self.joint_q, torch.tensor([q], device=self.device)])
        self.joint_qd = torch.cat(
            [self.joint_qd,
             torch.tensor([qd], device=self.device)])
        self.joint_act = torch.cat(
            [self.joint_act,
             torch.tensor([act], device=self.device)])

        # Add parent and child indices
        self.joint_parent = torch.cat(
            [self.joint_parent,
             torch.tensor([parent], device=self.device)])
        self.joint_child = torch.cat(
            [self.joint_child,
             torch.tensor([child], device=self.device)])

        # Add joint axis
        self.joint_axis = torch.cat([self.joint_axis, axis.unsqueeze(0)])

        # Add joint compliance
        self.joint_compliance = torch.cat([
            self.joint_compliance,
            torch.tensor([compliance], device=self.device)
        ])

        parent_xform = parent_xform or torch.cat(
            [torch.zeros(3),
             torch.tensor([1, 0, 0, 0], device=self.device)])
        child_xform = child_xform or torch.cat(
            [torch.zeros(3),
             torch.tensor([1, 0, 0, 0], device=self.device)])

        # Initialize joint transforms (you might want to modify these based on your needs)
        self.joint_X_p = torch.cat([self.joint_X_p, parent_xform.unsqueeze(0)])
        self.joint_X_c = torch.cat([self.joint_X_c, child_xform.unsqueeze(0)])

        # Set axis mode to force by default
        self.joint_axis_mode = torch.cat([
            self.joint_axis_mode,
            torch.tensor([JOINT_MODE_FORCE], device=self.device)
        ])

        # Set default stiffness and damping
        self.joint_ke = torch.cat([self.joint_ke, torch.zeros(1)])
        self.joint_kd = torch.cat([self.joint_kd, torch.zeros(1)])

        return self.joint_count - 1

    def construct_robot(self, name: str, pos: Vector3, rot: Quaternion,
                        body_indices: List[int], joint_indices: List[int]):
        self.robot_name = name
        self.robot_q = torch.cat([pos, rot])
        self.robot_qd = torch.zeros((6, ), device=self.device)
        self.robot_body = torch.tensor(body_indices, device=self.device)
        self.robot_joint = torch.tensor(joint_indices, device=self.device)

        # Calculate robot mass and inertia
        self.robot_mass = torch.sum(self.body_mass[self.robot_body])
        self.robot_inv_mass = 1 / self.robot_mass

        self.robot_inertia = torch.zeros((3, 3), device=self.device)
        for b in self.robot_body:
            X = body_to_robot_transform(self.body_q[b], self.robot_q)
            I_m = self.body_inertia[b] + self.body_mass[b] * torch.outer(
                X[:3], X[:3])

            # Rotate the I_m to the robot frame
            R = quat_to_rotmat(X[3:])

            self.robot_inertia += R @ I_m @ R.T

        self.robot_inv_inertia = torch.inverse(self.robot_inertia)

    def state(self):
        state = State()
        state.body_q = self.body_q.clone()
        state.body_qd = self.body_qd.clone()
        state.body_f = self.body_f.clone()

        state.joint_q = self.joint_q.clone()
        state.joint_qd = self.joint_qd.clone()
        state.joint_act = self.joint_act.clone()

        state.robot_q = self.robot_q.clone()
        state.robot_qd = self.robot_qd.clone()
        state.robot_f = self.robot_f.clone()
        return state

    def control(self):
        control = Control()
        control.joint_act = torch.zeros((self.joint_count, ),
                                        dtype=torch.float32)
        return control

    def serialize(self):

        def tensor_to_list(tensor):
            return tensor.cpu().tolist() if tensor.numel() > 0 else []

        data = {
            "body_count":
            self.body_count,
            "joint_count":
            self.joint_count,
            "gravity":
            tensor_to_list(self.gravity),
            "ground":
            self.ground,
            "bodies": [{
                "name":
                self.body_name[i],
                "shape":
                self.body_shapes[i].serialize(),
                "mass":
                self.body_mass[i].item(),
                "inv_mass":
                self.body_inv_mass[i].item(),
                "q":
                tensor_to_list(self.body_q[i]),
                "qd":
                tensor_to_list(self.body_qd[i]),
                "f":
                tensor_to_list(self.body_f[i]),
                "inertia":
                tensor_to_list(self.body_inertia[i]),
                "inv_inertia":
                tensor_to_list(self.body_inv_inertia[i]),
                "collision_points":
                tensor_to_list(
                    self.body_collision_points.get(i, torch.empty(0)))
            } for i in range(self.body_count)],
            "joints": [{
                "name": self.joint_name[i],
                "q": self.joint_q[i].item(),
                "qd": self.joint_qd[i].item(),
                "act": self.joint_act[i].item(),
                "axis": tensor_to_list(self.joint_axis[i]),
                "parent": self.joint_parent[i].item(),
                "child": self.joint_child[i].item(),
                "compliance": self.joint_compliance[i].item(),
                "X_p": tensor_to_list(self.joint_X_p[i]),
                "X_c": tensor_to_list(self.joint_X_c[i]),
            } for i in range(self.joint_count)],
        }
        return data


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


def _generate_fibonacci_surface_points(radius: float, height: float,
                                       n_points: int):
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


def sphere_collision_points(radius: float, n_points: int):
    return _generate_fibonacci_sphere_points(radius, n_points)


def cylinder_collision_points(radius: float, height: float, n_base_points: int,
                              n_surface_points: int):
    base_points = _generate_fibonacci_disk_points(radius, n_base_points)
    surface_vertices = _generate_fibonacci_surface_points(
        radius, height, n_surface_points)

    # Create top and bottom vertices for the bases
    top_vertices = torch.cat(
        [base_points,
         torch.ones(len(base_points), 1) * height / 2], dim=1)

    bottom_vertices = torch.cat(
        [base_points,
         torch.ones(len(base_points), 1) * -height / 2], dim=1)

    # Combine all vertices
    vertices = torch.cat([surface_vertices, top_vertices, bottom_vertices],
                         dim=0).type(torch.float32)

    return vertices


def box_collision_points(hx: float, hy: float, hz: float, n_points: int):
    points = []
    points_per_face = n_points // 6

    # Generate points for each face
    # Front and back faces (YZ planes)
    for x in [-hx, hx]:
        y = torch.linspace(-hy, hy,
                           int(torch.sqrt(torch.tensor(points_per_face))))
        z = torch.linspace(-hz, hz,
                           int(torch.sqrt(torch.tensor(points_per_face))))
        Y, Z = torch.meshgrid(y, z, indexing='ij')
        face_points = torch.stack(
            [torch.full_like(Y, x).flatten(),
             Y.flatten(),
             Z.flatten()], dim=1)
        points.append(face_points)

    # Left and right faces (XZ planes)
    for y in [-hy, hy]:
        x = torch.linspace(-hx, hx,
                           int(torch.sqrt(torch.tensor(points_per_face))))
        z = torch.linspace(-hz, hz,
                           int(torch.sqrt(torch.tensor(points_per_face))))
        X, Z = torch.meshgrid(x, z, indexing='ij')
        face_points = torch.stack(
            [X.flatten(),
             torch.full_like(X, y).flatten(),
             Z.flatten()], dim=1)
        points.append(face_points)

    # Top and bottom faces (XY planes)
    for z in [-hz, hz]:
        x = torch.linspace(-hx, hx,
                           int(torch.sqrt(torch.tensor(points_per_face))))
        y = torch.linspace(-hy, hy,
                           int(torch.sqrt(torch.tensor(points_per_face))))
        X, Y = torch.meshgrid(x, y, indexing='ij')
        face_points = torch.stack(
            [X.flatten(),
             Y.flatten(),
             torch.full_like(X, z).flatten()], dim=1)
        points.append(face_points)

    points = torch.cat(points, dim=0)
    points = torch.unique(points, dim=0)

    return points


# ===================================================================
# ------------------------------ DEMOS ------------------------------
# ===================================================================


def visualize_points(ax: Axes, points: torch.Tensor, title: str):
    # Plot the collision points
    ax.scatter(points[:, 0],
               points[:, 1],
               points[:, 2],
               c=points[:, 2],
               marker='o',
               s=10,
               cmap='viridis',
               alpha=0.7)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Make the plot more viewable
    ax.set_box_aspect([1, 1, 1])


def show_collision_models():
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    # Test sphere
    sphere_vertices = sphere_collision_points(radius=1.0, n_points=1000)
    visualize_points(ax1, sphere_vertices, "Sphere Points")

    # Test cylinder
    cylinder_vertices = cylinder_collision_points(radius=1.0,
                                                  height=2.0,
                                                  n_base_points=500,
                                                  n_surface_points=1000)
    visualize_points(ax2, cylinder_vertices, "Cylinder Points")

    # Test box
    box_vertices = box_collision_points(hx=2.0, hy=1.5, hz=1.0, n_points=2000)
    visualize_points(ax3, box_vertices, "Box Points")

    plt.show()


if __name__ == "__main__":
    show_collision_models()
