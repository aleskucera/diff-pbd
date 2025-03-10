from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import NewType
from typing import Union

import torch
from pbd_torch.constants import *
from pbd_torch.terrain import Terrain
from pbd_torch.transform import *

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


class IntermediateState:

    def __init__(self, state: 'State') -> None:
        self.time = 0.0

        self.body_q = state.body_q.clone()
        self.body_qd = state.body_qd.clone()
        self.body_f = state.body_f.clone()

        self.joint_q = state.joint_q.clone()
        self.joint_qd = state.joint_qd.clone()
        self.joint_f = state.joint_f.clone()

        # Robot-related attributes (only one robot)
        self.robot_q = state.robot_q.clone()
        self.robot_qd = state.robot_qd.clone()
        self.robot_f = state.robot_f.clone()

        # Contact-related attributes
        self.contact_count = 0
        self.contact_body = torch.zeros((0, ), dtype=torch.int32)
        self.contact_point = torch.zeros((0, 3), dtype=torch.float32)
        self.contact_normal = torch.zeros((0, 3), dtype=torch.float32)

        # Correction-related attributes
        self.contact_deltas = torch.zeros((0, 7), dtype=torch.float32)
        self.restitution_deltas = torch.zeros((0, 6), dtype=torch.float32)
        self.static_friction_deltas = torch.zeros((0, 6), dtype=torch.float32)
        self.dynamic_friction_deltas = torch.zeros((0, 6), dtype=torch.float32)


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
        self.contact_point_idx = torch.zeros((0, ), dtype=torch.int32)

        # Correction-related attributes
        self.contact_deltas = torch.zeros((0, 7), dtype=torch.float32)
        self.restitution_deltas = torch.zeros((0, 6), dtype=torch.float32)
        self.static_friction_deltas = torch.zeros((0, 6), dtype=torch.float32)
        self.dynamic_friction_deltas = torch.zeros((0, 6), dtype=torch.float32)

        self.robot_contact_deltas = torch.zeros((0, 7), dtype=torch.float32)

    @property
    def body_count(self):
        return self.body_q.shape[0]

    @property
    def joint_count(self):
        return self.joint_q.shape[0]

    def serialize(self, model: 'Model'):

        def tensor_to_list(tensor):
            return tensor.detach().cpu().tolist() if tensor.numel() > 0 else []

        contacts = [
            self.contact_point_idx[torch.where(self.contact_body == b)[0]]
            for b in range(self.body_count)
        ]

        data = {
            "time":
            self.time,
            "bodies": [{
                "name": model.body_name[i],
                "q": tensor_to_list(self.body_q[i]),
                "qd": tensor_to_list(self.body_qd[i]),
                "contacts": tensor_to_list(contacts[i]),
                "f": tensor_to_list(self.body_f[i])
            } for i in range(self.body_count)],
        }

        if model.is_robot:
            data["robot"] = {
                "name": model.robot_name,
                "q": tensor_to_list(self.robot_q),
                "qd": tensor_to_list(self.robot_qd),
                "f": tensor_to_list(self.robot_f)
            }

        return data


class Control:

    def __init__(self):
        self.joint_act = torch.zeros((0, ), dtype=torch.float32)


class Model:

    def __init__(self, device: torch.device = None, requires_grad: bool = False, terrain: Terrain = None):
        self.body_count = 0
        self.joint_count = 0

        # Set the device
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.requires_grad = requires_grad
        
        self.terrain = terrain

        # ====================== Body-related attributes ======================
        self.body_name: List[str] = []
        self.body_shapes: List[Shape] = []

        self.body_q = torch.zeros((0, 7), dtype=torch.float32, device=device, requires_grad=requires_grad)
        self.body_qd = torch.zeros((0, 6), dtype=torch.float32, device=device, requires_grad=requires_grad)
        self.body_f = torch.zeros((0, 6), dtype=torch.float32, device=device, requires_grad=requires_grad)

        self.body_mass = torch.zeros((0, ), dtype=torch.float32, device=device, requires_grad=requires_grad)
        self.body_inv_mass = torch.zeros((0, ),
                                         dtype=torch.float32,
                                         device=device, requires_grad=requires_grad)
        self.body_restitution = torch.zeros((0, ),
                                            dtype=torch.float32,
                                            device=device, requires_grad=requires_grad)
        self.body_inertia = torch.zeros((0, 3, 3),
                                        dtype=torch.float32,
                                        device=device, requires_grad=requires_grad)
        self.body_inv_inertia = torch.zeros((0, 3, 3),
                                            dtype=torch.float32,
                                            device=device, requires_grad=requires_grad)
        self.body_collision_points: dict[int, torch.Tensor] = {}

        self.restitution = torch.zeros((0, ),
                                       dtype=torch.float32,
                                       device=device, requires_grad=requires_grad)
        self.static_friction = torch.zeros((0, ),
                                           dtype=torch.float32,
                                           device=device, requires_grad=requires_grad)
        self.dynamic_friction = torch.zeros((0, ),
                                            dtype=torch.float32,
                                            device=device, requires_grad=requires_grad)

        # ====================== Robot-related attributes ======================
        self.is_robot = False
        self.robot_name: str = 'robot'

        self.robot_q = torch.zeros((7, ), dtype=torch.float32, device=device, requires_grad=requires_grad)
        self.robot_qd = torch.zeros((6, ), dtype=torch.float32, device=device, requires_grad=requires_grad)
        self.robot_f = torch.zeros((6, ), dtype=torch.float32, device=device, requires_grad=requires_grad)

        self.robot_mass = torch.tensor([0.0],
                                       dtype=torch.float32,
                                       device=device, requires_grad=requires_grad)
        self.robot_inv_mass = torch.tensor([0.0],
                                           dtype=torch.float32,
                                           device=device, requires_grad=requires_grad)

        self.robot_inertia = torch.zeros((3, 3),
                                         dtype=torch.float32,
                                         device=device, requires_grad=requires_grad)
        self.robot_inv_inertia = torch.zeros((3, 3),
                                             dtype=torch.float32,
                                             device=device, requires_grad=requires_grad)

        self.robot_body = torch.zeros((0, ), dtype=torch.int32, device=device)
        self.robot_joint = torch.zeros((0, ), dtype=torch.int32, device=device)

        # ====================== Joint-related attributes ======================
        self.joint_name = []
        self.joint_q = torch.zeros((0, ), dtype=torch.float32, device=device, requires_grad=requires_grad)
        self.joint_qd = torch.zeros((0, ), dtype=torch.float32, device=device, requires_grad=requires_grad)
        self.joint_act = torch.zeros((0, ), dtype=torch.float32, device=device, requires_grad=requires_grad)
        self.joint_axis_mode = torch.zeros((0, ),
                                           dtype=torch.int32,
                                           device=device)
        self.joint_ke = torch.zeros((0, ), dtype=torch.float32, device=device, requires_grad=requires_grad)
        self.joint_kd = torch.zeros((0, ), dtype=torch.float32, device=device, requires_grad=requires_grad)
        self.joint_parent = torch.zeros((0, ),
                                        dtype=torch.int32,
                                        device=device)
        self.joint_child = torch.zeros((0, ), dtype=torch.int32, device=device)

        self.joint_X_p = torch.zeros((0, 7),
                                     dtype=torch.float32,
                                     device=device, requires_grad=requires_grad)
        self.joint_X_c = torch.zeros((0, 7),
                                     dtype=torch.float32,
                                     device=device, requires_grad=requires_grad)
        self.joint_axis = torch.zeros((0, 3),
                                      dtype=torch.float32,
                                      device=device, requires_grad=requires_grad)
        self.joint_compliance = torch.zeros((0, ),
                                            dtype=torch.float32,
                                            device=device, requires_grad=requires_grad)

        # ====================== Contact-related attributes ======================
        self.contact_count = 0
        self.contact_body = torch.zeros((0, ),
                                        dtype=torch.int32,
                                        device=device)
        self.contact_point = torch.zeros((0, 3),
                                         dtype=torch.float32,
                                         device=device)
        self.contact_normal = torch.zeros((0, 3),
                                          dtype=torch.float32,
                                          device=device)
        self.contact_point_idx = torch.zeros((0, ),
                                             dtype=torch.int32,
                                             device=device)

        # System properties
        self.gravity = torch.tensor([0.0, 0.0, -9.81],
                                    dtype=torch.float32,
                                    device=device)

    def _append_body_mass(self, m: float):
        assert m > 0, 'Body mass must be positive'
        assert isinstance(m, float), 'Body mass must be a float'

        body_mass = torch.tensor([m], device=self.device)
        self.body_mass = torch.cat([self.body_mass, body_mass]).requires_grad_(self.requires_grad)
        body_inv_mass = torch.tensor([1 / m], device=self.device)
        self.body_inv_mass = torch.cat([self.body_inv_mass, body_inv_mass]).requires_grad_(self.requires_grad)

    def _append_body_inertia(self, inertia: torch.Tensor):
        assert inertia.shape == (3, 3), 'Inertia must be a 3x3 matrix'

        self.body_inertia = torch.cat(
            [self.body_inertia, inertia.unsqueeze(0)]).requires_grad_(self.requires_grad)
        self.body_inv_inertia = torch.cat(
            [self.body_inv_inertia,
             torch.inverse(inertia).unsqueeze(0)]).requires_grad_(self.requires_grad)

    def add_box(self,
                m: float,
                hx: float = 1.0,
                hy: float = 1.0,
                hz: float = 1.0,
                name: Union[str, None] = None,
                pos: Vector3 = Vector3(torch.zeros(3)),
                rot: Quaternion = Quaternion(torch.tensor([1, 0, 0, 0])),
                restitution: float = 0.5,
                static_friction: float = 0.5,
                dynamic_friction: float = 0.5,
                n_collision_points: int = 200) -> int:
        assert m > 0, 'Body mass must be positive'
        assert hx > 0, 'Box hx must be positive'
        assert hy > 0, 'Box hy must be positive'
        assert hz > 0, 'Box hz must be positive'

        self.body_count += 1
        body_idx = self.body_count - 1

        self.body_shapes.append(Box(hx=hx, hy=hy, hz=hz))
        self.body_name.append(name or f'body_{self.body_count}')

        body_q = torch.cat([pos, rot]).unsqueeze(0).to(self.device)
        self.body_q = torch.cat([self.body_q, body_q]).requires_grad_(self.requires_grad)
        doby_qd = torch.zeros((1, 6), device=self.device)
        self.body_qd = torch.cat([self.body_qd, doby_qd]).requires_grad_(self.requires_grad)
        body_f = torch.zeros((1, 6), device=self.device)
        self.body_f = torch.cat([self.body_f, body_f]).requires_grad_(self.requires_grad)

        self._append_body_mass(m)
        inertia = torch.tensor([[m / 12 * (hy**2 + hz**2), 0.0, 0.0],
                                [0.0, m / 12 * (hx**2 + hz**2), 0.0],
                                [0.0, 0.0, m / 12 * (hx**2 + hy**2)]],
                               device=self.device, requires_grad=self.requires_grad)
        self._append_body_inertia(inertia)

        coll_points = box_collision_points(hx, hy, hz, n_collision_points, self.device)
        self.body_collision_points[body_idx] = coll_points

        r = torch.tensor([restitution], device=self.device)
        self.restitution = torch.cat([self.restitution, r]).requires_grad_(self.requires_grad)
        s = torch.tensor([static_friction], device=self.device)
        self.static_friction = torch.cat([self.static_friction, s]).requires_grad_(self.requires_grad)
        d = torch.tensor([dynamic_friction], device=self.device)
        self.dynamic_friction = torch.cat([self.dynamic_friction, d]).requires_grad_(self.requires_grad)

        return body_idx

    def add_sphere(self,
                   m: float,
                   radius: float = 1.0,
                   name: Union[str, None] = None,
                   pos: Vector3 = Vector3(torch.zeros(3)),
                   rot: Quaternion = Quaternion(torch.tensor([1, 0, 0, 0])),
                   restitution: float = 0.5,
                   static_friction: float = 0.5,
                   dynamic_friction: float = 0.5,
                   n_collision_points: int = 1000) -> int:
        assert m > 0, 'Body mass must be positive'
        assert radius > 0, 'Sphere radius must be positive'

        self.body_count += 1
        body_idx = self.body_count - 1

        self.body_shapes.append(Sphere(radius=radius))
        self.body_name.append(name or f'body_{self.body_count}')

        body_q = torch.cat([pos, rot]).unsqueeze(0).to(self.device)
        self.body_q = torch.cat([self.body_q, body_q]).requires_grad_(self.requires_grad)
        doby_qd = torch.zeros((1, 6), device=self.device)
        self.body_qd = torch.cat([self.body_qd, doby_qd]).requires_grad_(self.requires_grad)
        body_f = torch.zeros((1, 6), device=self.device)
        self.body_f = torch.cat([self.body_f, body_f]).requires_grad_(self.requires_grad)

        self._append_body_mass(m)
        inertia = torch.tensor(
            [[2 / 5 * m * radius, 0.0, 0.0], [0.0, 2 / 5 * m * radius, 0.0],
             [0.0, 0.0, 2 / 5 * m * radius]],
            device=self.device, requires_grad=self.requires_grad)
        self._append_body_inertia(inertia)

        coll_points = sphere_collision_points(radius, n_collision_points, self.device)
        self.body_collision_points[body_idx] = coll_points

        r = torch.tensor([restitution], device=self.device)
        self.restitution = torch.cat([self.restitution, r]).requires_grad_(self.requires_grad)
        s = torch.tensor([static_friction], device=self.device)
        self.static_friction = torch.cat([self.static_friction, s]).requires_grad_(self.requires_grad)
        d = torch.tensor([dynamic_friction], device=self.device)
        self.dynamic_friction = torch.cat([self.dynamic_friction, d]).requires_grad_(self.requires_grad)

        return body_idx

    def add_cylinder(self,
                     m: float,
                     radius: float = 1.0,
                     height: float = 1.0,
                     name: Union[str, None] = None,
                     pos: Vector3 = Vector3(torch.zeros(3)),
                     rot: Quaternion = Quaternion(torch.tensor([1, 0, 0, 0])),
                     restitution: float = 0.5,
                     static_friction: float = 0.5,
                     dynamic_friction: float = 0.5,
                     n_collision_points_base: int = 500,
                     n_collision_points_surface: int = 1000) -> int:
        assert m > 0, 'Body mass must be positive'
        assert radius > 0, 'Cylinder radius must be positive'
        assert height > 0, 'Cylinder height must be positive'

        self.body_count += 1
        body_idx = self.body_count - 1

        self.body_shapes.append(Cylinder(radius=radius, height=height))
        self.body_name.append(name or f'body_{self.body_count}')

        body_q = torch.cat([pos, rot]).unsqueeze(0).to(self.device)
        self.body_q = torch.cat([self.body_q, body_q])
        doby_qd = torch.zeros((1, 6), device=self.device)
        self.body_qd = torch.cat([self.body_qd, doby_qd])
        body_f = torch.zeros((1, 6), device=self.device)
        self.body_f = torch.cat([self.body_f, body_f])

        self._append_body_mass(m)
        inertia = torch.tensor(
            [[m * (3 * radius**2 + height**2) / 12, 0.0, 0.0],
             [0.0, m * (3 * radius**2 + height**2) / 12, 0.0],
             [0.0, 0.0, m * radius**2 / 2]],
            device=self.device)
        self._append_body_inertia(inertia)

        coll_points = cylinder_collision_points(radius, height,
                                                n_collision_points_base,
                                                n_collision_points_surface,
                                                self.device)
        self.body_collision_points[body_idx] = coll_points

        r = torch.tensor([restitution], device=self.device)
        self.restitution = torch.cat([self.restitution, r])
        s = torch.tensor([static_friction], device=self.device)
        self.static_friction = torch.cat([self.static_friction, s])
        d = torch.tensor([dynamic_friction], device=self.device)
        self.dynamic_friction = torch.cat([self.dynamic_friction, d])

        return body_idx

    def add_hinge_joint(self,
                        parent: int,
                        child: int,
                        axis: Vector3,
                        name: Union[str, None] = None,
                        q: float = 0.0,
                        qd: float = 0.0,
                        act: float = 0.0,
                        parent_xform: torch.Tensor = TRANSFORM_IDENTITY,
                        child_xform: torch.Tensor = TRANSFORM_IDENTITY,
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
        joint_axis = (axis / torch.norm(axis)).unsqueeze(0).to(self.device)
        self.joint_axis = torch.cat([self.joint_axis, joint_axis])

        # Add joint compliance
        self.joint_compliance = torch.cat([
            self.joint_compliance,
            torch.tensor([compliance], device=self.device)
        ])

        parent_xform = parent_xform.unsqueeze(0).to(self.device)
        child_xform = child_xform.unsqueeze(0).to(self.device)
        # Initialize joint transforms (you might want to modify these based on your needs)
        self.joint_X_p = torch.cat([self.joint_X_p, parent_xform])
        self.joint_X_c = torch.cat([self.joint_X_c, child_xform])

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
        self.is_robot = True
        self.robot_name = name

        self.robot_q = torch.cat([pos, rot]).to(self.device)
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

            self.robot_inertia = self.robot_inertia + R @ I_m @ R.T

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
                                        dtype=torch.float32, device=self.device)
        return control

    def serialize(self):

        def tensor_to_list(tensor):
            return tensor.cpu().tolist() if tensor.numel() > 0 else []

        data = {
            "body_count":
            self.body_count,
            "joint_count":
            self.joint_count,
            "bodies": [{
                "name":
                self.body_name[i],
                "shape":
                self.body_shapes[i].serialize(),
                "q":
                tensor_to_list(self.body_q[i]),
                "qd":
                tensor_to_list(self.body_qd[i]),
                "f":
                tensor_to_list(self.body_f[i]),
                "collision_points":
                tensor_to_list(
                    self.body_collision_points.get(i, torch.empty(0)))
            } for i in range(self.body_count)],
        }

        # Add terrain data if available
        if self.terrain is not None:
            data["terrain"] = self.terrain.serialize()

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


def sphere_collision_points(radius: float, n_points: int, device: torch.device = torch.device('cpu')):
    return _generate_fibonacci_sphere_points(radius, n_points).to(device)


def cylinder_collision_points(radius: float, height: float, n_base_points: int,
                              n_surface_points: int, device: torch.device = torch.device('cpu')):
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

    return vertices.to(device)


def box_collision_points(hx: float, hy: float, hz: float, n_points: int, device: torch.device = torch.device('cpu')):
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

    return points.to(device)
