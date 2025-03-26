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
from pbd_torch.utils import *
from pbd_torch.utils import box_inertia

# Define new types for the simulation
Vector3 = NewType("Vector3", torch.Tensor)  # (3, ) tensor
Transform = NewType("Transform", torch.Tensor)  # (7, ) tensor
Matrix3x3 = NewType("Matrix3x3", torch.Tensor)  # (3, 3) tensor
Quaternion = NewType("Quaternion", torch.Tensor)  # (4, ) tensor

CUSTOM_SHAPE = 0
BOX_SHAPE = 1
SPHERE_SHAPE = 2
CYLINDER_SHAPE = 3

JOINT_MODE_FORCE = 0
JOINT_MODE_TARGET_POSITION = 1
JOINT_MODE_TARGET_VELOCITY = 2


@dataclass
class Shape:
    type: int

    def serialize(self):
        return asdict(self)

@dataclass
class Custom(Shape):
    type: int = field(init=False, default=CYLINDER_SHAPE)

@dataclass
class Box(Shape):
    hx: float
    hy: float
    hz: float
    type: int = field(init=False, default=BOX_SHAPE)


@dataclass
class Sphere(Shape):
    radius: float
    type: int = field(init=False, default=SPHERE_SHAPE)


@dataclass
class Cylinder(Shape):
    radius: float
    height: float
    type: int = field(init=False, default=CYLINDER_SHAPE)

class State:

    def __init__(self):
        self.time = 0.0

        # Body-related attributes
        self.body_q: torch.Tensor = None  # [body_count, 7, 1]
        self.body_qd: torch.Tensor = None  # [body_count, 6, 1]
        self.body_f: torch.Tensor = None  # [body_count, 6, 1]

        # Joint-related attributes
        self.joint_q: torch.Tensor = None  # [joint_count, 1]
        self.joint_qd: torch.Tensor = None  # [joint_count, 1]
        self.joint_act: torch.Tensor = None  # [joint_count, 1]

        # Contact points: Contact points are vectors from the center of mass of the body to the contact point. They are stored in the body frame.
        # Contact normals: Contact normals are vectors in which direction the contact force is applied (from body A towards body B). They are stored in the body frame.
        self.contact_mask: torch.Tensor = None  # [body_count, max_contacts]
        self.contact_normals: torch.Tensor = None  # [body_count, max_contacts, 3, 1]
        self.contact_points: torch.Tensor = None  # [body_count, max_contacts, 3, 1]
        self.contact_point_indices: torch.Tensor = None  # [body_count, max_contacts]
        self.contact_points_ground: torch.Tensor = (
            None  # [body_count, max_contacts, 3, 1]
        )

    @property
    def body_count(self):
        return self.body_q.shape[0]

    @property
    def joint_count(self):
        return self.joint_q.shape[0]

    def serialize(self, model: "Model"):

        def tensor_to_list(tensor):
            return tensor.detach().cpu().tolist() if tensor.numel() > 0 else []

        # How to keep the tensor of the similar shape?
        if self.contact_point_indices is not None:
            contacts = [
                self.contact_point_indices[i][self.contact_mask[i]]
                for i in range(self.body_count)
            ]
        else:
            contacts = [
                torch.zeros(0, dtype=torch.int32) for _ in range(self.body_count)
            ]

        data = {
            "time": self.time,
            "bodies": [
                {
                    "name": model.body_name[i],
                    "transform": [tensor_to_list(self.body_q[i].squeeze(1)), tensor_to_list(self.body_q[i].squeeze(1))], # [B, 7]
                    "velocity": [tensor_to_list(self.body_qd[i].squeeze(1)), tensor_to_list(self.body_qd[i].squeeze(1))], # [B, 6]
                    "force": [tensor_to_list(self.body_f[i].squeeze(1)), tensor_to_list(self.body_f[i].squeeze(1))], # [B, 6]
                    "contacts": [tensor_to_list(contacts[i]), tensor_to_list(contacts[i])], # [B, num_contacts] (indices)
                    "energy": [0.1, 0.1],
                }
                for i in range(self.body_count)
            ],
        }

        return data


class Control:

    def __init__(self):
        self.joint_act = torch.zeros((0,), dtype=torch.float32)


class Model:

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        requires_grad: bool = False,
        terrain: Terrain = None,
        dynamic_friction_threshold: float = 0.0,
        max_contacts_per_body: int = 20,
    ):
        self.device = device
        self.requires_grad = requires_grad

        self.terrain = terrain

        self.max_contacts_per_body = max_contacts_per_body
        self.dynamic_friction_threshold = dynamic_friction_threshold

        # ====================== Body-related attributes ======================
        self.body_name: List[str] = []
        self.body_shapes: List[Shape] = []

        self.body_q = torch.zeros(
            (0, 7, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )  # [body_count, 7, 1]
        self.body_qd = torch.zeros(
            (0, 6, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )  # [body_count, 6, 1]
        self.body_f = torch.zeros(
            (0, 6, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )  # [body_count, 6, 1]

        self.body_mass = torch.zeros(
            (0, 1, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )  # [body_count, 1]
        self.body_inv_mass = torch.zeros(
            (0, 1, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )  # [body_count, 1]
        self.body_inertia = torch.zeros(
            (0, 3, 3), dtype=torch.float32, device=device, requires_grad=requires_grad
        )  # [body_count, 3, 3]
        self.body_inv_inertia = torch.zeros(
            (0, 3, 3), dtype=torch.float32, device=device, requires_grad=requires_grad
        )  # [body_count, 3, 3]
        self.body_collision_points: dict[int, torch.Tensor] = {}
        self.coll_points = torch.zeros((0, 3, 1), dtype=torch.float32, device=device)
        self.coll_points_body_idx = torch.zeros((0,), dtype=torch.int32, device=device)

        self.restitution = torch.zeros(
            (0, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )  # [body_count, 1]
        self.static_friction = torch.zeros(
            (0, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )  # [body_count, 1]
        self.dynamic_friction = torch.zeros(
            (0, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )  # [body_count, 1]

        # ====================== Joint-related attributes ======================
        self.joint_name = []
        self.joint_q = torch.zeros(
            (0, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )
        self.joint_qd = torch.zeros(
            (0, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )
        self.joint_act = torch.zeros(
            (0, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )
        self.joint_axis_mode = torch.zeros((0,), dtype=torch.int32, device=device)
        self.joint_ke = torch.zeros(
            (0, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )
        self.joint_kd = torch.zeros(
            (0, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )
        self.joint_parent = torch.zeros((0,), dtype=torch.int32, device=device)
        self.joint_child = torch.zeros((0,), dtype=torch.int32, device=device)

        self.joint_X_p = torch.zeros(
            (0, 7, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )
        self.joint_X_c = torch.zeros(
            (0, 7, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )
        self.joint_axis = torch.zeros(
            (0, 3, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )
        self.joint_basis = torch.zeros(
            (0, 3, 3), dtype=torch.float32, device=device, requires_grad=requires_grad
        )
        self.joint_compliance = torch.zeros(
            (0, 1), dtype=torch.float32, device=device, requires_grad=requires_grad
        )

        # System properties
        self.gravity = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, -9.81], dtype=torch.float32, device=device
        ).view(6, 1)

    @property
    def body_count(self):
        return self.body_q.shape[0]

    @property
    def joint_count(self):
        return self.joint_q.shape[0]

    @property
    def mass_matrix(self):
        identity_blocks = (
            torch.eye(3, device=self.device).unsqueeze(0).repeat(self.body_count, 1, 1)
        )
        mass_blocks = identity_blocks * self.body_mass.view(-1, 1, 1)

        mass_matrix = torch.zeros((self.body_count, 6, 6), device=self.device)

        mass_matrix[:, :3, :3] = self.body_inertia
        mass_matrix[:, 3:, 3:] = mass_blocks

        return mass_matrix

    def _append_body_mass(self, m: float):
        assert m > 0, "Body mass must be positive"
        assert isinstance(m, float), "Body mass must be a float"

        body_mass = torch.tensor([m], device=self.device).view(1, 1, 1)
        self.body_mass = torch.cat([self.body_mass, body_mass]).requires_grad_(
            self.requires_grad
        )
        body_inv_mass = torch.tensor([1 / m], device=self.device).view(1, 1, 1)
        self.body_inv_mass = torch.cat(
            [self.body_inv_mass, body_inv_mass]
        ).requires_grad_(self.requires_grad)

    def _append_body_inertia(self, inertia: torch.Tensor):
        assert inertia.shape == (3, 3), "Inertia must be a 3x3 matrix"

        self.body_inertia = torch.cat(
            [self.body_inertia, inertia.unsqueeze(0)]
        ).requires_grad_(self.requires_grad)
        self.body_inv_inertia = torch.cat(
            [self.body_inv_inertia, torch.inverse(inertia).unsqueeze(0)]
        ).requires_grad_(self.requires_grad)

    def add_box(
        self,
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
        n_collision_points: int = 200,
    ) -> int:
        assert m > 0, "Body mass must be positive"
        assert hx > 0, "Box hx must be positive"
        assert hy > 0, "Box hy must be positive"
        assert hz > 0, "Box hz must be positive"

        body_idx = self.body_count

        self.body_shapes.append(Box(hx=hx, hy=hy, hz=hz))
        self.body_name.append(name or f"body_{self.body_count}")

        body_q = torch.cat([pos, rot]).view(1, 7, 1).to(self.device)  # [1, 7, 1]
        self.body_q = torch.cat([self.body_q, body_q]).requires_grad_(
            self.requires_grad
        )
        doby_qd = torch.zeros((1, 6, 1), device=self.device)  # [1, 6, 1]
        self.body_qd = torch.cat([self.body_qd, doby_qd]).requires_grad_(
            self.requires_grad
        )
        body_f = torch.zeros((1, 6, 1), device=self.device)  # [1, 6, 1]
        self.body_f = torch.cat([self.body_f, body_f]).requires_grad_(
            self.requires_grad
        )

        self._append_body_mass(m)
        inertia = box_inertia(hx, hy, hz, m, self.device, self.requires_grad)  # [3, 3]
        self._append_body_inertia(inertia)

        coll_points = box_collision_points(
            hx, hy, hz, n_collision_points, self.device
        )  # [n, 3]
        self.body_collision_points[body_idx] = coll_points
        self.coll_points = torch.cat([self.coll_points, coll_points.unsqueeze(-1)])
        self.coll_points_body_idx = torch.cat(
            [
                self.coll_points_body_idx,
                torch.full((coll_points.shape[0],), body_idx, device=self.device),
            ]
        )

        r = torch.tensor([restitution], device=self.device).view(1, 1)
        self.restitution = torch.cat([self.restitution, r]).requires_grad_(
            self.requires_grad
        )
        s = torch.tensor([static_friction], device=self.device).view(1, 1)
        self.static_friction = torch.cat([self.static_friction, s]).requires_grad_(
            self.requires_grad
        )
        d = torch.tensor([dynamic_friction], device=self.device).view(1, 1)
        self.dynamic_friction = torch.cat([self.dynamic_friction, d]).requires_grad_(
            self.requires_grad
        )

        return body_idx

    def add_sphere(
        self,
        m: float,
        radius: float = 1.0,
        name: Union[str, None] = None,
        pos: Vector3 = Vector3(torch.zeros(3)),
        rot: Quaternion = Quaternion(torch.tensor([1, 0, 0, 0])),
        restitution: float = 0.5,
        static_friction: float = 0.5,
        dynamic_friction: float = 0.5,
        n_collision_points: int = 1000,
    ) -> int:
        assert m > 0, "Body mass must be positive"
        assert radius > 0, "Sphere radius must be positive"

        body_idx = self.body_count

        self.body_shapes.append(Sphere(radius=radius))
        self.body_name.append(name or f"body_{self.body_count}")

        body_q = torch.cat([pos, rot]).to(self.device).view(1, 7, 1)
        self.body_q = torch.cat([self.body_q, body_q]).requires_grad_(
            self.requires_grad
        )
        doby_qd = torch.zeros((1, 6, 1), device=self.device)
        self.body_qd = torch.cat([self.body_qd, doby_qd]).requires_grad_(
            self.requires_grad
        )
        body_f = torch.zeros((1, 6, 1), device=self.device)
        self.body_f = torch.cat([self.body_f, body_f]).requires_grad_(
            self.requires_grad
        )

        self._append_body_mass(m)
        inertia = sphere_inertia(radius, m, self.device, self.requires_grad)  # [3, 3]
        self._append_body_inertia(inertia)

        coll_points = sphere_collision_points(
            radius, n_collision_points, self.device
        )  # [n, 3]
        self.body_collision_points[body_idx] = coll_points
        self.coll_points = torch.cat([self.coll_points, coll_points.unsqueeze(-1)])
        self.coll_points_body_idx = torch.cat(
            [
                self.coll_points_body_idx,
                torch.full((coll_points.shape[0],), body_idx, device=self.device),
            ]
        )

        r = torch.tensor([restitution], device=self.device).view(1, 1)
        self.restitution = torch.cat([self.restitution, r]).requires_grad_(
            self.requires_grad
        )
        s = torch.tensor([static_friction], device=self.device).view(1, 1)
        self.static_friction = torch.cat([self.static_friction, s]).requires_grad_(
            self.requires_grad
        )
        d = torch.tensor([dynamic_friction], device=self.device).view(1, 1)
        self.dynamic_friction = torch.cat([self.dynamic_friction, d]).requires_grad_(
            self.requires_grad
        )

        return body_idx

    def add_cylinder(
        self,
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
        n_collision_points_surface: int = 1000,
    ) -> int:
        assert m > 0, "Body mass must be positive"
        assert radius > 0, "Cylinder radius must be positive"
        assert height > 0, "Cylinder height must be positive"

        body_idx = self.body_count

        self.body_shapes.append(Cylinder(radius=radius, height=height))
        self.body_name.append(name or f"body_{self.body_count}")

        body_q = torch.cat([pos, rot]).to(self.device).view(1, 7, 1)
        self.body_q = torch.cat([self.body_q, body_q])
        doby_qd = torch.zeros((1, 6, 1), device=self.device)
        self.body_qd = torch.cat([self.body_qd, doby_qd])
        body_f = torch.zeros((1, 6, 1), device=self.device)
        self.body_f = torch.cat([self.body_f, body_f])

        self._append_body_mass(m)
        inertia = cylinder_inertia(radius, height, m, self.device, self.requires_grad)
        self._append_body_inertia(inertia)

        coll_points = cylinder_collision_points(
            radius,
            height,
            n_collision_points_base,
            n_collision_points_surface,
            self.device,
        )  # [n, 3]
        self.body_collision_points[body_idx] = coll_points
        self.coll_points = torch.cat([self.coll_points, coll_points.unsqueeze(-1)])
        self.coll_points_body_idx = torch.cat(
            [
                self.coll_points_body_idx,
                torch.full((coll_points.shape[0],), body_idx, device=self.device),
            ]
        )

        r = torch.tensor([restitution], device=self.device).view(1, 1)
        self.restitution = torch.cat([self.restitution, r])
        s = torch.tensor([static_friction], device=self.device).view(1, 1)
        self.static_friction = torch.cat([self.static_friction, s])
        d = torch.tensor([dynamic_friction], device=self.device).view(1, 1)
        self.dynamic_friction = torch.cat([self.dynamic_friction, d])

        return body_idx

    def add_hinge_joint(
        self,
        parent: int,
        child: int,
        axis: Vector3,
        name: Union[str, None] = None,
        q: float = 0.0,
        qd: float = 0.0,
        act: float = 0.0,
        parent_xform: torch.Tensor = TRANSFORM_IDENTITY,
        child_xform: torch.Tensor = TRANSFORM_IDENTITY,
        compliance: float = 0.0,
    ) -> int:
        assert parent >= -1 and parent < self.body_count, "Parent index out of bounds"
        assert child >= 0 and child < self.body_count, "Child index out of bounds"

        # Add joint name
        self.joint_name.append(name or f"joint_{self.joint_count}")

        # Add joint properties
        joint_q = torch.tensor([q], device=self.device).view(1, 1)
        self.joint_q = torch.cat([self.joint_q, joint_q])
        joint_qd = torch.tensor([qd], device=self.device).view(1, 1)
        self.joint_qd = torch.cat([self.joint_qd, joint_qd])
        joint_act = torch.tensor([act], device=self.device).view(1, 1)
        self.joint_act = torch.cat([self.joint_act, joint_act])

        # Add parent and child indices
        self.joint_parent = torch.cat(
            [self.joint_parent, torch.tensor([parent], device=self.device)]
        )
        self.joint_child = torch.cat(
            [self.joint_child, torch.tensor([child], device=self.device)]
        )

        # Add joint axis
        joint_axis = (axis / torch.norm(axis)).to(self.device).view(1, 3, 1)
        self.joint_axis = torch.cat([self.joint_axis, joint_axis])

        # Add joint compliance
        joint_compliance = torch.tensor([compliance], device=self.device).view(1, 1)
        self.joint_compliance = torch.cat([self.joint_compliance, joint_compliance])

        parent_xform = parent_xform.to(self.device).view(1, 7, 1)
        child_xform = child_xform.to(self.device).view(1, 7, 1)
        # Initialize joint transforms (you might want to modify these based on your needs)
        self.joint_X_p = torch.cat([self.joint_X_p, parent_xform])
        self.joint_X_c = torch.cat([self.joint_X_c, child_xform])

        # Set axis mode to force by default
        self.joint_axis_mode = torch.cat(
            [self.joint_axis_mode, torch.tensor([JOINT_MODE_FORCE], device=self.device)]
        )

        # Set default stiffness and damping
        self.joint_ke = torch.cat([self.joint_ke, torch.zeros((1, 1), device=self.device)])
        self.joint_kd = torch.cat([self.joint_kd, torch.zeros((1, 1), device=self.device)])

        return self.joint_count - 1

    def state(self):
        state = State()
        state.body_q = self.body_q.clone()
        state.body_qd = self.body_qd.clone()
        state.body_f = self.body_f.clone()

        state.joint_q = self.joint_q.clone()
        state.joint_qd = self.joint_qd.clone()
        state.joint_act = self.joint_act.clone()

        return state

    def control(self):
        control = Control()
        control.joint_act = torch.zeros(
            (self.joint_count,), dtype=torch.float32, device=self.device
        )
        return control

    def serialize(self):

        def tensor_to_list(tensor):
            return tensor.cpu().tolist() if tensor.numel() > 0 else []

        data = {
            "simBatches": 2,
            "scalarNames": ["energy"],
            "bodies": [
                {
                    "name": self.body_name[i],
                    "shape": self.body_shapes[i].serialize(),
                    "transform": [tensor_to_list(self.body_q[i].squeeze(1)), tensor_to_list(self.body_q[i].squeeze(1))], # [batch_size, 7]
                    "velocity": [tensor_to_list(self.body_qd[i].squeeze(1)), tensor_to_list(self.body_qd[i].squeeze(1))], # [batch_size, 6]
                    "force": [tensor_to_list(self.body_f[i].squeeze(1)), tensor_to_list(self.body_f[i].squeeze(1))], # [batch_size, 3]
                    "bodyPoints": tensor_to_list(self.body_collision_points.get(i, torch.empty(0))), # [num_points, 3]
                }
                for i in range(self.body_count)
            ],
        }

        # Add terrain data if available
        if self.terrain is not None:
            data["terrain"] = self.terrain.serialize()

        return data
