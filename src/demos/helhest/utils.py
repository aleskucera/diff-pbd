from typing import Tuple, Dict, Any, Union

import torch
from pbd_torch.constants import ROT_90_X
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.constants import ROT_NEG_90_X
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from pbd_torch.terrain import Terrain

def create_helhest_model(
    base_pos: Tuple,
    device: torch.device,
    terrain: Union[Terrain, None],
    max_contacts_per_body: int
) -> Tuple[Model, Dict[str, Any]]:
    model = Model(
        device=device,
        terrain=terrain,
        max_contacts_per_body=max_contacts_per_body,
    )

    # Add robot base
    base = model.add_box(
        m=1.0,
        hx=0.5,
        hy=0.3,
        hz=0.15,
        name="box",
        pos=Vector3(torch.tensor([base_pos[0], base_pos[1], base_pos[2]])),
        rot=Quaternion(ROT_IDENTITY),
        n_collision_points=500,
        restitution=0.2,
        dynamic_friction=0.8,
    )

    # Add left wheel
    left_wheel = model.add_cylinder(
        m=1.0,
        radius=0.937/2.0,
        height=0.142,
        name="left_wheel",
        pos=Vector3(torch.tensor([base_pos[0] + 0.4, base_pos[1] + 0.52, base_pos[2]])),
        rot=Quaternion(ROT_NEG_90_X),
        n_collision_points_base=128,
        n_collision_points_surface=128,
        restitution=0.2,
        dynamic_friction=0.8,
    )

    # Add right wheel
    right_wheel = model.add_cylinder(
        m=1.0,
        radius=0.937/2.0,
        height=0.142,
        name="right_wheel",
        pos=Vector3(torch.tensor([base_pos[0] + 0.4, base_pos[1] - 0.52, base_pos[2]])),
        rot=Quaternion(ROT_90_X),
        n_collision_points_base=128,
        n_collision_points_surface=128,
        restitution=0.2,
        dynamic_friction=0.8,
    )

    # Back wheel
    back_wheel = model.add_cylinder(
        m=1.0,
        radius=0.937/2.0,
        height=0.142,
        name="back_wheel",
        pos=Vector3(torch.tensor([base_pos[0] - 0.5, base_pos[1], base_pos[2]])),
        rot=Quaternion(ROT_NEG_90_X),
        n_collision_points_base=128,
        n_collision_points_surface=128,
        restitution=0.4,
        dynamic_friction=0.6,
    )

    # Add left hinge joint
    left_wheel_joint = model.add_hinge_joint(
        parent=base,
        child=left_wheel,
        axis=Vector3(torch.tensor([0.0, 0.0, 1.0])),
        name="left_wheel_joint",
        parent_trans=torch.cat((torch.tensor([0.4, 0.4, 0.0]), ROT_NEG_90_X)),
        child_trans=torch.tensor([0.0, 0.0, -0.12, 1.0, 0.0, 0.0, 0.0]),
        ke=50.0,
        kd=0.5,
    )

    # Add right hinge joint
    right_wheel_joint = model.add_hinge_joint(
        parent=base,
        child=right_wheel,
        axis=Vector3(torch.tensor([0.0, 0.0, 1.0])),
        name="right_wheel_joint",
        parent_trans=torch.cat((torch.tensor([0.4, -0.4, 0.0]), ROT_90_X)),
        child_trans=torch.tensor([0.0, 0.0, -0.12, 1.0, 0.0, 0.0, 0.0]),
        ke=50.0,
        kd=0.5,
    )

    # Add back hinge joint
    back_wheel_joint = model.add_hinge_joint(
        parent=base,
        child=back_wheel,
        axis=Vector3(torch.tensor([0.0, 0.0, 1.0])),
        name="back_wheel_joint",
        parent_trans=torch.cat((torch.tensor([-0.5, 0.0, 0.0]), ROT_NEG_90_X)),
        child_trans=torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ke=0.0,
        kd=0.0,
    )


    return model, {
        "base": base,
        "left_wheel": left_wheel,
        "right_wheel": right_wheel,
        "back_wheel": back_wheel,
        "left_wheel_joint": left_wheel_joint,
        "right_wheel_joint": right_wheel_joint,
        "back_wheel_joint": back_wheel_joint,
    }

