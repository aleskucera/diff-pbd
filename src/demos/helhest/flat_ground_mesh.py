import os

import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide, collide_terrain_friction
from pbd_torch.newton_engine import NonSmoothNewtonEngine
from pbd_torch.xpbd_engine import XPBDEngine
from pbd_torch.constants import ROT_IDENTITY, ROT_90_Y
from pbd_torch.transform import transform_multiply_batch
from pbd_torch.terrain import create_terrain_from_exr_file
from demos.helhest.utils import create_helhest_model
from tqdm import tqdm
from simview import SimulationScene
from simview import SimViewLauncher
from simview import BodyShapeType
from simview import SimViewBodyState

ENGINE = "NonSmoothNewton"  # Choose between "XPBD" and "NonSmoothNewton"

# Get the file path of the current script
curr_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(curr_dir, "..", "..", ".."))
OUTPUT_FILE = os.path.join(project_dir, "simulation", "helhest_flat_ground_mesh4.json")
HEIGHT_MAP = os.path.join(project_dir, "data", "blender", "ground4", "textures", "ant01.002_heightmap.exr")
ASSETS_DIR = os.path.join(project_dir, "data", "helhest")

device = torch.device("cpu")
WHEEL_TRANSFORM = torch.cat([
    torch.zeros(3, device=device),
    ROT_90_Y.flatten()]
).view(7, 1)

BASE_TRANSFORM = torch.cat([
    torch.tensor([0.2, 0.0, -0.44], device=device),
    ROT_IDENTITY.flatten()]
).view(7, 1)


def main():
    dt = 0.002
    n_steps = 500
    collision_margin = 0.0
    friction_collision_margin = 0.08

    terrain = create_terrain_from_exr_file(
        heightmap_path=HEIGHT_MAP,
        size_x=20.0,
        size_y=20.0,
        device=device,
        height_scale=0.25,
    )

    model, idxs = create_helhest_model(
        base_pos=(3.0, -0.0, 0.55),
        device=device,
        terrain=None,
        max_contacts_per_body=16,
    )

    if ENGINE == "XPBD":
        engine = XPBDEngine(model, pos_iters=5, device=device)
    elif ENGINE == "NonSmoothNewton":
        engine = NonSmoothNewtonEngine(model, iterations=200, device=device)
    else:
        raise ValueError(f"Unknown engine: {ENGINE}")

    sim_scene = SimulationScene(
        batch_size=1,
        scalar_names=['x', 'y', 'z'],
        dt=dt,
    )

    # Create [0, 0, 1] like the terrain.normals.permute(2,0,1)
    terrain_normals = torch.zeros_like(terrain.normals)
    terrain_normals[:, :, 2] = 1.0

    sim_scene.create_terrain(
        heightmap=torch.zeros_like(terrain.height_data),
        normals=terrain_normals.permute(2, 0, 1),
        x_lim=(-10, 10),
        y_lim=(-10, 10),
    )

    sim_scene.create_body(
        body_name=model.body_name[idxs['base']],
        shape_type=BodyShapeType('pointcloud'),
        points=model.body_collision_points[idxs['base']],
    )

    sim_scene.create_body(
        body_name='base_mesh',
        shape_type=BodyShapeType.MESH,
        mesh_path=os.path.join(ASSETS_DIR, "base.obj"),
    )

    sim_scene.create_body(
        body_name=model.body_name[idxs['left_wheel']],
        shape_type=BodyShapeType('pointcloud'),
        points=model.body_collision_points[idxs['left_wheel']],
    )

    sim_scene.create_body(
        body_name='left_wheel_mesh',
        shape_type=BodyShapeType.MESH,
        mesh_path=os.path.join(ASSETS_DIR, "wheel.obj"),
    )

    sim_scene.create_body(
        body_name=model.body_name[idxs['right_wheel']],
        shape_type=BodyShapeType('pointcloud'),
        points=model.body_collision_points[idxs['right_wheel']],
    )

    sim_scene.create_body(
        body_name='right_wheel_mesh',
        shape_type=BodyShapeType.MESH,
        mesh_path=os.path.join(ASSETS_DIR, "wheel.obj"),
    )

    sim_scene.create_body(
        body_name=model.body_name[idxs['back_wheel']],
        shape_type=BodyShapeType('pointcloud'),
        points=model.body_collision_points[idxs['back_wheel']],
    )

    sim_scene.create_body(
        body_name='back_wheel_mesh',
        shape_type=BodyShapeType.MESH,
        mesh_path=os.path.join(ASSETS_DIR, "wheel.obj"),
    )


    control = model.control()
    states = [model.state() for _ in range(n_steps)]

    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        collide(model, states[i], collision_margin=collision_margin)
        # collide_terrain_friction(model, states[i], collision_margin=friction_collision_margin)

        if i < 1500:
            control.add_actuation(idxs['left_wheel_joint'], 0.5)
            control.add_actuation(idxs['right_wheel_joint'], -0.5)
        elif i > 1500 and i < 2200:
            control.add_actuation(idxs['left_wheel_joint'], -15.5)
            control.add_actuation(idxs['right_wheel_joint'], 15.5)
        elif i > 2200 and i < 2600:
            control.add_actuation(idxs['left_wheel_joint'], -10.0)
            control.add_actuation(idxs['right_wheel_joint'], 10.0)
        elif i > 2600 and i < 3000:
            control.add_actuation(idxs['left_wheel_joint'], -6.0)
            control.add_actuation(idxs['right_wheel_joint'], 6.0)
        elif i > 3000 and i < 3200:
            control.add_actuation(idxs['left_wheel_joint'], -3.0)
            control.add_actuation(idxs['right_wheel_joint'], 3.0)
        elif i > 3200 and i < 4000:
            control.add_actuation(idxs['left_wheel_joint'], 8.0)
            control.add_actuation(idxs['right_wheel_joint'], -8.0)
        elif i > 3000 and i < 4000:
            control.add_actuation(idxs['left_wheel_joint'], 1.0)
            control.add_actuation(idxs['right_wheel_joint'], -5.0)
        else:
            control.add_actuation(idxs['left_wheel_joint'], 0.5)
            control.add_actuation(idxs['right_wheel_joint'], -0.5)


        engine.simulate_xitorch(states[i], states[i + 1], control, dt)

        body_states = []
        for body in range(model.body_count):
            state = SimViewBodyState(
                body_name=model.body_name[body],
                position=states[i].body_q[body, :3].flatten(),
                orientation=states[i].body_q[body, 3:].flatten(),
            )
            body_states.append(state)

        # Add the mesh body states with the correct transform
        base_mesh_body_q = transform_multiply_batch(
            states[i].body_q[idxs['base']],
            BASE_TRANSFORM
        )

        base_mesh_state = SimViewBodyState(
            body_name='base_mesh',
            position=base_mesh_body_q[:3].flatten(),
            orientation=base_mesh_body_q[3:].flatten(),
        )
        body_states.append(base_mesh_state)

        wheel_mesh_body_q = transform_multiply_batch(
            states[i].body_q[idxs['left_wheel']],
            WHEEL_TRANSFORM
        )

        wheel_mesh_state = SimViewBodyState(
            body_name='left_wheel_mesh',
            position=wheel_mesh_body_q[:3].flatten(),
            orientation=wheel_mesh_body_q[3:].flatten(),
        )

        body_states.append(wheel_mesh_state)
        wheel_mesh_body_q = transform_multiply_batch(
            states[i].body_q[idxs['right_wheel']],
            WHEEL_TRANSFORM
        )

        wheel_mesh_state = SimViewBodyState(
            body_name='right_wheel_mesh',
            position=wheel_mesh_body_q[:3].flatten(),
            orientation=wheel_mesh_body_q[3:].flatten(),
        )
        body_states.append(wheel_mesh_state)

        wheel_mesh_body_q = transform_multiply_batch(
            states[i].body_q[idxs['back_wheel']],
            WHEEL_TRANSFORM
        )

        wheel_mesh_state = SimViewBodyState(
            body_name='back_wheel_mesh',
            position=wheel_mesh_body_q[:3].flatten(),
            orientation=wheel_mesh_body_q[3:].flatten(),
        )
        body_states.append(wheel_mesh_state)

        sim_scene.add_state(
            time=i * dt,
            body_states=body_states,
            scalar_values={'x': states[i].body_q[idxs['base'], 0],
                           'y': states[i].body_q[idxs['base'], 1],
                           'z': states[i].body_q[idxs['base'], 2]},
        )

    # Save the simulation data
    sim_scene.save(OUTPUT_FILE)

    viewer_launcher = SimViewLauncher(OUTPUT_FILE)
    viewer_launcher.launch()


if __name__ == "__main__":
    main()

    viewer_launcher = SimViewLauncher(OUTPUT_FILE)
    viewer_launcher.launch()