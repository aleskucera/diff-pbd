import os

import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide
from pbd_torch.constants import ROT_IDENTITY, ROT_NEG_90_X, ROT_90_Z, ROT_90_Y, ROT_90_X
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from pbd_torch.transform import transform_multiply_batch, normalize_quat_batch, normalize_quat
from pbd_torch.newton_engine import NonSmoothNewtonEngine
from pbd_torch.terrain import create_terrain_from_exr_file
from tqdm import tqdm
from simview import SimulationScene
from simview import SimViewLauncher
from simview import BodyShapeType
from simview import SimViewBodyState


def main():
    dt = 0.01
    n_steps = 400
    device = torch.device("cpu")
    output_file = os.path.join('simulation', 'wheel.json')

    terrain = create_terrain_from_exr_file(
        heightmap_path="/home/kuceral4/school/diff-pbd/data/blender/ground4/textures/ant01.002_heightmap.exr",
        size_x=10.0,
        size_y=10.0,
        device=device,
        height_scale=0.25
    )

    model = Model(terrain=terrain, max_contacts_per_body=36)

    cylinder = model.add_cylinder(
        m=1.0,
        radius=0.937/2.0,
        height=0.142,
        name="cylinder",
        pos=Vector3(torch.tensor([-0.0, -0.3, 1.5])),
        rot=Quaternion(ROT_NEG_90_X),
        restitution=0.5,
        dynamic_friction=0.7,
        n_collision_points_base=256,
        n_collision_points_surface=256
    )

    engine = NonSmoothNewtonEngine(model, iterations=150)

    control = model.control()
    states = [model.state() for _ in range(n_steps)]

    sim_scene = SimulationScene(
        batch_size=1,
        scalar_names=['loss'],
        dt=dt,
    )

    sim_scene.create_terrain(
        heightmap=terrain.height_data,
        normals=terrain.normals.permute(2, 0, 1),
        x_lim=(-5, 5),
        y_lim=(-5, 5),
    )

    sim_scene.create_body(
        body_name=model.body_name[cylinder],
        shape_type=BodyShapeType('pointcloud'),
        points=model.body_collision_points[cylinder]
    )

    sim_scene.create_body(
        body_name='cylinder_mesh',
        shape_type=BodyShapeType.MESH,
        mesh_path="/home/kuceral4/test2/wheel.obj",
    )

    # Relative transform for the mesh
    mesh_transform = torch.cat([
        torch.zeros(3, device=device),
        ROT_90_Y.flatten()]
    ).view(7, 1)

    # Simulate the model
    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        collide(model, states[i], collision_margin=0.0)
        engine.simulate(states[i], states[i + 1], control, dt)

        body_states = []

        cylinder_state = SimViewBodyState(
            body_name=model.body_name[cylinder],
            position=states[i].body_q[cylinder, :3].flatten(),
            orientation=states[i].body_q[cylinder, 3:].flatten(),
        )
        body_states.append(cylinder_state)

        cylinder_mesh_body_q = transform_multiply_batch(
            states[i].body_q[cylinder],
            mesh_transform
        )
        # cylinder_mesh_body_q[3:] = normalize_quat(cylinder_mesh_body_q[3:])
        cylinder_mesh_state = SimViewBodyState(
            body_name='cylinder_mesh',
            position=cylinder_mesh_body_q[:3].flatten(),
            orientation=cylinder_mesh_body_q[3:].flatten(),
        )
        body_states.append(cylinder_mesh_state)

        sim_scene.add_state(
            time=i * dt,
            body_states=body_states,
            scalar_values={"loss": torch.tensor([0.0])},
        )

    # Save the simulation data
    sim_scene.save(output_file)

    viewer_launcher = SimViewLauncher(output_file)
    viewer_launcher.launch()

if __name__ == "__main__":
    main()
    # viewer_launcher = SimViewLauncher(output_file)
    # viewer_launcher.launch()
