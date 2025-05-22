import os

import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from pbd_torch.newton_engine import NonSmoothNewtonEngine
from pbd_torch.terrain import create_terrain_from_exr_file
from tqdm import tqdm
from simview import SimulationScene
from simview import SimViewLauncher
from simview import BodyShapeType
from simview import SimViewBodyState


def main():
    dt = 0.005
    n_steps = 1000
    device = torch.device("cpu")
    output_file = "/home/kuceral4/school/simview/simview/simulation.json"

    terrain = create_terrain_from_exr_file(
        heightmap_path="/home/kuceral4/school/diff-pbd/data/blender/ground4/textures/ant01.002_heightmap.exr",
        size_x=10.0,
        size_y=10.0,
        device=device,
        height_scale=0.25
    )

    model = Model(terrain=terrain, max_contacts_per_body=36)

    sphere = model.add_sphere(
        m=1.0,
        radius=0.3,
        name="sphere",
        pos=Vector3(torch.tensor([0.5, 0.5, 1.6])),
        rot=Quaternion(ROT_IDENTITY),
        restitution=0.2,
        dynamic_friction=0.8,
        n_collision_points=800,
    )

    engine = NonSmoothNewtonEngine(model, iterations=100)

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
        body_name=model.body_name[sphere],
        shape_type=BodyShapeType('pointcloud'),
        points=model.body_collision_points[sphere]
    )

    # sim_scene.create_body(
    #     body_name=model.body_name[sphere],
    #     shape_type=BodyShapeType.MESH,
    #     mesh_path="/home/kuceral4/test2/wheel.obj",
    # )

    # Simulate the model
    for i in tqdm(range(n_steps - 1), desc="Simulating"):
        collide(model, states[i], collision_margin=0.0)
        engine.simulate(states[i], states[i + 1], control, dt)

        body_states = []
        for body in range(model.body_count):
            state = SimViewBodyState(
                body_name=model.body_name[body],
                position=states[i].body_q[body, :3].flatten(),
                orientation=states[i].body_q[body, 3:].flatten(),
            )
            body_states.append(state)

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
