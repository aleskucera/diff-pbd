import os

import torch
from demos.utils import save_simulation
from pbd_torch.collision import collide
from pbd_torch.constants import ROT_NEG_90_X
from pbd_torch.model import Model
from pbd_torch.model import Quaternion
from pbd_torch.model import Vector3
from pbd_torch.newton_engine import NonSmoothNewtonEngine
from pbd_torch.terrain import create_terrain_from_exr_file
from tqdm import tqdm


def main():
    dt = 0.01
    n_steps = 400
    device = torch.device("cpu")
    output_file = os.path.join('simulation', 'cylinder.json')

    terrain = create_terrain_from_exr_file(
            heightmap_path="/home/kuceral4/school/diff-pbd/data/blender/ground4/textures/ant01.002_heightmap.exr",
            size_x=40.0,
            size_y=40.0,
            device=device,
        )

    model = Model(terrain=terrain, max_contacts_per_body=36)

    # Add robot base
    cylinder = model.add_cylinder(m=1.0,
                                  radius=1.0,
                                  height=0.2,
                                  name='cylinder',
                                  pos=Vector3(torch.tensor([0.0, 0.0, 4.5])),
                                  rot=Quaternion(ROT_NEG_90_X),
                                  restitution=0.5,
                                  dynamic_friction=0.8,
                                  n_collision_points_base=64,
                                  n_collision_points_surface=128)

    # # Add initial rotation to the cylinder
    # model.body_qd[cylinder, :3] = torch.tensor([0.0, 0.0, 0.0]).view(3, 1)
    # model.body_qd[cylinder, 3:] = torch.tensor([0.0, 0.0, 0.0]).view(3, 1)

    engine = NonSmoothNewtonEngine(model, iterations=100)

    control = model.control()
    states = [model.state() for _ in range(n_steps)]

    # Simulate the model
    for i in tqdm(range(n_steps - 1), desc='Simulating'):
        collide(model, states[i], collision_margin=0.0)
        engine.simulate(states[i], states[i + 1], control, dt)

    print(f'Saving simulation to {output_file}')
    save_simulation(model, states, output_file)


if __name__ == '__main__':
    main()
