from dns.rdatatype import is_singleton
from simview import BodyShapeType
from pbd_torch.terrain import create_terrain_from_exr_file
from pbd_torch.utils import sphere_collision_points
from simview import SimulationScene
from simview import SimViewLauncher

def test_simview():
    terrain = create_terrain_from_exr_file(
        heightmap_path="/home/kuceral4/school/diff-pbd/data/blender/ground4/textures/ant01.002_heightmap.exr",
        size_x=40.0,
        size_y=40.0
    )

    sim_scene = SimulationScene(batch_size=1,
                              scalar_names=['loss'],
                              dt=0.01)

    sim_scene.create_terrain(
        heightmap=terrain.height_data,
        normals=terrain.normals.permute(2, 0, 1),
        x_lim=(-20, 20),
        y_lim=(-20, 20),
    )

    sim_scene.create_body(
        body_name='sphere',
        shape_type=BodyShapeType('pointcloud'),
        points=sphere_collision_points(1.0, 400)
    )

    viewer_launcher = SimViewLauncher(sim_scene)
    viewer_launcher.launch()


if __name__ == "__main__":
    test_simview()
