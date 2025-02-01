import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
from mujoco_base import MuJoCoBase


class CylinderSimulation(MuJoCoBase):
    def __init__(self, xml_path):
        super().__init__(xml_path)

    def reset(self):
        # Set initial position and orientation
        self.data.qpos[:3] = [0.0, 0.0, 10.0]  # Initial position
        self.data.qpos[3:7] = [0.7071, -0.7071, 0.0, 0.0]  # Initial orientation (quaternion)

        # Set initial velocities
        self.data.qvel[:3] = [0.0, 0.0, 0.0]  # Linear velocity
        self.data.qvel[3:] = [1.0, 2.0, 3.0]  # Angular velocity

        # Configure the camera
        self.cam.azimuth = 90.0
        self.cam.distance = 40.0
        self.cam.elevation = -15
        self.cam.lookat = np.array([0.0, 0.0, 1.0])

    def simulate(self):
        while not glfw.window_should_close(self.window):
            simstart = self.data.time

            # Step the simulation
            while self.data.time - simstart < 1.0 / 60.0:
                mj.mj_step(self.model, self.data)

            # Get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Update the scene and render
            mj.mjv_updateScene(
                self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene
            )
            mj.mjr_render(viewport, self.scene, self.context)

            # Swap OpenGL buffers
            glfw.swap_buffers(self.window)

            # Process pending GUI events
            glfw.poll_events()

        glfw.terminate()


def main():
    xml_path = "./cylinder.xml"  # Path to your MuJoCo XML file
    simulation = CylinderSimulation(xml_path)
    simulation.reset()
    simulation.simulate()


if __name__ == "__main__":
    main()
