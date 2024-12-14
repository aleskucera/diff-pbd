from tqdm import tqdm

from pbd_torch.old.body import Trajectory
from pbd_torch.correction import *
from pbd_torch.old.cuboid import Cuboid
from pbd_torch.old.cylinder import Cylinder
from pbd_torch.transform import *
from pbd_torch.utils import *
from pbd_torch.animation import *


class Robot:
    def __init__(self, box: Cuboid, left_wheel: Cylinder, right_wheel: Cylinder):
        self.box = box
        self.left_wheel = left_wheel
        self.right_wheel = right_wheel

        self.left_joint_box_r = torch.tensor([0, self.box.hy + 0.2, 0], dtype=torch.float32)
        self.left_joint_wheel_r = torch.tensor([0, 0, -self.left_wheel.height / 2 - 0.2], dtype=torch.float32)

        self.left_joint_box_u = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        self.left_joint_wheel_u = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

        self.right_joint_box_r = torch.tensor([0, -self.box.hy - 0.2, 0], dtype=torch.float32)
        self.right_joint_wheel_r = torch.tensor([0, 0, -self.right_wheel.height / 2 - 0.2], dtype=torch.float32)

        self.right_joint_box_u = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)
        self.right_joint_wheel_u = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    def plot(self, ax):
        assert torch.norm(self.box.q - 1.0) < 1e-6
        assert torch.norm(self.left_wheel.q - 1.0) < 1e-6
        assert torch.norm(self.right_wheel.q - 1.0) < 1e-6
        plot_axis(ax, self.box.x, self.box.q)
        self.box.plot_geometry(ax, self.box.x, self.box.q)

        plot_axis(ax, self.left_wheel.x, self.left_wheel.q)
        self.left_wheel.plot_geometry(ax, self.left_wheel.x, self.left_wheel.q)

        plot_axis(ax, self.right_wheel.x, self.right_wheel.q)
        self.right_wheel.plot_geometry(ax, self.right_wheel.x, self.right_wheel.q)

        left_r_a = LocalVector(r=self.left_joint_box_r.clone(), x=self.box.x.clone(), q=self.box.q.clone(), color=ORANGE)
        left_r_b = LocalVector(r=self.left_joint_wheel_r.clone(), x=self.left_wheel.x.clone(), q=self.left_wheel.q.clone(), color=ORANGE)

        plot_vectors(ax, [left_r_a, left_r_b])

        right_r_a = LocalVector(r=self.right_joint_box_r.clone(), x=self.box.x.clone(), q=self.box.q.clone(), color=ORANGE)
        right_r_b = LocalVector(r=self.right_joint_wheel_r.clone(), x=self.right_wheel.x.clone(), q=self.right_wheel.q.clone(), color=ORANGE)

        plot_vectors(ax, [right_r_a, right_r_b])

    def correct_joints(self):
        self.correct_joint_left_joint()
        self.correct_joint_right_joint()

    def correct_joint_left_joint(self):
        r_a, r_b = self.left_joint_box_r, self.left_joint_wheel_r
        m_a_inv, m_b_inv = self.box.m_inv, self.left_wheel.m_inv
        I_a_inv, I_b_inv = self.box.I_inv, self.left_wheel.I_inv

        u_a, u_b = self.left_joint_box_u, self.left_joint_wheel_u
        x_a, x_b = self.box.x, self.left_wheel.x
        q_a, q_b = self.box.q, self.left_wheel.q

        dq_a_ang, dq_b_ang = angular_delta(u_a, u_b, q_a, q_b, I_a_inv, I_b_inv)
        dx_a, dx_b, dq_a_pos, dq_b_pos, _  = positional_delta(x_a, x_b, q_a, q_b, r_a, r_b, m_a_inv, m_b_inv, I_a_inv, I_b_inv)

        self.box.dx_hinge += dx_a
        self.left_wheel.dx_hinge += dx_b

        self.box.dq_hinge += dq_a_ang + dq_a_pos
        self.left_wheel.dq_hinge += dq_b_ang + dq_b_pos

    def correct_joint_right_joint(self):
        r_a, r_b = self.right_joint_box_r, self.right_joint_wheel_r
        m_a_inv, m_b_inv = self.box.m_inv, self.right_wheel.m_inv
        I_a_inv, I_b_inv = self.box.I_inv, self.right_wheel.I_inv

        u_a, u_b = self.right_joint_box_u, self.right_joint_wheel_u
        x_a, x_b = self.box.x, self.right_wheel.x
        q_a, q_b = self.box.q, self.right_wheel.q

        dq_a_ang, dq_b_ang = angular_delta(u_a, u_b, q_a, q_b, I_a_inv, I_b_inv)
        dx_a, dx_b, dq_a_pos, dq_b_pos, _  = positional_delta(x_a, x_b, q_a, q_b, r_a, r_b, m_a_inv, m_b_inv, I_a_inv, I_b_inv)

        self.box.dx_hinge += dx_a
        self.right_wheel.dx_hinge += dx_b

        self.box.dq_hinge += dq_a_ang + dq_a_pos
        self.right_wheel.dq_hinge += dq_b_ang + dq_b_pos

    def simulate(self, n_steps: int, time: float):
        sim_time = torch.linspace(0, time, n_steps)

        self.box.dt = time / n_steps
        self.left_wheel.dt = time / n_steps
        self.right_wheel.dt = time / n_steps

        self.box.trajectory = Trajectory(sim_time)
        self.left_wheel.trajectory = Trajectory(sim_time)
        self.right_wheel.trajectory = Trajectory(sim_time)

        for i in tqdm(range(n_steps)):
            self.box.step = i
            self.left_wheel.step = i
            self.right_wheel.step = i

            self.box.integrate()
            self.left_wheel.integrate(torque=torch.tensor([0.0, 0.0, 40.0]))
            self.right_wheel.integrate(torque=torch.tensor([0.0, 0.0, -5.0]))

            # self.box.detect_collisions()
            # self.left_wheel.detect_collisions()
            # self.right_wheel.detect_collisions()

            for _ in range(30):
                self.box.detect_collisions()
                self.left_wheel.detect_collisions()
                self.right_wheel.detect_collisions()

                self.box.collision_delta()
                self.left_wheel.collision_delta()
                self.right_wheel.collision_delta()
                self.correct_joints()

                print(f"Iteration {i}")
                print(f"COLLISSIONS")
                print(f"Box x: {self.box.dx_collision}, q: {self.box.dq_collision}")
                print(f"Left Wheel x: {self.left_wheel.dx_collision}, q: {self.left_wheel.dq_collision}")
                print(f"Right Wheel x: {self.right_wheel.dx_collision}, q: {self.right_wheel.dq_collision}")

                print(f"FRICTION")
                print(f"Box x: {self.box.dx_friction}, q: {self.box.dq_friction}")
                print(f"Left Wheel x: {self.left_wheel.dx_friction}, q: {self.left_wheel.dq_friction}")
                print(f"Right Wheel x: {self.right_wheel.dx_friction}, q: {self.right_wheel.dq_friction}")

                print(f"JOINTS")
                print(f"Box x: {self.box.dx_hinge}, q: {self.box.dq_hinge}")
                print(f"Left Wheel x: {self.left_wheel.dx_hinge}, q: {self.left_wheel.dq_hinge}")
                print(f"Right Wheel x: {self.right_wheel.dx_hinge}, q: {self.right_wheel.dq_hinge}")

                self.box.add_deltas()
                self.left_wheel.add_deltas()
                self.right_wheel.add_deltas()

            self.box.update_velocity()
            self.left_wheel.update_velocity()
            self.right_wheel.update_velocity()

            self.box.solve_velocity()
            self.left_wheel.solve_velocity()
            self.right_wheel.solve_velocity()

            self.box.save_state(i)
            self.left_wheel.save_state(i)
            self.right_wheel.save_state(i)

def apply_positional_correction(x_a: torch.Tensor, x_b: torch.Tensor,
                                q_a: torch.Tensor, q_b: torch.Tensor,
                                m_a_inv: torch.Tensor, m_b_inv: torch.Tensor,
                                r_a: torch.Tensor, r_b: torch.Tensor):

    r_a_w = rotate_vectors(r_a, q_a) + x_a
    r_b_w = rotate_vectors(r_b, q_b) + x_b

    diff = r_b_w - r_a_w

    # Update positions (x_a, x_b)
    x_a_new = x_a + diff * m_a_inv / (m_a_inv + m_b_inv)
    x_b_new = x_b - diff * m_b_inv / (m_a_inv + m_b_inv)

    return x_a_new, x_b_new


def apply_rotational_correction(u_a: torch.Tensor, u_b: torch.Tensor,
                                q_a: torch.Tensor, q_b: torch.Tensor,
                                I_a_inv: torch.Tensor, I_b_inv: torch.Tensor):
    u_a_w = rotate_vectors(u_a, q_a)
    u_b_w = rotate_vectors(u_b, q_b)

    rot_vector = torch.linalg.cross(u_a_w, u_b_w)
    theta = torch.linalg.norm(rot_vector)

    if theta < 1e-6:
        return q_a, q_b

    n = rot_vector / theta

    n_a = rotate_vectors_inverse(n, q_a)
    n_b = rotate_vectors_inverse(n, q_b)

    w1 = n_a @ I_a_inv @ n_a
    w2 = n_a @ I_b_inv @ n_b
    w = w1 + w2

    theta_a = theta * w1 / w
    theta_b = - theta * w2 / w

    q_a_correction = rotvec_to_quat(theta_a * n_a)
    q_b_correction = rotvec_to_quat(theta_b * n_b)

    q_a_new = quat_mul(q_a, q_a_correction)
    q_b_new = quat_mul(q_b, q_b_correction)

    return q_a_new, q_b_new

def show_configuration():
    x_lims = (-3, 3)
    y_lims = (-3, 3)
    z_lims = (-1, 5)

    box = Cuboid(x=torch.tensor([0.0, 0.0, 2.5]),
                    q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    v=torch.tensor([0.0, 0.0, 0.0]),
                    w=torch.tensor([0.0, 0.0, 0.0]),
                    hx=1.0,
                    hy=1.0,
                    hz=1.0,
                    m=1.0,
                    n_collision_points=32)

    left_wheel_rot = rotvec_to_quat(torch.tensor([-np.pi / 2, 0.0, 0.0]))
    left_wheel = Cylinder(x=torch.tensor([0.0, 1.6, 2.5]),
                        q=left_wheel_rot,
                        v=torch.tensor([0.0, 0.0, 0.0]),
                        w=torch.tensor([0.0, 0.0, 0.0]),
                        radius=2.0,
                        height=0.4,
                        m=1.0,
                        n_collision_points_base=40,
                        n_collision_points_surface=80,
                        restitution=0.6,
                        static_friction=0.4,
                        dynamic_friction=1.0)

    right_wheel_rot = rotvec_to_quat(torch.tensor([np.pi / 2, 0.0, 0.0]))
    right_wheel = Cylinder(x=torch.tensor([0.0, -1.6, 2.5]),
                          q=right_wheel_rot,
                          v=torch.tensor([0.0, 0.0, 0.0]),
                          w=torch.tensor([0.0, 0.0, 0.0]),
                          radius=2.0,
                          height=0.4,
                          m=1.0,
                          n_collision_points_base=40,
                          n_collision_points_surface=80,
                          restitution=0.6,
                          static_friction=0.4,
                          dynamic_friction=1.0)

    robot = Robot(box, left_wheel, right_wheel)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    robot.plot(ax)  # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_zlim(z_lims)
    ax.set_title('Cylinder Wireframe')

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.show()

def show_hinge_correction():
    x_lims = (-3, 3)
    y_lims = (-3, 3)
    z_lims = (-1, 5)

    fig = plt.figure(figsize=(20, 10))

    # Left 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim(x_lims)
    ax1.set_ylim(y_lims)
    ax1.set_zlim(z_lims)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Before Correction')
    ax1.set_box_aspect([1, 1, 1])

    # Right 3D plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlim(x_lims)
    ax2.set_ylim(y_lims)
    ax2.set_zlim(z_lims)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('After Correction')
    ax2.set_box_aspect([1, 1, 1])

    cuboid = Cuboid(x=torch.tensor([0.0, 0.0, 2.5]),
                    q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    v=torch.tensor([0.0, 0.0, 0.0]),
                    w=torch.tensor([0.0, 0.0, 0.0]),
                    hx=1.0,
                    hy=1.0,
                    hz=1.0,
                    m=1.0,
                    n_collision_points=64)

    cylinder_rot = rotvec_to_quat(torch.tensor([np.pi / 2 + 0.1, 0.2, 0.0]))
    cylinder = Cylinder(x=torch.tensor([0.0, -2.0, 2.5]),
                        q=cylinder_rot,
                        v=torch.tensor([0.0, 0.0, 0.0]),
                        w=torch.tensor([0.0, 0.0, 0.0]),
                        radius=2.0,
                        height=1.0,
                        m=1.0,
                        n_collision_points_base=50,
                        n_collision_points_surface=100,
                        restitution=0.6,
                        static_friction=0.4,
                        dynamic_friction=1.0)

    robot = Robot(cuboid, cylinder)

    robot.plot(ax1)
    robot.correct_joint()
    robot.plot(ax2)

    plt.show()

def simulate():
    n_frames = 100
    time_len = 1.0
    dt = time_len / n_frames
    time = torch.linspace(0, time_len, n_frames)

    box = Cuboid(x=torch.tensor([0.0, 0.0, 2.5]),
                    q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    v=torch.tensor([0.0, 0.0, 0.0]),
                    w=torch.tensor([0.0, 0.0, 0.0]),
                    hx=1.0,
                    hy=1.0,
                    hz=1.0,
                    m=1.0,
                    n_collision_points=32)

    left_wheel_rot = rotvec_to_quat(torch.tensor([-np.pi / 2, 0.0, 0.0]))
    left_wheel = Cylinder(x=torch.tensor([0.0, 1.6, 2.5]),
                        q=left_wheel_rot,
                        v=torch.tensor([0.0, 0.0, 0.0]),
                        w=torch.tensor([0.0, 0.0, 0.0]),
                        radius=2.0,
                        height=0.4,
                        m=1.0,
                        n_collision_points_base=40,
                        n_collision_points_surface=80,
                        restitution=0.6,
                        static_friction=0.4,
                        dynamic_friction=1.0)

    right_wheel_rot = rotvec_to_quat(torch.tensor([np.pi / 2, 0.0, 0.0]))
    right_wheel = Cylinder(x=torch.tensor([0.0, -1.6, 2.5]),
                          q=right_wheel_rot,
                          v=torch.tensor([0.0, 0.0, 0.0]),
                          w=torch.tensor([0.0, 0.0, 0.0]),
                          radius=2.0,
                          height=0.4,
                          m=1.0,
                          n_collision_points_base=40,
                          n_collision_points_surface=80,
                          restitution=0.6,
                          static_friction=0.4,
                          dynamic_friction=1.0)

    robot = Robot(box, left_wheel, right_wheel)

    robot.simulate(n_frames, time_len)

    controller = AnimationController(bodies=[robot.box, robot.left_wheel, robot.right_wheel],
                                     time=time,
                                     x_lims=(-1-1, 5+1),
                                     y_lims=(-3-1, 3+1),
                                     z_lims=(-1-1, 5+1))
    controller.start()



if __name__ == '__main__':
    # show_configuration()
    # show_hinge_correction()
    simulate()