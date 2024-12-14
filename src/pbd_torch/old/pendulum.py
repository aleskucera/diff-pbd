from tqdm import tqdm

from pbd_torch.old.body import Trajectory
from pbd_torch.correction import *
from pbd_torch.old.cuboid import Cuboid
from pbd_torch.old.sphere import Sphere
from pbd_torch.transform import *
from pbd_torch.utils import *
from pbd_torch.animation import *


class Robot:
    def __init__(self, base: Cuboid, mid: Sphere, end: Sphere):
        self.base = base
        self.mid = mid
        self.end = end

        self.base_r = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.mid_top_r = torch.tensor([0.0, 0.0, (self.base.hz + self.mid.radius + 1.0)], dtype=torch.float32)

        self.base_u = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        self.mid_top_u = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

        self.mid_bot_r = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.end_r = torch.tensor([0.0, 0.0, (self.mid.radius + self.end.radius + 1.0)], dtype=torch.float32)

        self.mid_bot_u = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        self.end_u = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    def plot(self, ax):
        assert torch.norm(self.base.q) - 1.0 < 1e-6
        assert torch.norm(self.mid.q) - 1.0 < 1e-6
        assert torch.norm(self.end.q) - 1.0 < 1e-6

        plot_axis(ax, self.base.x, self.base.q, scale=1.0)
        self.base.plot_geometry(ax, self.base.x, self.base.q)

        plot_axis(ax, self.mid.x, self.mid.q, scale=1.0)
        self.mid.plot_geometry(ax, self.mid.x, self.mid.q)

        plot_axis(ax, self.end.x, self.end.q, scale=1.0)
        self.end.plot_geometry(ax, self.end.x, self.end.q)

        top_r_a = LocalVector(r=self.base_r.clone(), x=self.base.x.clone(), q=self.base.q.clone(), color=ORANGE)
        top_r_b = LocalVector(r=self.mid_top_r.clone(), x=self.mid.x.clone(), q=self.mid.q.clone(), color=ORANGE)

        plot_vectors(ax, [top_r_a, top_r_b])

        bot_r_a = LocalVector(r=self.mid_bot_r.clone(), x=self.mid.x.clone(), q=self.mid.q.clone(), color=ORANGE)
        bot_r_b = LocalVector(r=self.end_r.clone(), x=self.end.x.clone(), q=self.end.q.clone(), color=ORANGE)

        plot_vectors(ax, [bot_r_a, bot_r_b])

    def correct_joints(self):
        self.correct_top_joint()
        self.correct_bot_joint()

    def correct_top_joint(self):
        r_a, r_b = self.base_r, self.mid_top_r
        m_a_inv, m_b_inv = self.base.m_inv, self.mid.m_inv
        I_a_inv, I_b_inv = self.base.I_inv, self.mid.I_inv

        u_a, u_b = self.base_u, self.mid_top_u
        x_a, x_b = self.base.x, self.mid.x
        q_a, q_b = self.base.q, self.mid.q

        dq_a_ang, dq_b_ang = angular_delta(u_a, u_b, q_a, q_b, I_a_inv, I_b_inv)
        dx_a, dx_b, dq_a_pos, dq_b_pos, _ = positional_delta(x_a, x_b, q_a, q_b, r_a, r_b, m_a_inv, m_b_inv, I_a_inv,
                                                             I_b_inv)

        self.base.dx_hinge += dx_a
        self.mid.dx_hinge += dx_b

        self.base.dq_hinge += dq_a_ang + dq_a_pos
        self.mid.dq_hinge += dq_b_ang + dq_b_pos

    def correct_bot_joint(self):
        r_a, r_b = self.mid_bot_r, self.end_r
        m_a_inv, m_b_inv = self.base.m_inv, self.end.m_inv
        I_a_inv, I_b_inv = self.base.I_inv, self.end.I_inv

        u_a, u_b = self.mid_bot_u, self.end_u
        x_a, x_b = self.mid.x, self.end.x
        q_a, q_b = self.mid.q, self.end.q

        dq_a_ang, dq_b_ang = angular_delta(u_a, u_b, q_a, q_b, I_a_inv, I_b_inv)
        dx_a, dx_b, dq_a_pos, dq_b_pos, _ = positional_delta(x_a, x_b, q_a, q_b, r_a, r_b, m_a_inv, m_b_inv, I_a_inv,
                                                             I_b_inv)

        self.mid.dx_hinge += dx_a
        self.end.dx_hinge += dx_b

        self.mid.dq_hinge += dq_a_ang + dq_a_pos
        self.end.dq_hinge += dq_b_ang + dq_b_pos

    def simulate(self, n_steps: int, time: float):
        sim_time = torch.linspace(0, time, n_steps)

        self.base.dt = time / n_steps
        self.mid.dt = time / n_steps
        self.end.dt = time / n_steps

        self.base.trajectory = Trajectory(sim_time)
        self.mid.trajectory = Trajectory(sim_time)
        self.end.trajectory = Trajectory(sim_time)

        for i in tqdm(range(n_steps)):
            self.base.step = i
            self.mid.step = i
            self.end.step = i

            self.base.integrate()
            self.mid.integrate(lin_force=torch.tensor([15 * torch.sin(2 * np.pi * sim_time[i]), 0.0, 0.0]))
            self.end.integrate()

            self.base.detect_collisions()
            self.mid.detect_collisions()
            self.end.detect_collisions()

            for _ in range(3):
                # self.box.detect_collisions()
                # self.left_wheel.detect_collisions()
                # self.right_wheel.detect_collisions()

                self.base.collision_delta()
                self.mid.collision_delta()
                self.end.collision_delta()
                self.correct_joints()

                self.base.add_deltas()
                self.mid.add_deltas()
                self.end.add_deltas()

            self.base.update_velocity()
            self.mid.update_velocity()
            self.end.update_velocity()

            self.base.solve_velocity()
            self.mid.solve_velocity()
            self.end.solve_velocity()

            self.base.save_state(i)
            self.mid.save_state(i)
            self.end.save_state(i)


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
    x_lims = (-5, 5)
    y_lims = (-5, 5)
    z_lims = (-1, 9)

    box = Cuboid(x=torch.tensor([0.0, 0.0, 7.5]),
                 q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                 v=torch.tensor([0.0, 0.0, 0.0]),
                 w=torch.tensor([0.0, 0.0, 0.0]),
                 hx=0.5,
                 hy=0.5,
                 hz=0.5,
                 m=1000000.0,
                 n_collision_points=32)

    # left_wheel_rot = rotvec_to_quat(torch.tensor([-np.pi / 2, 0.0, 0.0]))
    left_wheel = Sphere(x=torch.tensor([0.0, 0.0, 5.0]),
                        q=ROT_IDENTITY,
                        v=torch.tensor([0.0, 0.0, 0.0]),
                        w=torch.tensor([0.0, 0.0, 0.0]),
                        radius=1.0,
                        m=1.0,
                        n_collision_points=64,
                        restitution=0.6,
                        static_friction=0.4,
                        dynamic_friction=1.0)

    # right_wheel_rot = rotvec_to_quat(torch.tensor([np.pi / 2, 0.0, 0.0]))
    right_wheel = Sphere(x=torch.tensor([0.0, 0.0, 2.0]),
                         q=ROT_IDENTITY,
                         v=torch.tensor([0.0, 0.0, 0.0]),
                         w=torch.tensor([0.0, 0.0, 0.0]),
                         radius=1.0,
                         m=1.0,
                         n_collision_points=64,
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

def simulate():
    n_frames = 200
    time_len = 2.0
    dt = time_len / n_frames
    time = torch.linspace(0, time_len, n_frames)

    box = Cuboid(x=torch.tensor([0.0, 0.0, 7.5]),
                 q=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                 v=torch.tensor([0.0, 0.0, 0.0]),
                 w=torch.tensor([0.0, 0.0, 0.0]),
                 hx=0.5,
                 hy=0.5,
                 hz=0.5,
                 m=1e6,
                 n_collision_points=32)

    # left_wheel_rot = rotvec_to_quat(torch.tensor([-np.pi / 2, 0.0, 0.0]))
    left_wheel = Sphere(x=torch.tensor([0.0, 0.0, 5.0]),
                        q=ROT_IDENTITY,
                        v=torch.tensor([0.0, 0.0, 0.0]),
                        w=torch.tensor([0.0, 0.0, 0.0]),
                        radius=1.0,
                        m=1.0,
                        n_collision_points=64,
                        restitution=0.6,
                        static_friction=0.4,
                        dynamic_friction=1.0)

    # right_wheel_rot = rotvec_to_quat(torch.tensor([np.pi / 2, 0.0, 0.0]))
    right_wheel = Sphere(x=torch.tensor([0.0, 0.0, 2.0]),
                         q=ROT_IDENTITY,
                         v=torch.tensor([0.0, 0.0, 0.0]),
                         w=torch.tensor([0.0, 0.0, 0.0]),
                         radius=1.0,
                         m=1.0,
                         n_collision_points=64,
                         restitution=0.6,
                         static_friction=0.4,
                         dynamic_friction=1.0)

    robot = Robot(box, left_wheel, right_wheel)

    robot.simulate(n_frames, time_len)

    controller = AnimationController(bodies=[robot.base, robot.mid, robot.end],
                                     time=time,
                                     x_lims=(-5, 5),
                                     y_lims=(-5, 5),
                                     z_lims=(-1, 9))
    controller.start()


if __name__ == '__main__':
    # show_configuration()
    simulate()
