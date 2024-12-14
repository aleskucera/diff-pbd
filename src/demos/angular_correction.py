from pbd_torch.old.cuboid import Cuboid
from pbd_torch.old.body import Trajectory
from pbd_torch.correction import *
from pbd_torch.animation import AnimationController

RANDOM_SEED = 0


def visualize_angular_correction(u_a: torch.Tensor,
                                 u_b: torch.Tensor,
                                 q_a: torch.Tensor,
                                 q_b: torch.Tensor,
                                 I_a_inv: torch.Tensor,
                                 I_b_inv: torch.Tensor):
    n_steps = 20
    time_len = 3.0
    time = torch.linspace(0, time_len, n_steps)

    box_a = Cuboid(x=torch.tensor([0.0, 0.0, 0.0]), q=q_a.clone(), n_collision_points=4)
    box_b = Cuboid(x=torch.tensor([0.0, 0.0, 2.0]), q=q_b.clone(), n_collision_points=4)

    box_a.trajectory = Trajectory(time)
    box_b.trajectory = Trajectory(time)

    box_a.vector_history[0] = []
    box_a.vector_history[0].append(LocalVector(u_a, box_a.x.clone(), box_a.q.clone()))

    box_b.vector_history[0] = []
    box_b.vector_history[0].append(LocalVector(u_b, box_b.x.clone(), box_b.q.clone()))

    box_a.save_state(0)
    box_b.save_state(0)

    for i in range(1, n_steps):
        q_a, q_b = box_a.q.clone(), box_b.q.clone()
        dq_a, dq_b = angular_delta(u_a=u_a, u_b=u_b,
                                   q_a=q_a, q_b=q_b,
                                   I_a_inv=I_a_inv,
                                   I_b_inv=I_b_inv)

        box_a.q += dq_a
        box_b.q += dq_b
        box_a.q = normalize_quat(box_a.q)
        box_b.q = normalize_quat(box_b.q)

        box_a.vector_history[i] = []
        box_a.vector_history[i].append(LocalVector(u_a, box_a.x.clone(), box_a.q.clone()))

        box_b.vector_history[i] = []
        box_b.vector_history[i].append(LocalVector(u_b, box_b.x.clone(), box_b.q.clone()))

        box_a.save_state(i)
        box_b.save_state(i)

    controller = AnimationController(bodies=[box_a, box_b],
                                     time=time,
                                     x_lims=(-3, 3),
                                     y_lims=(-3, 3),
                                     z_lims=(-1, 5))
    controller.start()


def angular_correction_demo():
    q_a = ROT_45_XY
    q_b = ROT_NEG_90_Z
    u_a = torch.tensor([0.5, 0.5, 0.0])
    u_b = torch.tensor([0.5, 0.0, -0.5])
    I_a_inv = torch.eye(3)
    I_b_inv = torch.eye(3)

    visualize_angular_correction(u_a, u_b, q_a, q_b, I_a_inv, I_b_inv)


def show_test_cases(seed: int = 0):
    torch.manual_seed(seed)

    n_cases = 10
    for _ in range(n_cases):
        u_a = torch.rand(3)
        u_b = torch.rand(3)
        q_a = torch.rand(4)
        q_b = torch.rand(4)
        q_a = normalize_quat(q_a)
        q_b = normalize_quat(q_b)
        I_a_inv = torch.eye(3)
        I_b_inv = torch.eye(3)

        visualize_angular_correction(u_a, u_b, q_a, q_b, I_a_inv, I_b_inv)


if __name__ == "__main__":
    angular_correction_demo()
    show_test_cases(RANDOM_SEED)
