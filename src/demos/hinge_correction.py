from pbd_torch.old.cuboid import Cuboid
from pbd_torch.old.body import Trajectory
from pbd_torch.correction import *
from pbd_torch.animation import AnimationController

RANDOM_SEED = 0


def visualize_hinge_correction(x_a: torch.Tensor,
                               x_b: torch.Tensor,
                               q_a: torch.Tensor,
                               q_b: torch.Tensor,
                               r_a: torch.Tensor,
                               r_b: torch.Tensor,
                               u_a: torch.Tensor,
                               u_b: torch.Tensor,
                               m_a_inv: float,
                               m_b_inv: float,
                               I_a_inv: torch.Tensor,
                               I_b_inv: torch.Tensor):
    n_steps = 20
    time_len = 3.0
    time = torch.linspace(0, time_len, n_steps)

    box_a = Cuboid(x=x_a, q=q_a.clone(), n_collision_points=4)
    box_b = Cuboid(x=x_b, q=q_b.clone(), n_collision_points=4)

    box_a.trajectory = Trajectory(time)
    box_b.trajectory = Trajectory(time)

    box_a.vector_history[0] = []
    box_a.vector_history[0].append(LocalVector(r_a, box_a.x.clone(), box_a.q.clone(), color=ORANGE))
    box_a.vector_history[0].append(LocalVector(u_a, box_a.x.clone(), box_a.q.clone(), color=CYAN))

    box_b.vector_history[0] = []
    box_b.vector_history[0].append(LocalVector(r_b, box_b.x.clone(), box_b.q.clone(), color=ORANGE))
    box_b.vector_history[0].append(LocalVector(u_b, box_b.x.clone(), box_b.q.clone(), color=CYAN))

    box_a.save_state(0)
    box_b.save_state(0)

    for i in range(1, n_steps):
        x_a, x_b = box_a.x.clone(), box_b.x.clone()
        q_a, q_b = box_a.q.clone(), box_b.q.clone()

        dx_a_pos, dx_b_pos, dq_a_pos, dq_b_pos, _ = positional_delta(x_a=x_a,
                                                                     x_b=x_b,
                                                                     q_a=q_a,
                                                                     q_b=q_b,
                                                                     r_a=r_a,
                                                                     r_b=r_b,
                                                                     m_a_inv=m_a_inv,
                                                                     m_b_inv=m_b_inv,
                                                                     I_a_inv=I_a_inv,
                                                                     I_b_inv=I_b_inv)

        dq_a_rot, dq_b_rot = angular_delta(u_a=u_a, u_b=u_b,
                                           q_a=q_a, q_b=q_b,
                                           I_a_inv=I_a_inv,
                                           I_b_inv=I_b_inv)
        box_a.x = box_a.x + dx_a_pos
        box_b.x = box_b.x + dx_b_pos
        box_a.q = box_a.q + dq_a_pos + dq_a_rot
        box_b.q = box_b.q + dq_b_pos + dq_b_rot
        box_a.q = normalize_quat(box_a.q)
        box_b.q = normalize_quat(box_b.q)

        box_a.vector_history[i] = []
        box_a.vector_history[i].append(LocalVector(r_a, box_a.x.clone(), box_a.q.clone(), color=ORANGE))
        box_a.vector_history[i].append(LocalVector(u_a, box_a.x.clone(), box_a.q.clone(), color=CYAN))

        box_b.vector_history[i] = []
        box_b.vector_history[i].append(LocalVector(r_b, box_b.x.clone(), box_b.q.clone(), color=ORANGE))
        box_b.vector_history[i].append(LocalVector(u_b, box_b.x.clone(), box_b.q.clone(), color=CYAN))

        box_a.save_state(i)
        box_b.save_state(i)

    controller = AnimationController(bodies=[box_a, box_b],
                                     time=time,
                                     x_lims=(-3, 3),
                                     y_lims=(-3, 3),
                                     z_lims=(-1, 5))
    controller.start()


def hinge_correction_demo():
    x_a = torch.tensor([0.0, 0.0, 0.0])
    x_b = torch.tensor([0.0, 0.5, 2.0])
    q_a = ROT_IDENTITY
    q_b = ROT_IDENTITY
    r_a = torch.tensor([0.5, 0.5, 0.5])
    r_b = torch.tensor([0.5, -0.5, 0.0])
    u_a = torch.tensor([0.5, 0.5, 0.0])
    u_b = torch.tensor([0.5, 0.0, -0.5])
    m_a_inv = 1.0
    m_b_inv = 1.0
    I_a_inv = torch.eye(3)
    I_b_inv = torch.eye(3)

    visualize_hinge_correction(x_a, x_b, q_a, q_b, r_a, r_b, u_a, u_b, m_a_inv, m_b_inv, I_a_inv, I_b_inv)


def show_test_cases(seed: int = 0):
    torch.manual_seed(seed)

    n_cases = 10
    for _ in range(n_cases):
        x_a = torch.rand(3)
        x_b = torch.rand(3)
        q_a = torch.rand(4)
        q_b = torch.rand(4)
        r_a = torch.rand(3)
        r_b = torch.rand(3)
        u_a = torch.rand(3)
        u_b = torch.rand(3)
        m_a_inv = torch.rand(1).item()
        m_b_inv = torch.rand(1).item()
        I_a_inv = torch.rand(3, 3)
        I_b_inv = torch.rand(3, 3)

        q_a = normalize_quat(q_a)
        q_b = normalize_quat(q_b)

        visualize_hinge_correction(x_a, x_b, q_a, q_b, r_a, r_b, u_a, u_b, m_a_inv, m_b_inv, I_a_inv, I_b_inv)


if __name__ == "__main__":
    hinge_correction_demo()
    show_test_cases(RANDOM_SEED)
