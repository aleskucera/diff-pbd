import matplotlib
import torch
from pbd_torch.animation import AnimationController
from pbd_torch.animation import BodyFrame
from pbd_torch.constants import ROT_90_X
from pbd_torch.constants import ROT_IDENTITY
from pbd_torch.constants import ROT_NEG_45_Y
from pbd_torch.constants import ROT_NEG_90_X
from pbd_torch.correction import joint_delta
from pbd_torch.transform import normalize_quat
from pbd_torch.transform import transform_multiply

matplotlib.use("TkAgg")


def joint_correction_demo():
    frames_sequence = []

    body_q_p = torch.cat([torch.tensor([1.0, 0.0, 0.0]), ROT_IDENTITY])
    X_p = torch.cat([torch.tensor([1.0, 0.0, 0.0]), ROT_NEG_45_Y])
    body_q_c = torch.cat([torch.tensor([5.0, 0.0, 0.0]), ROT_90_X])
    X_c = torch.cat([torch.tensor([-1.0, 0.0, 0.0]), ROT_NEG_90_X])

    X_wj_p = transform_multiply(body_q_p, X_p)
    X_wj_c = transform_multiply(body_q_c, X_c)

    # Store initial frame
    frames_sequence.append(
        {
            "parent_body": BodyFrame(
                body_q_p, scale=1.0, color="purple", label="Parent Body"
            ),
            "parent_joint": BodyFrame(
                X_wj_p, scale=1.0, color="orange", label="Parent Joint"
            ),
            "child_body": BodyFrame(
                body_q_c, scale=0.5, color="green", label="Child Body"
            ),
            "child_joint": BodyFrame(
                X_wj_c, scale=0.5, color="blue", label="Child Joint"
            ),
        }
    )

    # Perform iterations and store frames
    for _ in range(5):
        dbody_q_p, dbody_q_c = joint_delta(
            body_q_p,
            body_q_c,
            X_p,
            X_c,
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.eye(3),
            torch.eye(3),
        )
        body_q_p += dbody_q_p
        body_q_c += dbody_q_c
        body_q_p[3:] = normalize_quat(body_q_p[3:])
        body_q_c[3:] = normalize_quat(body_q_c[3:])

        X_wj_p = transform_multiply(body_q_p, X_p)
        X_wj_c = transform_multiply(body_q_c, X_c)

        frames_sequence.append(
            {
                "parent_body": BodyFrame(
                    body_q_p, scale=1.0, color="purple", label="Parent Body"
                ),
                "parent_joint": BodyFrame(
                    X_wj_p, scale=1.0, color="orange", label="Parent Joint"
                ),
                "child_body": BodyFrame(
                    body_q_c, scale=0.5, color="green", label="Child Body"
                ),
                "child_joint": BodyFrame(
                    X_wj_c, scale=0.5, color="blue", label="Child Joint"
                ),
            }
        )

    # Create and start the animation controller
    controller = AnimationController(
        frames_sequence, x_lims=(-1, 6), y_lims=(-3, 3), z_lims=(-3, 3)
    )
    controller.start()


if __name__ == "__main__":
    joint_correction_demo()
