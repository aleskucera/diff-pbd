import pytest
import torch

from pbd_torch.transform import numerical_angular_velocity, quat_mul

class TestNumericalAngularVelocity:
    @pytest.mark.parametrize(
        "q_prev, axis, angle, dt",
        [
            # Rotation starting from identity
            (torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),  # Initial rotation
             torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),  # Axis
             0.1,  # Angle
             1.0,  # Time step
            ),
            # Rotation about Y-axis from a non-identity initial rotation
            (torch.tensor([0.7071, 0.7071, 0.0, 0.0], dtype=torch.float32),  # Initial 90-degree rotation about X
             torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),  # Axis
             0.2,  # Angle
             1.0,  # Time step
             ),
            # Rotation about Z-axis
            (torch.tensor([0.7071, 0.0, 0.7071, 0.0], dtype=torch.float32),  # Initial 90-degree rotation about Y
             torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),  # Axis
             0.3,  # Angle
             1.5,  # Time step
             )
        ],
    )
    def test_numerical_angular_velocity(self, q_prev, axis, angle, dt):
        """
        Test numerical angular velocity for various rotations given an initial rotation, axis, and angle.
        """

        from scipy.spatial.transform import Rotation as R

        # Normalize the axis
        axis = axis / torch.norm(axis)

        # Compute the expected angular velocity
        expected_w = angle * axis / dt

        # Create the current rotation quaternion by applying the rotation to the initial one
        q_relative = torch.from_numpy(R.from_rotvec(axis * angle).as_quat()).type(torch.float32)
        q_relative = torch.tensor([q_relative[3], q_relative[0], q_relative[1], q_relative[2]])  # [w, x, y, z]

        q = quat_mul(q_relative, q_prev)  # Final rotation

        # Compute the numerical angular velocity
        result = numerical_angular_velocity(q, q_prev, dt)

        # Assert the computed angular velocity matches the expected one
        assert torch.allclose(result, expected_w, atol=1e-2)

