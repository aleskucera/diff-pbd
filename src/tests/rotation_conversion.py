import torch
import pytest
from scipy.spatial.transform import Rotation as R

from pbd_torch.constants import *
from pbd_torch.transform import (
    rotvec_to_quat,
    quat_to_rotvec,
)


class TestRotationConversions:
    def test_rotation_vector_to_quaternion(self):
        """Test conversion from rotation vector to quaternion."""
        rotation_vector = torch.tensor([0.0, 0.0, torch.pi], dtype=torch.float32)  # 180° about Z-axis
        expected_quaternion = ROT_180_Z

        result = rotvec_to_quat(rotation_vector)
        assert torch.allclose(result, expected_quaternion, atol=1e-6)

    def test_quaternion_to_rotation_vector(self):
        """Test conversion from quaternion to rotation vector."""
        quaternion = ROT_180_Z
        expected_rotation_vector = torch.tensor([0.0, 0.0, torch.pi], dtype=torch.float32)

        result = quat_to_rotvec(quaternion)
        assert torch.allclose(result, expected_rotation_vector, atol=1e-6)

    def test_round_trip_conversion(self):
        """Test round-trip conversion for consistency."""
        rotation_vector = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)  # Arbitrary rotation
        quaternion = rotvec_to_quat(rotation_vector)
        result_vector = quat_to_rotvec(quaternion)

        assert torch.allclose(result_vector, rotation_vector, atol=1e-6)

    def test_compare_with_scipy(self):
        """Compare rotation vector to quaternion conversion with scipy."""
        rotation_vector = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)  # Arbitrary rotation
        scipy_quaternion = R.from_rotvec(rotation_vector.numpy()).as_quat()  # [x, y, z, w]
        expected_quaternion = torch.tensor([scipy_quaternion[3], *scipy_quaternion[:3]], dtype=torch.float32)  # [w, x, y, z]

        result = rotvec_to_quat(rotation_vector)
        assert torch.allclose(result, expected_quaternion, atol=1e-6)

    def test_batch_rotation_vector_to_quaternion(self):
        """Test batch conversion from rotation vectors to quaternions."""
        rotation_vectors = torch.tensor([
            [0.0, 0.0, 0.0],  # Identity rotation
            [0.0, 0.0, torch.pi],  # 180° about Z-axis
            [torch.pi, 0.0, 0.0]  # 180° about X-axis
        ], dtype=torch.float32)
        expected_quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # Identity quaternion
            [0.0, 0.0, 0.0, 1.0],  # 180° about Z-axis
            [0.0, 1.0, 0.0, 0.0]  # 180° about X-axis
        ], dtype=torch.float32)

        result = rotvec_to_quat(rotation_vectors)
        assert torch.allclose(result, expected_quaternions, atol=1e-6)

    def test_batch_quaternion_to_rotation_vector(self):
        """Test batch conversion from quaternions to rotation vectors."""
        quaternions = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # Identity quaternion
            [0.0, 0.0, 0.0, 1.0],  # 180° about Z-axis
            [0.0, 1.0, 0.0, 0.0]  # 180° about X-axis
        ], dtype=torch.float32)
        expected_rotation_vectors = torch.tensor([
            [0.0, 0.0, 0.0],  # Identity rotation
            [0.0, 0.0, torch.pi],  # 180° about Z-axis
            [torch.pi, 0.0, 0.0]  # 180° about X-axis
        ], dtype=torch.float32)

        result = quat_to_rotvec(quaternions)
        assert torch.allclose(result, expected_rotation_vectors, atol=1e-6)

    def test_edge_case_small_angles(self):
        """Test handling of very small rotation vectors."""
        rotation_vector = torch.tensor([1e-9, 1e-9, 1e-9], dtype=torch.float32)
        quaternion = rotvec_to_quat(rotation_vector)
        result_vector = quat_to_rotvec(quaternion)

        assert torch.allclose(result_vector, rotation_vector, atol=1e-6)

