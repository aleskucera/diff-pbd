import numpy as np
import pytest
import torch
from pbd_torch.constants import *
from pbd_torch.transform import *
from scipy.spatial.transform import Rotation as R


# Helper function to convert between pytorch quaternion and scipy format
def to_scipy_quat(q_torch):
    """Convert pytorch quaternion [w,x,y,z] to scipy format [x,y,z,w]"""
    return np.array(
        [q_torch[1].item(), q_torch[2].item(), q_torch[3].item(), q_torch[0].item()]
    )


def from_scipy_quat(q_scipy):
    """Convert scipy quaternion [x,y,z,w] to pytorch format [w,x,y,z]"""
    return torch.tensor([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])


class TestQuaternionMultiplication:

    @pytest.mark.parametrize(
        "q1, q2, expected",
        [
            (
                ROT_IDENTITY,
                torch.tensor([0.0, 1.0, 0.0, 0.0]),
                torch.tensor([0.0, 1.0, 0.0, 0.0]),
            ),
            (
                torch.tensor([0.0, 1.0, 0.0, 0.0]),
                ROT_IDENTITY,
                torch.tensor([0.0, 1.0, 0.0, 0.0]),
            ),
        ],
    )
    def test_identity_multiplication(self, q1, q2, expected):
        result = quat_mul(q1, q2)
        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"

    def test_consecutive_rotations(self):
        q_180z = quat_mul(ROT_90_Z, ROT_90_Z)
        assert torch.allclose(
            q_180z, ROT_180_Z, atol=1e-6
        ), "90-degree rotation composition failed"

    @pytest.mark.parametrize("n_tests", [100])  # Number of random tests to run
    def test_random_quaternions(self, n_tests):
        for _ in range(n_tests):
            # Generate random quaternions
            q1 = torch.randn(4, dtype=torch.float32)
            q2 = torch.randn(4, dtype=torch.float32)

            # Normalize the quaternions
            q1 = q1 / torch.norm(q1)
            q2 = q2 / torch.norm(q2)

            # Convert to scipy format (x, y, z, w) -> (w, x, y, z)
            q1_scipy = R.from_quat(to_scipy_quat(q1))
            q2_scipy = R.from_quat(to_scipy_quat(q2))

            # Compute expected result using scipy
            expected = from_scipy_quat((q1_scipy * q2_scipy).as_quat()).type(q1.dtype)

            # Compute result using your implementation
            result = quat_mul(q1, q2)

            assert torch.allclose(
                result, expected, atol=1e-6
            ), f"Random quaternion multiplication failed:\nq1: {q1}\nq2: {q2}\nExpected: {expected}\nGot: {result}"


class TestQuaternionInversion:

    def test_identity_inversion(self):
        """Test that inverting identity quaternion returns identity."""
        assert torch.allclose(quat_inv(ROT_IDENTITY), ROT_IDENTITY)

    @pytest.mark.parametrize("n_tests", [50])
    def test_random_inversion(self, n_tests):
        """Test inversion of random quaternions."""
        for _ in range(n_tests):
            q = torch.randn(4, dtype=torch.float32)
            q = q / torch.norm(q)

            # Inverse in our implementation
            q_inv = quat_inv(q)

            # Check product with original is identity
            identity = quat_mul(q, q_inv)
            assert torch.allclose(identity, ROT_IDENTITY, atol=1e-6)

            # Check against scipy's inverse
            q_scipy = R.from_quat(to_scipy_quat(q))
            q_inv_scipy = from_scipy_quat(q_scipy.inv().as_quat()).type(q.dtype)

            # Account for sign ambiguity in quaternions (-q represents the same rotation as q)
            assert torch.allclose(q_inv, q_inv_scipy, atol=1e-6) or torch.allclose(
                q_inv, -q_inv_scipy, atol=1e-6
            )


class TestQuaternionBatch:

    def test_batch_multiplication(self):
        """Test batch quaternion multiplication matches individual multiplication."""
        batch_size = 10
        q1_batch = torch.randn(batch_size, 4, dtype=torch.float32)
        q2_batch = torch.randn(batch_size, 4, dtype=torch.float32)

        # Normalize
        q1_batch = q1_batch / torch.norm(q1_batch, dim=1, keepdim=True)
        q2_batch = q2_batch / torch.norm(q2_batch, dim=1, keepdim=True)

        # Batch multiply
        result_batch = quat_mul_batch(q1_batch, q2_batch)

        # Individual multiply
        individual_results = torch.stack(
            [quat_mul(q1, q2) for q1, q2 in zip(q1_batch, q2_batch)]
        )

        assert torch.allclose(result_batch, individual_results, atol=1e-6)

    def test_batch_inversion(self):
        """Test batch quaternion inversion matches individual inversion."""
        batch_size = 10
        q_batch = torch.randn(batch_size, 4, dtype=torch.float32)
        q_batch = q_batch / torch.norm(q_batch, dim=1, keepdim=True)

        # Batch invert
        result_batch = quat_inv_batch(q_batch)

        # Individual invert
        individual_results = torch.stack([quat_inv(q) for q in q_batch])

        assert torch.allclose(result_batch, individual_results, atol=1e-6)


class TestVectorRotation:

    def test_identity_rotation(self):
        """Test that rotation by identity quaternion doesn't change the vector."""
        vector = torch.tensor([1.0, 2.0, 3.0])
        rotated = rotate_vector(vector, ROT_IDENTITY)
        assert torch.allclose(rotated, vector, atol=1e-6)

    def test_90_degree_rotation(self):
        """Test simple 90-degree rotations."""
        # Rotate [1,0,0] by 90 degrees around Z
        vector = torch.tensor([1.0, 0.0, 0.0])
        rotated = rotate_vector(vector, ROT_90_Z)
        expected = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(rotated, expected, atol=1e-6)

        # Rotate [0,1,0] by 90 degrees around X
        vector = torch.tensor([0.0, 1.0, 0.0])
        rotated = rotate_vector(vector, ROT_90_X)
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(rotated, expected, atol=1e-6)

    @pytest.mark.parametrize("n_tests", [50])
    def test_random_rotations(self, n_tests):
        """Test random vector rotations against scipy."""
        for _ in range(n_tests):
            # Random vector
            vector = torch.randn(3, dtype=torch.float32)

            # Random quaternion
            quat = torch.randn(4, dtype=torch.float32)
            quat = quat / torch.norm(quat)

            # Rotate using our function
            rotated = rotate_vector(vector, quat)

            # Rotate using scipy
            quat_scipy = R.from_quat(to_scipy_quat(quat))
            expected = torch.from_numpy(quat_scipy.apply(vector.numpy())).float()

            assert torch.allclose(rotated, expected, atol=1e-6)

    def test_batch_rotation(self):
        """Test that batch rotation matches individual rotation."""
        batch_size = 10
        vectors = torch.randn(batch_size, 3, dtype=torch.float32)
        quats = torch.randn(batch_size, 4, dtype=torch.float32)
        quats = quats / torch.norm(quats, dim=1, keepdim=True)

        # Batch rotate
        result_batch = rotate_vectors_batch(vectors, quats)

        # Individual rotate
        individual_results = torch.stack(
            [rotate_vector(v, q) for v, q in zip(vectors, quats)]
        )

        assert torch.allclose(result_batch, individual_results, atol=1e-6)


class TestConversions:

    @pytest.mark.parametrize("n_tests", [50])
    def test_rotvec_to_quat(self, n_tests):
        """Test rotation vector to quaternion conversion."""
        for _ in range(n_tests):
            # Random rotation vector
            angle = np.random.uniform(0, np.pi)
            axis = torch.randn(3, dtype=torch.float32)
            axis = axis / torch.norm(axis)
            rotvec = angle * axis

            # Convert to quaternion
            quat = rotvec_to_quat(rotvec)

            # Compare with scipy
            rotvec_np = rotvec.numpy()
            quat_scipy = from_scipy_quat(R.from_rotvec(rotvec_np).as_quat()).type(
                rotvec.dtype
            )

            # Account for sign ambiguity
            assert torch.allclose(quat, quat_scipy, atol=1e-6) or torch.allclose(
                quat, -quat_scipy, atol=1e-6
            )

    @pytest.mark.parametrize("n_tests", [50])
    def test_quat_to_rotvec(self, n_tests):
        """Test quaternion to rotation vector conversion."""
        for _ in range(n_tests):
            # Random quaternion
            quat = torch.randn(4, dtype=torch.float32)
            quat = quat / torch.norm(quat)

            # Convert to rotation vector
            rotvec = quat_to_rotvec(quat)

            # Compare with scipy
            quat_scipy = R.from_quat(to_scipy_quat(quat))
            expected = torch.from_numpy(quat_scipy.as_rotvec()).float()

            assert torch.allclose(rotvec, expected, atol=1e-6)

    @pytest.mark.parametrize("n_tests", [50])
    def test_quat_to_rotmat(self, n_tests):
        """Test quaternion to rotation matrix conversion."""
        for _ in range(n_tests):
            # Random quaternion
            quat = torch.randn(4, dtype=torch.float32)
            quat = quat / torch.norm(quat)

            # Convert to rotation matrix
            rotmat = quat_to_rotmat(quat)

            # Compare with scipy
            quat_scipy = R.from_quat(to_scipy_quat(quat))
            expected = torch.from_numpy(quat_scipy.as_matrix()).float()

            assert torch.allclose(rotmat, expected, atol=1e-6)

    @pytest.mark.parametrize("n_tests", [50])
    def test_rotmat_to_quat(self, n_tests):
        """Test rotation matrix to quaternion conversion."""
        for _ in range(n_tests):
            # Random quaternion (to ensure valid rotation matrix)
            quat = torch.randn(4, dtype=torch.float32)
            quat = quat / torch.norm(quat)

            # Convert to rotation matrix
            quat_scipy = R.from_quat(to_scipy_quat(quat))
            rotmat = torch.from_numpy(quat_scipy.as_matrix()).float()

            # Convert back to quaternion
            result = rotmat_to_quat(rotmat)

            # Account for sign ambiguity
            assert torch.allclose(result, quat, atol=1e-6) or torch.allclose(
                result, -quat, atol=1e-6
            )

    def test_batch_conversions(self):
        """Test that batch conversions match individual conversions."""
        batch_size = 10

        # Test rotvec_to_quat_batch
        rotvecs = torch.randn(batch_size, 3, dtype=torch.float32)
        result_batch = rotvec_to_quat_batch(rotvecs)
        individual_results = torch.stack([rotvec_to_quat(rv) for rv in rotvecs])
        for i in range(batch_size):
            # Account for sign ambiguity
            assert torch.allclose(
                result_batch[i], individual_results[i], atol=1e-6
            ) or torch.allclose(result_batch[i], -individual_results[i], atol=1e-6)

        # Test quat_to_rotvec_batch
        quats = torch.randn(batch_size, 4, dtype=torch.float32)
        quats = quats / torch.norm(quats, dim=1, keepdim=True)
        result_batch = quat_to_rotvec_batch(quats)
        individual_results = torch.stack([quat_to_rotvec(q) for q in quats])
        assert torch.allclose(result_batch, individual_results, atol=1e-6)

        # Test quat_to_rotmat_batch
        result_batch = quat_to_rotmat_batch(quats)
        individual_results = torch.stack([quat_to_rotmat(q) for q in quats])
        assert torch.allclose(result_batch, individual_results, atol=1e-6)


class TestTransforms:

    def test_transform_point(self):
        """Test that transform_point correctly applies rotation and translation."""
        point = torch.tensor([1.0, 2.0, 3.0])
        transform = torch.tensor(
            [10.0, 20.0, 30.0, 1.0, 0.0, 0.0, 0.0]
        )  # Identity rotation + translation
        result = transform_point(point, transform)
        expected = point + transform[:3]  # With identity rotation, just add translation
        assert torch.allclose(result, expected, atol=1e-5)

        # Test with rotation
        transform = torch.tensor(
            [10.0, 20.0, 30.0, 0.7071, 0.7071, 0.0, 0.0]
        )  # 90 degrees around Y + translation
        result = transform_point(point, transform)
        rotated_point = rotate_vector(point, transform[3:])
        expected = rotated_point + transform[:3]
        assert torch.allclose(result, expected, atol=1e-5)

    def test_world_body_transformations(self):
        """Test body_to_world and world_to_body are inverses of each other."""
        point = torch.tensor([1.0, 2.0, 3.0])
        transform = torch.tensor(
            [10.0, 20.0, 30.0, 0.7071, 0.0, 0.7071, 0.0]
        )  # Some rotation + translation

        # Transform to world
        world_point = body_to_world(point, transform)

        # Transform back to body
        body_point = world_to_body(world_point, transform)

        assert torch.allclose(body_point, point, atol=1e-3)

    def test_transform_multiplication(self):
        """Test transform composition."""
        # Create two transforms
        transform_a = torch.tensor(
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        )  # Identity rotation + translation
        transform_b = torch.tensor(
            [4.0, 5.0, 6.0, 0.7071, 0.0, 0.7071, 0.0]
        )  # Some rotation + translation

        # Compose transforms
        combined = transform_multiply(transform_a, transform_b)

        # Create a test point
        point = torch.tensor([0.5, 1.5, 2.5])

        # Apply transforms sequentially
        intermediate = transform_point(point, transform_b)
        expected = transform_point(intermediate, transform_a)

        # Apply combined transform
        result = transform_point(point, combined)

        assert torch.allclose(result, expected, atol=1e-4)

    def test_batch_transforms(self):
        """Test that batch transforms match individual transforms."""
        batch_size = 10
        points = torch.randn(batch_size, 3, dtype=torch.float32)
        transforms = torch.zeros(batch_size, 7, dtype=torch.float32)
        transforms[:, :3] = torch.randn(batch_size, 3)  # Random translations
        transforms[:, 3] = 1.0  # Identity rotations for simplicity

        # Batch transform
        result_batch = transform_points_batch(points, transforms)

        # Individual transform
        individual_results = torch.stack(
            [transform_point(p, t) for p, t in zip(points, transforms)]
        )

        assert torch.allclose(result_batch, individual_results, atol=1e-3)

        # Test body_to_world_batch and world_to_body_batch are inverses
        world_points = body_to_world_batch(points, transforms)
        body_points = world_to_body_batch(world_points, transforms)
        assert torch.allclose(body_points, points, atol=1e-3)
