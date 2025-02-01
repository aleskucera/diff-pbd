import pytest
import torch
from pbd_torch.constants import *
from pbd_torch.transform import *
from scipy.spatial.transform import Rotation as R


class TestQuaternionMultiplication:

    @pytest.mark.parametrize(
        "q1, q2, expected",
        [
            (ROT_IDENTITY, torch.tensor(
                [0.0, 1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0, 0.0])),
            (torch.tensor([0.0, 1.0, 0.0, 0.0]), ROT_IDENTITY,
             torch.tensor([0.0, 1.0, 0.0, 0.0])),
        ],
    )
    def test_identity_multiplication(self, q1, q2, expected):
        result = quat_mul(q1, q2)
        assert torch.allclose(result,
                              expected), f"Expected {expected}, got {result}"

    def test_consecutive_rotations(self):
        q_180z = quat_mul(ROT_90_Z, ROT_90_Z)
        assert torch.allclose(
            q_180z, ROT_180_Z,
            atol=1e-6), "90-degree rotation composition failed"

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
            q1_scipy = R.from_quat(
                [q1[1].item(), q1[2].item(), q1[3].item(), q1[0].item()])
            q2_scipy = R.from_quat(
                [q2[1].item(), q2[2].item(), q2[3].item(), q2[0].item()])

            # Compute expected result using scipy
            expected = torch.from_numpy(
                (q1_scipy * q2_scipy).as_quat()).type(q1.dtype)
            expected = torch.tensor(
                [expected[3], expected[0], expected[1], expected[2]])

            # Compute result using your implementation
            result = quat_mul(q1, q2)

            assert torch.allclose(result, expected, atol=1e-6), \
                f"Random quaternion multiplication failed:\nq1: {q1}\nq2: {q2}\nExpected: {expected}\nGot: {result}"


class TestVectorsRotation:

    @pytest.mark.parametrize(
        "vector, quaternion, expected",
        [
            (
                torch.tensor([1.0, 0.0, 0.0]),
                ROT_90_Z,
                torch.tensor([0.0, 1.0, 0.0]),
            ),
        ],
    )
    def test_vector_rotation(self, vector, quaternion, expected):
        result = rotate_vectors(vector, quaternion)
        assert torch.allclose(
            result, expected,
            atol=1e-6), f"Rotation failed: expected {expected}, got {result}"

    @pytest.mark.parametrize(
        "vector, angle, axis",
        [
            (torch.tensor([1.0, 0.0, 0.0]), np.pi / 3, np.array(
                [1.0, 0.0, 0.0])),
            (torch.tensor([0.0, 1.0, 0.0]), np.pi / 3, np.array(
                [0.0, 1.0, 0.0])),
            (torch.tensor([0.0, 0.0, 1.0]), np.pi / 3, np.array(
                [0.0, 0.0, 1.0])),
        ],
    )
    def test_compare_with_scipy(self, vector: torch.Tensor, angle: float,
                                axis: np.ndarray):
        scipy_quat = R.from_rotvec(angle * axis).as_quat()
        our_quat = torch.tensor(
            [scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]])
        our_result = rotate_vectors(vector, our_quat)
        scipy_result = R.from_quat(scipy_quat).apply(vector.numpy())
        scipy_result = torch.from_numpy(scipy_result).type(our_result.dtype)
        assert torch.allclose(our_result, scipy_result,
                              atol=1e-6), "Scipy comparison failed"

    def test_batch_rotation(self):
        vectors = torch.randn(10, 3)
        angles = torch.randn(10) * np.pi
        quats = torch.stack([
            torch.cos(angles / 2),
            torch.zeros_like(angles),
            torch.zeros_like(angles),
            torch.sin(angles / 2)
        ],
                            dim=1)
        rotated = rotate_vectors(vectors, quats)
        assert rotated.shape == vectors.shape, "Batch rotation shape mismatch"

        for i in range(10):
            single_rotated = rotate_vectors(vectors[i], quats[i])
            assert torch.allclose(single_rotated,
                                  rotated[i]), f"Mismatch in batch item {i}"


class TestRotateVectorsInverse:

    def test_identity_rotation(self):
        """Test that rotation with identity quaternion does not change the vector."""
        v = torch.tensor([1.0, 0.0, 0.0])
        q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = rotate_vectors_inverse(v, q_identity)
        assert torch.allclose(result, v, atol=1e-6)

    def test_rotation_inverse_property(self):
        """Test that rotating a vector by a quaternion's inverse undoes the rotation."""
        v = torch.tensor([1.0, 0.0, 0.0])

        # 90-degree rotation around Z-axis
        q = ROT_90_Z

        rotated = rotate_vectors(v, q)
        inverted = rotate_vectors_inverse(rotated, q)
        assert torch.allclose(inverted, v, atol=1e-6)

    def test_compare_with_scipy(self):
        """Compare rotation inverse with scipy's implementation."""
        from scipy.spatial.transform import Rotation as R

        # Random vector and quaternion
        v = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        angle = np.pi / 3
        axis = np.array([1., 0., 0.])
        axis = axis / np.linalg.norm(axis)

        # Quaternion [w, x, y, z]
        scipy_quat = R.from_rotvec(angle * axis).as_quat()
        q = torch.tensor(
            [scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]],
            dtype=torch.float32)

        # Rotate vector with inverse quaternion
        result = rotate_vectors_inverse(v, q)

        # Scipy rotation
        scipy_rot = R.from_quat(scipy_quat).inv()
        scipy_result = scipy_rot.apply(v.numpy())
        scipy_result = torch.from_numpy(scipy_result).type(result.dtype)

        assert torch.allclose(result, scipy_result, atol=1e-6)

    def test_batch_inverse_rotation(self):
        """Test batch rotation inverse with multiple vectors and quaternions."""
        vectors = torch.randn(10, 3)
        angles = torch.randn(10) * np.pi
        quats = torch.stack([
            torch.cos(angles / 2),
            torch.zeros_like(angles),
            torch.zeros_like(angles),
            torch.sin(angles / 2)
        ],
                            dim=1)

        # Rotate vectors
        rotated = rotate_vectors(vectors, quats)

        # Rotate back using inverse
        restored = rotate_vectors_inverse(rotated, quats)

        assert torch.allclose(restored, vectors, atol=1e-6)

    def test_zero_rotation(self):
        """Test with a zero rotation quaternion."""
        v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        q = torch.tensor([1.0, 0.0, 0.0, 0.0],
                         dtype=torch.float32)  # Identity quaternion
        result = rotate_vectors_inverse(v, q)
        assert torch.allclose(result, v, atol=1e-6)

    def test_small_angles(self):
        """Test rotation inverse for very small rotation angles."""
        v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        angle = 1e-6  # Small angle
        q = torch.tensor([np.cos(angle / 2), 0.0, 0.0,
                          np.sin(angle / 2)],
                         dtype=torch.float32)

        rotated = rotate_vectors(v, q)
        inverted = rotate_vectors_inverse(rotated, q)

        assert torch.allclose(inverted, v, atol=1e-6)

    def test_large_rotations(self):
        """Test rotation inverse for angles larger than 2π."""
        v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        angle = 4 * np.pi  # Large angle
        q = torch.tensor([np.cos(angle / 2), 0.0, 0.0,
                          np.sin(angle / 2)],
                         dtype=torch.float32)

        rotated = rotate_vectors(v, q)
        inverted = rotate_vectors_inverse(rotated, q)

        # Should result in the same vector since rotation is effectively 0 mod 2π
        assert torch.allclose(inverted, v, atol=1e-6)


class TestQuaternionConjugate:

    def test_single_conjugate(self):
        """Test quaternion conjugate for a single quaternion."""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.tensor([1.0, -2.0, -3.0, -4.0])
        result = quat_inv(q)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_compare_with_scipy(self):
        """Compare quaternion conjugate with scipy's implementation."""
        from scipy.spatial.transform import Rotation as R

        # Random quaternion
        q = torch.tensor([0.707, 0.0, 0.707, 0.0],
                         dtype=torch.float32)  # [w, x, y, z]
        scipy_quat = [q[1].item(), q[2].item(), q[3].item(),
                      q[0].item()]  # Scipy uses [x, y, z, w]

        # Scipy conjugate
        scipy_conjugate = R.from_quat(scipy_quat).inv().as_quat()
        expected = torch.tensor([
            scipy_conjugate[3], scipy_conjugate[0], scipy_conjugate[1],
            scipy_conjugate[2]
        ],
                                dtype=torch.float32)

        # Function result
        result = quat_inv(q)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_batch_conjugate(self):
        """Test quaternion conjugate for a batch of quaternions."""
        q_batch = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # Identity quaternion
            [0.0, 1.0, 0.0, 0.0],  # 180-degree rotation around X-axis
            [0.0, 0.0, 1.0, 0.0],  # 180-degree rotation around Y-axis
            [0.0, 0.0, 0.0, 1.0]  # 180-degree rotation around Z-axis
        ])
        expected = torch.tensor([[1.0, -0.0, -0.0, -0.0],
                                 [0.0, -1.0, -0.0, -0.0],
                                 [0.0, -0.0, -1.0, -0.0],
                                 [0.0, -0.0, -0.0, -1.0]])
        result = quat_inv(q_batch)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_large_batch_with_random_values(self):
        """Test quaternion conjugate for a large batch of random quaternions."""
        q_batch = torch.randn(1000, 4)
        result = quat_inv(q_batch)
        expected = torch.cat([q_batch[..., :1], -q_batch[..., 1:]], dim=-1)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_zero_quaternion(self):
        """Test quaternion conjugate with a zero quaternion."""
        q = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        expected = torch.tensor([0.0, -0.0, -0.0, -0.0], dtype=torch.float32)
        result = quat_inv(q)
        assert torch.allclose(result, expected, atol=1e-6)


class TestRotVecToQuat:

    def test_zero_rotation(self):
        """Test conversion of zero rotation vector."""
        rotvec = torch.zeros(3)
        result = rotvec_to_quat(rotvec)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0]).type(result.dtype)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_compare_with_scipy(self):
        """Compare with scipy's implementation."""
        rotvec = torch.tensor([np.pi / 3, np.pi / 4, np.pi / 6])
        result = rotvec_to_quat(rotvec)
        scipy_quat = R.from_rotvec(rotvec.numpy()).as_quat()
        expected = torch.tensor(
            [scipy_quat[3], scipy_quat[0], scipy_quat[1],
             scipy_quat[2]]).type(result.dtype)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_batch_conversion(self):
        """Test batch conversion of rotation vectors."""
        rotvecs = torch.randn(10, 3)
        result = rotvec_to_quat(rotvecs)
        assert result.shape == (10, 4)

        for i in range(10):
            single_result = rotvec_to_quat(rotvecs[i])
            assert torch.allclose(single_result, result[i], atol=1e-6)


class TestQuatToRotVec:

    def test_identity_quaternion(self):
        """Test conversion of identity quaternion."""
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = quat_to_rotvec(quat)
        expected = torch.zeros(3).type(result.dtype)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_compare_with_scipy(self):
        """Compare with scipy's implementation."""
        quat = torch.tensor([0.7071, 0.0, 0.7071, 0.0])
        scipy_quat = [
            quat[1].item(), quat[2].item(), quat[3].item(), quat[0].item()
        ]
        expected = torch.from_numpy(R.from_quat(scipy_quat).as_rotvec())
        result = quat_to_rotvec(quat).type(expected.dtype)
        assert torch.allclose(expected, result, atol=1e-6)

    def test_batch_conversion(self):
        """Test batch conversion of quaternions."""
        quats = torch.randn(10, 4)
        quats = quats / torch.norm(quats, dim=1, keepdim=True)
        result = quat_to_rotvec(quats)
        assert result.shape == (10, 3)


class TestQuatToRotMat:

    def test_identity_quaternion(self):
        """Test conversion of identity quaternion."""
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = quat_to_rotmat(quat)
        expected = torch.eye(3).type(result.dtype)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_compare_with_scipy(self):
        """Compare with scipy's implementation."""
        quat = torch.tensor([0.7071, 0.0, 0.7071, 0.0])
        scipy_quat = [
            quat[1].item(), quat[2].item(), quat[3].item(), quat[0].item()
        ]
        expected = torch.from_numpy(R.from_quat(scipy_quat).as_matrix())
        result = quat_to_rotmat(quat).type(expected.dtype)
        assert torch.allclose(expected, result, atol=1e-6)


class TestRotMatToQuat:

    def test_identity_matrix(self):
        """Test conversion of identity matrix."""
        rotmat = torch.eye(3)
        result = rotmat_to_quat(rotmat)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0]).type(result.dtype)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_compare_with_scipy(self):
        """Compare with scipy's implementation."""
        angle = np.pi / 3
        rotmat = torch.tensor(R.from_rotvec([0, 0, angle]).as_matrix())
        result = rotmat_to_quat(rotmat)
        scipy_quat = R.from_matrix(rotmat.numpy()).as_quat()
        expected = torch.tensor(
            [scipy_quat[3], scipy_quat[0], scipy_quat[1],
             scipy_quat[2]]).type(result.dtype)
        assert torch.allclose(result, expected, atol=1e-6)


class TestRelativeRotation:

    def test_identity_case(self):
        """Test relative rotation with identity quaternion."""
        q = torch.tensor([0.7071, 0.0, 0.7071, 0.0])
        q_init = q.clone()
        result = relative_rotation(q, q_init)
        expected = ROT_IDENTITY.type(result.dtype)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_known_rotation(self):
        """Test relative rotation with known rotation."""
        q_a = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity
        q_b = ROT_90_Z  # 90-degree rotation around Z
        result = relative_rotation(q_a, q_b).type(q_b.dtype)
        assert torch.allclose(result, q_b, atol=1e-6)

    def test_compare_with_matrices(self):
        """Compare relative rotation with rotation matrices."""
        mat_a = R.from_euler('z', 45, degrees=True).as_matrix()
        mat_b = R.from_euler('z', 90, degrees=True).as_matrix()

        q_a = ROT_45_Z
        q_b = ROT_90_Z

        result = relative_rotation(q_a, q_b)
        scipy_quat = torch.tensor(R.from_matrix(mat_b @ mat_a.T).as_quat())
        expected = torch.tensor(
            [scipy_quat[3], scipy_quat[0], scipy_quat[1],
             scipy_quat[2]]).type(result.dtype)

        assert torch.allclose(result, expected, atol=1e-6)


class TestTransform:

    def test_identity_transform(self):
        """Test transform with identity rotation and zero translation."""
        points = torch.tensor([1.0, 0.0, 0.0])
        x = torch.zeros(3)
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        body_q = torch.cat([x, q])
        result = transform(points, body_q).type(points.dtype)
        assert torch.allclose(result, points, atol=1e-6)

    def test_translation_only(self):
        """Test transform with only translation."""
        points = torch.tensor([1.0, 0.0, 0.0])
        x = torch.tensor([0.0, 1.0, 0.0])
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        body_q = torch.cat([x, q])
        expected = torch.tensor([1.0, 1.0, 0.0])
        result = transform(points, body_q).type(points.dtype)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_rotation_and_translation(self):
        """Test transform with both rotation and translation."""
        points = torch.tensor([1.0, 0.0, 0.0])
        x = torch.tensor([0.0, 1.0, 0.0])
        q = ROT_90_Z
        body_q = torch.cat([x, q])
        expected = torch.tensor([0.0, 2.0, 0.0])
        result = transform(points, body_q).type(points.dtype)
        assert torch.allclose(result, expected, atol=1e-6)


class TestCoordinateTransformations:

    @pytest.mark.parametrize(
        "body_v, body_pos, body_rot, world_v",
        [
            (torch.tensor([1.0, 1.0, 0.0]), torch.tensor(
                [0.0, 0.0, 0.0]), ROT_90_Z, torch.tensor([-1.0, 1.0, 0.0])),
            (torch.tensor([0.0, 0.0, 0.0]), torch.tensor(
                [1.0, 1.0, 0.0]), ROT_90_Z, torch.tensor([1.0, 1.0, 0.0])),
            (torch.tensor([-2.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 0.0]),
             ROT_IDENTITY, torch.tensor([-1.0, 1.0, 0.0])),
        ],
    )
    def test_body_to_world(self, body_v, body_pos, body_rot, world_v):
        result = body_to_world(body_v, torch.cat([body_pos, body_rot]))
        expected = world_v.type(result.dtype)
        assert torch.allclose(result, expected, atol=1e-6)

    @pytest.mark.parametrize(
        "world_v, body_pos, body_rot, body_v",
        [
            (torch.tensor([1.0, 1.0, 0.0]), torch.tensor(
                [0.0, 0.0, 0.0]), ROT_90_Z, torch.tensor([1.0, -1.0, 0.0])),
            (torch.tensor([1.0, 1.0, 0.0]), torch.tensor(
                [1.0, 1.0, 0.0]), ROT_90_Z, torch.tensor([0.0, 0.0, 0.0])),
            (torch.tensor([-1.0, 1.0, 0.0]), torch.tensor([1.0, 1.0, 0.0]),
             ROT_IDENTITY, torch.tensor([-2.0, 0.0, 0.0])),
        ],
    )
    def test_world_to_body(self, world_v, body_pos, body_rot, body_v):
        result = world_to_body(world_v, torch.cat([body_pos, body_rot]))
        expected = body_v.type(result.dtype)
        assert torch.allclose(result, expected, atol=1e-6)
