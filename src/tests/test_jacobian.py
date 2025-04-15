import torch
import pytest
import scipy.sparse
import numpy as np
from pbd_torch.model import Model
from pbd_torch.newton_engine import NonSmoothNewtonEngine


class TestNewtonEngine:
    @pytest.fixture
    def setup_engine(self):
        device = torch.device("cpu")
        body_count = 2
        max_contacts = 3
        joint_count = 1
        dt = 0.01
        torch.random.manual_seed(0)

        # Mock Model
        class MockModel:
            def __init__(self):
                self.device = device
                self.body_count = body_count
                self.max_contacts_per_body = max_contacts
                self.mass_matrix = torch.randn(body_count, 6, 6, device=device)
                self.mass_matrix = self.mass_matrix @ self.mass_matrix.transpose(1, 2)  # Positive definite
                self.g_accel = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -9.81], device=device).view(6, 1)

                self.joint_parent = torch.tensor([0], dtype=torch.long, device=device)
                self.joint_child = torch.tensor([1], dtype=torch.long, device=device)
                self.joint_X_p = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device=device).view(1, 7, 1)
                self.joint_X_c = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device=device).view(1, 7, 1)

                self.body_q = torch.zeros((body_count, 7, 1), device=device)
                for i in range(body_count):
                    self.body_q[i, 3, 0] = 1.0  # Initialize with identity quaternion

                self.restitution = torch.ones(body_count, 1, device=device) * 0.2
                self.dynamic_friction = torch.ones(body_count, 1, device=device) * 0.5

        model = MockModel()
        engine = NonSmoothNewtonEngine(model=model, iterations=1, device=device)

        # Initialize engine's internal variables that would normally be set in the simulate method
        B = body_count
        C = max_contacts
        D = joint_count

        # Set up contact data
        engine._contact_mask = torch.zeros((B, C), device=device, dtype=torch.bool)
        engine._contact_mask[0, 0] = True  # One active contact on first body
        engine._contact_mask[1, 1] = True  # One active contact on second body

        engine._penetration_depth = torch.zeros((B, C, 1), device=device)
        engine._penetration_depth[0, 0, 0] = -0.1  # Some penetration
        engine._penetration_depth[1, 1, 0] = -0.2  # Some penetration

        # Set Jacobians
        engine._J_n = torch.zeros((B, C, 6), device=device)
        engine._J_t = torch.zeros((B, 2 * C, 6), device=device)

        # For normal contact Jacobians, use some non-zero values for active contacts
        engine._J_n[0, 0, 3] = 1.0  # First body, first contact, related to x-velocity
        engine._J_n[1, 1, 5] = 1.0  # Second body, second contact, related to z-velocity

        # For tangential Jacobians
        engine._J_t[0, 0, 4] = 1.0  # First body, first tangential dir, related to y-velocity
        engine._J_t[0, C, 5] = 1.0  # First body, second tangential dir, related to z-velocity
        engine._J_t[1, 1, 3] = 1.0  # Second body, first tangential dir for contact 2, related to x-velocity
        engine._J_t[1, C + 1, 4] = 1.0  # Second body, second tangential dir for contact 2, related to y-velocity

        # For joint Jacobians
        engine._J_j_p = torch.zeros((5 * D, 6), device=device)
        engine._J_j_c = torch.zeros((5 * D, 6), device=device)

        # Set up some values for joint Jacobians
        engine._J_j_p[0, 0] = 1.0  # First constraint, parent, related to angular x
        engine._J_j_c[0, 0] = -1.0  # First constraint, child, opposite to parent

        engine._J_j_p = torch.randn_like(engine._J_j_p)
        engine._J_j_c = torch.randn_like(engine._J_j_c)

        engine._body_trans = model.body_q.clone()
        engine._body_vel_prev = torch.zeros((B, 6, 1), device=device)
        engine._body_force = torch.zeros((B, 6, 1), device=device)

        # Create test variables
        body_vel = torch.randn(B, 6, 1, device=device) * 0.1
        lambda_n = torch.zeros(B, C, 1, device=device)
        lambda_t = torch.zeros(B, 2 * C, 1, device=device)
        gamma = torch.zeros(B, C, 1, device=device)
        lambda_j = torch.zeros(5 * D, 1, device=device)

        return {
            "engine": engine,
            "body_vel": body_vel,
            "lambda_n": lambda_n,
            "lambda_t": lambda_t,
            "gamma": gamma,
            "lambda_j": lambda_j,
            "dt": dt,
            "body_count": B,
            "max_contacts": C,
            "joint_count": D
        }

    def test_jacobian_structure(self, setup_engine):
        """Test the basic structure of the computed Jacobian."""
        engine = setup_engine["engine"]
        body_vel = setup_engine["body_vel"]
        lambda_n = setup_engine["lambda_n"]
        lambda_t = setup_engine["lambda_t"]
        gamma = setup_engine["gamma"]
        lambda_j = setup_engine["lambda_j"]
        dt = setup_engine["dt"]

        B = setup_engine["body_count"]
        C = setup_engine["max_contacts"]
        D = setup_engine["joint_count"]

        total_dim_batched = 6 + 4 * C  # Per-body variables
        full_size = B * total_dim_batched + 5 * D  # Total size including unbatched lambda_j

        # Compute the Jacobian
        J_F = engine.compute_jacobian(body_vel, lambda_n, lambda_t, gamma, lambda_j, dt)

        # Check that the Jacobian has the correct size
        assert J_F.shape == (full_size, full_size)

        # Check that the Jacobian is not empty
        assert J_F._values().numel() > 0

    def test_jacobian_joint_constraints(self, setup_engine):
        """Test the Jacobian computation specifically for joint constraints."""
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        engine = setup_engine["engine"]
        body_vel = setup_engine["body_vel"]
        lambda_n = setup_engine["lambda_n"]
        lambda_t = setup_engine["lambda_t"]
        gamma = setup_engine["gamma"]
        lambda_j = setup_engine["lambda_j"]
        dt = setup_engine["dt"]

        B = setup_engine["body_count"]
        C = setup_engine["max_contacts"]
        D = setup_engine["joint_count"]

        total_dim_batched = 6 + 4 * C  # Per-body variables
        full_size = B * total_dim_batched + 5 * D  # Total size

        # Create directory for visualizations
        os.makedirs("jacobian_viz", exist_ok=True)

        # Get analytical Jacobian
        J_F_analytical = engine.compute_jacobian(body_vel, lambda_n, lambda_t, gamma, lambda_j, dt)
        J_F_dense = J_F_analytical.coalesce().to_dense()

        # Function to compute residuals as a flat vector
        def compute_flat_residuals(bv, ln, lt, g, lj):
            res_d, res_n, res_t, res_j = engine.compute_residuals(bv, ln, lt, g, lj, dt)
            res_batched = torch.cat((res_d, res_n, res_t), dim=1)
            return torch.cat((res_batched.view(-1, 1), res_j), dim=0).squeeze(-1)

        # Compute residuals at the current point
        F_x = compute_flat_residuals(body_vel, lambda_n, lambda_t, gamma, lambda_j)

        # Compute full numerical Jacobian
        eps = 1e-4
        J_F_numerical = torch.zeros_like(J_F_dense)

        for i in range(full_size):
            x_plus = torch.zeros(full_size, device=engine.device)
            x_plus[i] = eps

            # Reshape back to original format
            if i < B * total_dim_batched:  # Batched variables
                idx_body = i // total_dim_batched
                idx_var = i % total_dim_batched

                if idx_var < 6:  # body_vel
                    bv_plus = body_vel.clone()
                    bv_plus[idx_body, idx_var, 0] += eps
                    F_plus = compute_flat_residuals(bv_plus, lambda_n, lambda_t, gamma, lambda_j)
                elif idx_var < 6 + C:  # lambda_n
                    ln_plus = lambda_n.clone()
                    ln_plus[idx_body, idx_var - 6, 0] += eps
                    F_plus = compute_flat_residuals(body_vel, ln_plus, lambda_t, gamma, lambda_j)
                elif idx_var < 6 + 3 * C:  # lambda_t
                    lt_plus = lambda_t.clone()
                    lt_plus[idx_body, idx_var - 6 - C, 0] += eps
                    F_plus = compute_flat_residuals(body_vel, lambda_n, lt_plus, gamma, lambda_j)
                else:  # gamma
                    g_plus = gamma.clone()
                    g_plus[idx_body, idx_var - 6 - 3 * C, 0] += eps
                    F_plus = compute_flat_residuals(body_vel, lambda_n, lambda_t, g_plus, lambda_j)
            else:  # lambda_j (unbatched)
                lj_plus = lambda_j.clone()
                lj_plus[i - B * total_dim_batched, 0] += eps
                F_plus = compute_flat_residuals(body_vel, lambda_n, lambda_t, gamma, lj_plus)

            # Compute derivative
            J_F_numerical[:, i] = (F_plus - F_x) / eps

        # Visualize the full analytical and numerical Jacobians
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Convert to numpy for visualization
        full_analytical_np = J_F_dense.cpu().numpy()
        full_numerical_np = J_F_numerical.cpu().numpy()
        full_diff_np = np.abs(full_analytical_np - full_numerical_np)

        # Plot analytical Jacobian
        im0 = axes[0].imshow(full_analytical_np, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[0].set_title('Analytical Jacobian')
        plt.colorbar(im0, ax=axes[0])

        # Plot numerical Jacobian
        im1 = axes[1].imshow(full_numerical_np, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[1].set_title('Numerical Jacobian')
        plt.colorbar(im1, ax=axes[1])

        # Threshold the difference array to only show significant differences
        threshold = 0.05
        full_diff_thresholded = np.copy(full_diff_np)
        full_diff_thresholded[full_diff_thresholded < threshold] = 0

        # Plot thresholded difference
        im2 = axes[2].imshow(full_diff_thresholded, aspect='auto', cmap='hot', interpolation='nearest')
        axes[2].set_title(f'Absolute Difference (Threshold: {threshold})')
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.savefig("jacobian_viz/jacobian_comparison.png")
        plt.close()

        # Now focus on the joint constraint rows
        joint_rows_start = B * total_dim_batched
        joint_rows_end = joint_rows_start + 5 * D

        # Extract numerical derivative for joint constraints with respect to body velocities
        J_F_numerical_joint = J_F_numerical[joint_rows_start:joint_rows_end, :6 * B]

        # Extract analytical joint constraint derivatives
        dres_j_dbody_vel, dres_j_dlambda_j = engine.revolute_constraint.get_derivatives(
            body_vel, engine._body_trans, engine._J_j_p, engine._J_j_c, dt
        )

        # Extract the joint constraint rows from the full Jacobian
        J_F_joint_rows = J_F_dense[joint_rows_start:joint_rows_end, :]
        J_F_joint_vs_vel = J_F_joint_rows[:, :6 * B]  # Only vel columns

        # Visualize the joint constraint derivatives
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Convert to numpy for visualization
        analytical_np = J_F_joint_vs_vel.cpu().numpy()
        numerical_np = J_F_numerical_joint.cpu().numpy()
        dres_j_dbody_vel_np = dres_j_dbody_vel.cpu().numpy()
        diff_np = np.abs(analytical_np - numerical_np)

        # Plot analytical Jacobian from full J_F
        im0 = axes[0].imshow(analytical_np, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[0].set_title('Analytical Joint Jacobian (from full J_F)')
        plt.colorbar(im0, ax=axes[0])

        # Plot numerical Jacobian
        im1 = axes[1].imshow(numerical_np, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[1].set_title('Numerical Joint Jacobian')
        plt.colorbar(im1, ax=axes[1])

        # Plot difference
        threshold = 0.05
        diff_thresholded = np.copy(diff_np)
        diff_thresholded[diff_thresholded < threshold] = 0

        im2 = axes[2].imshow(diff_thresholded, aspect='auto', cmap='hot', interpolation='nearest')
        axes[2].set_title(f'Absolute Difference (Threshold: {threshold})')
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.savefig("jacobian_viz/joint_jacobian_comparison.png")
        plt.close()

        # Now compare the directly computed derivatives with the analytical Jacobian
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot directly computed analytical derivatives
        im0 = axes[0].imshow(dres_j_dbody_vel_np, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[0].set_title('Direct Analytical Joint Derivatives')
        plt.colorbar(im0, ax=axes[0])

        # Plot analytical Jacobian from full J_F
        im1 = axes[1].imshow(analytical_np, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[1].set_title('Full Jacobian Joint Rows')
        plt.colorbar(im1, ax=axes[1])

        # Plot difference between direct derivatives and full Jacobian
        diff_direct_np = np.abs(dres_j_dbody_vel_np - analytical_np)
        diff_direct_thresholded = np.copy(diff_direct_np)
        diff_direct_thresholded[diff_direct_thresholded < threshold] = 0

        im2 = axes[2].imshow(diff_direct_thresholded, aspect='auto', cmap='hot', interpolation='nearest')
        axes[2].set_title('Diff: Direct vs Full Jacobian')
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.savefig("jacobian_viz/direct_vs_full_joint_jacobian.png")
        plt.close()

        # Find indices with large differences
        large_diff_indices = np.where(diff_np > 0.1)
        if len(large_diff_indices[0]) > 0:
            print(f"Found {len(large_diff_indices[0])} entries with difference > 0.1 between numerical and analytical:")
            for i in range(min(20, len(large_diff_indices[0]))):
                r, c = large_diff_indices[0][i], large_diff_indices[1][i]
                print(
                    f"  Joint row {r}, Velocity column {c}: Analytical={analytical_np[r, c]:.6f}, Numerical={numerical_np[r, c]:.6f}, Diff={diff_np[r, c]:.6f}")

        # Check difference between direct computation and full Jacobian
        large_diff_direct = np.where(diff_direct_np > 0.1)
        if len(large_diff_direct[0]) > 0:
            print(f"Found {len(large_diff_direct[0])} entries with difference > 0.1 between direct and full Jacobian:")
            for i in range(min(20, len(large_diff_direct[0]))):
                r, c = large_diff_direct[0][i], large_diff_direct[1][i]
                print(
                    f"  Row {r}, Col {c}: Direct={dres_j_dbody_vel_np[r, c]:.6f}, Full J={analytical_np[r, c]:.6f}, Diff={diff_direct_np[r, c]:.6f}")

        # Now let's analyze the specific part of the code that might be problematic
        print("\nAnalyzing the joint block in compute_jacobian():")
        print(f"dres_j_dbody_vel shape: {dres_j_dbody_vel.shape}")

        # Analyze how nonzero entries are extracted and indexed
        nonzero_mask = dres_j_dbody_vel != 0
        if nonzero_mask.any():
            nonzero_count = nonzero_mask.sum().item()
            print(f"Total nonzero entries in dres_j_dbody_vel: {nonzero_count}")

            # Get row and column indices of nonzeros
            rows, cols = torch.where(nonzero_mask)
            print(f"Sample of nonzero indices (up to 10):")
            for i in range(min(10, rows.shape[0])):
                r, c = rows[i].item(), cols[i].item()
                print(f"  ({r}, {c}): {dres_j_dbody_vel[r, c].item():.6f}")

            # Verify the indexing by reconstructing a sparse tensor
            col_indices_base = cols
            row_indices = rows + B * total_dim_batched

            # The issue might be here - cols might need adjustment
            # Let's check how they're distributed
            col_counts = torch.bincount(cols, minlength=6 * B)
            print(f"Column distribution: {col_counts.tolist()}")

            # Check if there are columns with indices >= 6 (which would be invalid for body velocities)
            if (cols >= 6 * B).any():
                print("WARNING: Some column indices exceed the expected range for body velocities!")

        # Test an alternative implementation to see if it fixes the issue
        print("\nTesting alternative implementation for joint block:")

        # Alternative implementation that carefully maps the indices
        all_row_indices_alt = []
        all_col_indices_alt = []
        all_values_alt = []

        # Joint block: ∂res_j/∂v (alternative implementation)
        for joint_row in range(5 * D):
            for body_idx in range(B):
                for vel_idx in range(6):
                    col_idx = body_idx * 6 + vel_idx
                    if col_idx < 6 * B and dres_j_dbody_vel[joint_row, col_idx] != 0:
                        row_idx = joint_row + B * total_dim_batched
                        all_row_indices_alt.append(row_idx)
                        all_col_indices_alt.append(col_idx)
                        all_values_alt.append(dres_j_dbody_vel[joint_row, col_idx].item())

        print(f"Alternative implementation found {len(all_row_indices_alt)} nonzero entries")
        if all_row_indices_alt:
            print("Sample entries (up to 10):")
            for i in range(min(10, len(all_row_indices_alt))):
                r, c, v = all_row_indices_alt[i], all_col_indices_alt[i], all_values_alt[i]
                print(f"  ({r}, {c}): {v:.6f}")

            # Check how these entries would be placed in the full Jacobian
            row_offset = B * total_dim_batched
            for i in range(min(5, len(all_row_indices_alt))):
                r, c, v = all_row_indices_alt[i], all_col_indices_alt[i], all_values_alt[i]
                print(f"  Joint constraint {r - row_offset} affects body velocity component {c}")

        # Compare the specific entries in the joint rows between analytical and numerical results
        print("\nAnalyzing specific entries in the joint constraint rows:")
        joint_rows_indices = list(range(joint_rows_start, joint_rows_end))
        vel_cols_indices = list(range(6 * B))

        # Identify entries with significant differences
        for row in joint_rows_indices:
            row_rel = row - joint_rows_start  # Relative row index within joint constraints
            for col in vel_cols_indices:
                analytical_val = J_F_dense[row, col].item()
                numerical_val = J_F_numerical[row, col].item()
                diff_val = abs(analytical_val - numerical_val)

                if diff_val > 0.1:
                    body_idx = col // 6
                    vel_component = col % 6
                    print(f"Row {row_rel}, Col {col} (Body {body_idx}, Vel {vel_component}): "
                          f"Analytical={analytical_val:.6f}, Numerical={numerical_val:.6f}, Diff={diff_val:.6f}")

    def test_jacobian_valid_sparse_tensor(self, setup_engine):
        """Test that the jacobian is a valid sparse tensor without duplicate indices."""
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        engine = setup_engine["engine"]
        body_vel = setup_engine["body_vel"]
        lambda_n = setup_engine["lambda_n"]
        lambda_t = setup_engine["lambda_t"]
        gamma = setup_engine["gamma"]
        lambda_j = setup_engine["lambda_j"]
        dt = setup_engine["dt"]

        B = setup_engine["body_count"]
        C = setup_engine["max_contacts"]
        D = setup_engine["joint_count"]

        total_dim_batched = 6 + 4 * C  # Per-body variables
        full_size = B * total_dim_batched + 5 * D  # Total size

        # Create directory for visualizations
        os.makedirs("jacobian_viz", exist_ok=True)

        # Compute the Jacobian
        J_F = engine.compute_jacobian(body_vel, lambda_n, lambda_t, gamma, lambda_j, dt)

        # Visualize the structure of the sparse tensor
        indices = J_F._indices()

        # Create a binary matrix showing where non-zero values are
        sparse_matrix = np.zeros((full_size, full_size))
        row_indices = indices[0].cpu().numpy()
        col_indices = indices[1].cpu().numpy()
        for r, c in zip(row_indices, col_indices):
            sparse_matrix[r, c] = 1

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(sparse_matrix, aspect='auto', cmap='Blues', interpolation='nearest')
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        ax.set_title('Sparse Jacobian Structure')
        plt.tight_layout()
        plt.savefig("jacobian_viz/jacobian_structure.png")
        plt.close()

        # Check for duplicate indices
        indices_transposed = indices.t().cpu()  # Shape: [nnz, 2]

        # Check for duplicates manually
        indices_dict = {}
        for i, (row, col) in enumerate(indices_transposed):
            key = (row.item(), col.item())
            if key in indices_dict:
                indices_dict[key].append(i)
            else:
                indices_dict[key] = [i]

        duplicates = {k: v for k, v in indices_dict.items() if len(v) > 1}

        if duplicates:
            print(f"Found {len(duplicates)} duplicate index pairs")

            # Visualize duplicate entries
            duplicate_matrix = np.zeros((full_size, full_size))
            for (row, col) in duplicates.keys():
                duplicate_matrix[row, col] = 1

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(duplicate_matrix, aspect='auto', cmap='Reds', interpolation='nearest')
            ax.set_xlabel('Column Index')
            ax.set_ylabel('Row Index')
            ax.set_title('Duplicate Indices in Jacobian')
            plt.tight_layout()
            plt.savefig("jacobian_viz/duplicate_indices.png")
            plt.close()

            # Print the values of these duplicates
            print("Sample of duplicate entries (up to 10):")
            values = J_F._values().cpu()
            for i, (key, indices_list) in enumerate(duplicates.items()):
                if i >= 10:  # Limit to 10 examples
                    break
                row, col = key
                print(f"  ({row}, {col}):")
                for idx in indices_list:
                    print(f"    Value at index {idx}: {values[idx].item()}")

            # Analyze duplicate locations to understand where the duplication comes from
            blocks = {
                "Dynamics+Body": (0, 6, 0, 6),  # (start_row, end_row, start_col, end_col)
                "Dynamics+LambdaN": (0, 6, 6, 6 + C),
                "Dynamics+LambdaT": (0, 6, 6 + C, 6 + 3 * C),
                "Dynamics+Gamma": (0, 6, 6 + 3 * C, 6 + 4 * C),
                "Contact+Body": (6, 6 + C, 0, 6),
                "Contact+LambdaN": (6, 6 + C, 6, 6 + C),
                "Friction+Body": (6 + C, 6 + 4 * C, 0, 6),
                "Joint+Body": (B * total_dim_batched, full_size, 0, 6 * B)
            }

            for block_name, (r_start, r_end, c_start, c_end) in blocks.items():
                block_duplicates = [(r, c) for (r, c) in duplicates if r_start <= r < r_end and c_start <= c < c_end]
                if block_duplicates:
                    print(f"Block {block_name} has {len(block_duplicates)} duplicate entries")

            # Try to coalesce the tensor
            try:
                J_F_coalesced = J_F.coalesce()
                print("Successfully coalesced the tensor")

                # Check if some specific entries match expected values after coalescing
                dynamics_block = J_F_coalesced.to_dense()[:6, :6]
                try:
                    assert torch.allclose(dynamics_block, engine.dynamics_constraint.mass_matrix[0], atol=1e-5)
                    print("After coalescing, mass matrix block is correct")
                except AssertionError as e:
                    print(f"After coalescing, mass matrix block is INCORRECT: {e}")

            except RuntimeError as e:
                print(f"Failed to coalesce the tensor: {e}")

        else:
            print("No duplicate indices found - tensor is already coalesced!")

        try:
            # Try to coalesce the tensor first
            J_F = J_F.coalesce()
            indices = J_F.indices()
            size = J_F.size()

            assert (indices[0] >= 0).all() and (indices[0] < size[0]).all(), "Row indices out of bounds"
            assert (indices[1] >= 0).all() and (indices[1] < size[1]).all(), "Column indices out of bounds"
            print("Index bounds check: PASSED")
        except Exception as e:
            print(f"Index bounds check: FAILED - {e}")

    def test_jacobian_consistent_with_residuals(self, setup_engine):
        """Test that the Jacobian is consistent with the residual computation."""
        engine = setup_engine["engine"]
        body_vel = setup_engine["body_vel"]
        lambda_n = setup_engine["lambda_n"]
        lambda_t = setup_engine["lambda_t"]
        gamma = setup_engine["gamma"]
        lambda_j = setup_engine["lambda_j"]
        dt = setup_engine["dt"]

        # Compute residuals
        res_d, res_n, res_t, res_j = engine.compute_residuals(
            body_vel, lambda_n, lambda_t, gamma, lambda_j, dt
        )

        # Compute Jacobian
        J_F = engine.compute_jacobian(body_vel, lambda_n, lambda_t, gamma, lambda_j, dt)

        # Perturbation test - check that a small change in variables results
        # in the change predicted by the Jacobian
        eps = 1e-6

        # Perturb body velocity for first body
        bv_perturbed = body_vel.clone()
        bv_perturbed[0, 0, 0] += eps  # Perturb angular x-velocity of first body

        # Compute new residuals
        res_d_new, res_n_new, res_t_new, res_j_new = engine.compute_residuals(
            bv_perturbed, lambda_n, lambda_t, gamma, lambda_j, dt
        )

        # Compute predicted change using Jacobian
        J_F_dense = J_F.to_dense()

        # Index for the perturbed variable in the flattened vector
        B = setup_engine["body_count"]
        C = setup_engine["max_contacts"]
        total_dim_batched = 6 + 4 * C  # Per-body variables
        perturb_idx = 0  # First angular velocity component of first body

        # Extract the column of the Jacobian corresponding to the perturbed variable
        jacobian_col = J_F_dense[:, perturb_idx]

        # Compute predicted change in first body dynamics residuals
        pred_change_d = jacobian_col[:6] * eps
        actual_change_d = (res_d_new - res_d)[0, :, 0]

        # Compare
        assert torch.allclose(pred_change_d, actual_change_d, atol=1e-4), \
            f"Jacobian prediction mismatch: pred={pred_change_d}, actual={actual_change_d}"


if __name__ == "__main__":
    pytest.main(["-v"])