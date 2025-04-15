import torch
import pytest
from pbd_torch.constraints import DynamicsConstraint
from pbd_torch.constraints import ContactConstraint
from pbd_torch.constraints import FrictionConstraint
from pbd_torch.constraints import RevoluteConstraint
from pbd_torch.transform import normalize_quat_batch
import matplotlib.pyplot as plt

def visualize_jacobian_difference(analytical, numerical, title, filename):
    B = analytical.shape[0]
    for i in range(B):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        diff = torch.abs(analytical[i] - numerical[i]).detach().cpu().numpy()
        analytical_np = analytical[i].detach().cpu().numpy()
        numerical_np = numerical[i].detach().cpu().numpy()

        # Analytical heatmap
        im0 = axs[0].imshow(analytical_np, cmap='viridis', aspect='auto')
        axs[0].set_title(f"Analytical (Body {i})")
        plt.colorbar(im0, ax=axs[0])

        # Numerical heatmap
        im1 = axs[1].imshow(numerical_np, cmap='viridis', aspect='auto')
        axs[1].set_title(f"Numerical (Body {i})")
        plt.colorbar(im1, ax=axs[1])

        # Difference heatmap
        im2 = axs[2].imshow(diff, cmap='hot', aspect='auto')
        axs[2].set_title(f"Difference (Body {i})")
        plt.colorbar(im2, ax=axs[2])

        # Label axes based on derivative type
        if "body_vel" in title:
            axs[0].set_xlabel("Body Velocity (6)")
            axs[0].set_ylabel("Residuals (6)")
        elif "lambda_n" in title:
            axs[0].set_xlabel("Lambda_n (C = 3)")
            axs[0].set_ylabel("Residuals (6)")
        elif "lambda_t" in title:
            axs[0].set_xlabel("Lambda_t (2*C = 6)")
            axs[0].set_ylabel("Residuals (6)")
        elif "lambda_j" in title:
            axs[0].set_xlabel("Lambda_j (5*D = 5)")
            axs[0].set_ylabel("Residuals (6)")

        plt.suptitle(f"{title} - Body {i}")
        plt.tight_layout()
        plt.savefig(f"{filename}_body_{i}.png")
        plt.close()

        if not torch.allclose(analytical[i], numerical[i], atol=1e-6):
            print(f"\n{title} - Body {i} failed:")
            print("Analytical:\n", analytical[i])
            print("Numerical:\n", numerical[i])
            print("Difference:\n", diff)

class TestDynamicsConstraint:
    @pytest.fixture
    def setup_constraint(self):
        device = torch.device("cpu")
        body_count = 2
        max_contacts = 3
        joint_count = 1
        dt = 0.01
        torch.random.manual_seed(0)

        # Mock Model for joint data
        class MockModel:
            def __init__(self):
                self.device = device
                self.mass_matrix = torch.randn(body_count, 6, 6, device=device)
                self.mass_matrix = self.mass_matrix @ self.mass_matrix.transpose(1, 2)  # Positive definite
                self.g_accel = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -9.81], device=device).view(6, 1)
                self.joint_parent = torch.tensor([0], dtype=torch.long, device=device)
                self.joint_child = torch.tensor([1], dtype=torch.long, device=device)

        model = MockModel()
        constraint = DynamicsConstraint(model=model)

        body_vel = torch.randn(body_count, 6, 1, device=device)
        body_vel_prev = torch.randn(body_count, 6, 1, device=device)
        lambda_n = torch.randn(body_count, max_contacts, 1, device=device)
        lambda_t = torch.randn(body_count, 2 * max_contacts, 1, device=device)
        lambda_j = torch.randn(5 * joint_count, 1, device=device)
        body_f = torch.randn(body_count, 6, 1, device=device)
        J_n = torch.randn(body_count, max_contacts, 6, device=device)
        J_t = torch.randn(body_count, 2 * max_contacts, 6, device=device)
        J_j_p = torch.randn(5 * joint_count, 6, device=device)
        J_j_c = torch.randn(5 * joint_count, 6, device=device)

        return {
            "constraint": constraint,
            "body_vel": body_vel,
            "body_vel_prev": body_vel_prev,
            "lambda_n": lambda_n,
            "lambda_t": lambda_t,
            "lambda_j": lambda_j,
            "body_f": body_f,
            "J_n": J_n,
            "J_t": J_t,
            "J_j_p": J_j_p,
            "J_j_c": J_j_c,
            "dt": dt,
            "body_count": body_count,
            "max_contacts": max_contacts,
            "joint_count": joint_count
        }

    def test_gradient_body_vel(self, setup_constraint):
        """Test derivative with respect to body_vel"""
        c = setup_constraint["constraint"]
        inputs = {
            "body_vel_prev": setup_constraint["body_vel_prev"].clone(),
            "lambda_n": setup_constraint["lambda_n"].clone(),
            "lambda_t": setup_constraint["lambda_t"].clone(),
            "lambda_j": setup_constraint["lambda_j"].clone(),
            "body_f": setup_constraint["body_f"].clone(),
            "J_n": setup_constraint["J_n"].clone(),
            "J_t": setup_constraint["J_t"].clone(),
            "J_j_p": setup_constraint["J_j_p"].clone(),
            "J_j_c": setup_constraint["J_j_c"].clone(),
            "dt": setup_constraint["dt"]
        }
        body_vel = setup_constraint["body_vel"].clone().requires_grad_(True)

        # Analytical gradient
        analytical_grad, _, _, _ = c.get_derivatives(
            J_n=inputs["J_n"],
            J_t=inputs["J_t"],
            J_j_c=inputs["J_j_c"],
            J_j_p=inputs["J_j_p"]
        )  # [B, 6, 6]

        # Numerical gradient using autograd
        residuals = c.get_residuals(body_vel=body_vel, **inputs)  # [B, 6, 1]
        numerical_grad_rows = []
        for i in range(setup_constraint["body_count"]):
            grad_rows_per_body = []
            for j in range(6):
                grad_output = torch.zeros_like(residuals)
                grad_output[i, j, 0] = 1.0
                grad = torch.autograd.grad(residuals, body_vel, grad_outputs=grad_output, create_graph=True)[0][i]  # [6, 1]
                grad_rows_per_body.append(grad.squeeze(-1))  # [6]
            numerical_grad_rows.append(torch.stack(grad_rows_per_body))  # [6, 6]
        numerical_grad = torch.stack(numerical_grad_rows)  # [B, 6, 6]

        # Compare
        assert analytical_grad.shape == (setup_constraint["body_count"], 6, 6)
        assert numerical_grad.shape == analytical_grad.shape
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-6)
        assert torch.allclose(analytical_grad, c.mass_matrix, atol=1e-6)

    def test_gradient_lambda_n(self, setup_constraint):
        """Test derivative with respect to lambda_n"""
        c = setup_constraint["constraint"]
        inputs = {
            "body_vel": setup_constraint["body_vel"].clone(),
            "body_vel_prev": setup_constraint["body_vel_prev"].clone(),
            "lambda_n": setup_constraint["lambda_n"].clone().requires_grad_(True),
            "lambda_t": setup_constraint["lambda_t"].clone(),
            "lambda_j": setup_constraint["lambda_j"].clone(),
            "body_f": setup_constraint["body_f"].clone(),
            "J_n": setup_constraint["J_n"].clone(),
            "J_t": setup_constraint["J_t"].clone(),
            "J_j_p": setup_constraint["J_j_p"].clone(),
            "J_j_c": setup_constraint["J_j_c"].clone(),
            "dt": setup_constraint["dt"]
        }

        # Analytical gradient
        _, analytical_grad, _, _ = c.get_derivatives(
            J_n=inputs["J_n"],
            J_t=inputs["J_t"],
            J_j_c=inputs["J_j_c"],
            J_j_p=inputs["J_j_p"]
        )

        # Numerical gradient using autograd
        residuals = c.get_residuals(**inputs)  # [B, 6, 1]
        numerical_grad_rows = []
        for i in range(setup_constraint["body_count"]):
            grad_rows_per_body = []
            for j in range(6):
                grad_output = torch.zeros_like(residuals)
                grad_output[i, j, 0] = 1.0
                grad = torch.autograd.grad(residuals, inputs["lambda_n"], grad_outputs=grad_output, create_graph=True)[0][i]  # [C, 1]
                grad_rows_per_body.append(grad.squeeze(-1))  # [C]
            numerical_grad_rows.append(torch.stack(grad_rows_per_body))  # [6, C]
        numerical_grad = torch.stack(numerical_grad_rows)  # [B, 6, C]

        # Compare
        assert analytical_grad.shape == (setup_constraint["body_count"], 6, setup_constraint["max_contacts"])
        assert numerical_grad.shape == analytical_grad.shape
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-6)
        assert torch.allclose(analytical_grad, -inputs["J_n"].transpose(1, 2), atol=1e-6)

    def test_gradient_lambda_t(self, setup_constraint):
        """Test derivative with respect to lambda_t"""
        c = setup_constraint["constraint"]
        inputs = {
            "body_vel": setup_constraint["body_vel"].clone(),
            "body_vel_prev": setup_constraint["body_vel_prev"].clone(),
            "lambda_n": setup_constraint["lambda_n"].clone(),
            "lambda_t": setup_constraint["lambda_t"].clone().requires_grad_(True),
            "lambda_j": setup_constraint["lambda_j"].clone(),
            "body_f": setup_constraint["body_f"].clone(),
            "J_n": setup_constraint["J_n"].clone(),
            "J_t": setup_constraint["J_t"].clone(),
            "J_j_p": setup_constraint["J_j_p"].clone(),
            "J_j_c": setup_constraint["J_j_c"].clone(),
            "dt": setup_constraint["dt"]
        }

        # Analytical gradient
        _, _, analytical_grad, _ = c.get_derivatives(
            J_n=inputs["J_n"],
            J_t=inputs["J_t"],
            J_j_c=inputs["J_j_c"],
            J_j_p=inputs["J_j_p"]
        )

        # Numerical gradient using autograd
        residuals = c.get_residuals(**inputs)  # [B, 6, 1]
        numerical_grad_rows = []
        for i in range(setup_constraint["body_count"]):
            grad_rows_per_body = []
            for j in range(6):
                grad_output = torch.zeros_like(residuals)
                grad_output[i, j, 0] = 1.0
                grad = torch.autograd.grad(residuals, inputs["lambda_t"], grad_outputs=grad_output, create_graph=True)[0][i]  # [2C, 1]
                grad_rows_per_body.append(grad.squeeze(-1))  # [2C]
            numerical_grad_rows.append(torch.stack(grad_rows_per_body))  # [6, 2C]
        numerical_grad = torch.stack(numerical_grad_rows)  # [B, 6, 2C]

        # Compare
        assert analytical_grad.shape == (setup_constraint["body_count"], 6, 2 * setup_constraint["max_contacts"])
        assert numerical_grad.shape == analytical_grad.shape
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-6)
        assert torch.allclose(analytical_grad, -inputs["J_t"].transpose(1, 2), atol=1e-6)

    def test_gradient_lambda_j(self, setup_constraint):
        """Test derivative with respect to lambda_j"""
        c = setup_constraint["constraint"]
        inputs = {
            "body_vel": setup_constraint["body_vel"].clone(),
            "body_vel_prev": setup_constraint["body_vel_prev"].clone(),
            "lambda_n": setup_constraint["lambda_n"].clone(),
            "lambda_t": setup_constraint["lambda_t"].clone(),
            "lambda_j": setup_constraint["lambda_j"].clone().requires_grad_(True),
            "body_f": setup_constraint["body_f"].clone(),
            "J_n": setup_constraint["J_n"].clone(),
            "J_t": setup_constraint["J_t"].clone(),
            "J_j_p": setup_constraint["J_j_p"].clone(),
            "J_j_c": setup_constraint["J_j_c"].clone(),
            "dt": setup_constraint["dt"]
        }

        # Analytical gradient
        _, _, _, analytical_grad = c.get_derivatives(
            J_n=inputs["J_n"],
            J_t=inputs["J_t"],
            J_j_c=inputs["J_j_c"],
            J_j_p=inputs["J_j_p"]
        )  # [B, 6, 5 * D]

        # Numerical gradient using autograd
        residuals = c.get_residuals(**inputs)  # [B, 6, 1]
        numerical_grad_rows = []
        for i in range(setup_constraint["body_count"]):
            grad_rows_per_body = []
            for j in range(6):
                grad_output = torch.zeros_like(residuals)
                grad_output[i, j, 0] = 1.0
                grad = torch.autograd.grad(residuals, inputs["lambda_j"], grad_outputs=grad_output, create_graph=True)[0]  # [5*D, 1]
                grad_rows_per_body.append(grad.squeeze(-1))  # [5*D]
            numerical_grad_rows.append(torch.stack(grad_rows_per_body))  # [6, 5*D]
        numerical_grad = torch.stack(numerical_grad_rows)  # [B, 6, 5*D]

        # Optionally visualize differences
        # visualize_jacobian_difference(analytical_grad, numerical_grad, "Jacobian w.r.t. lambda_j", "lambda_j_jacobian")

        # Compare
        assert analytical_grad.shape == (setup_constraint["body_count"], 6, 5 * setup_constraint["joint_count"])
        assert numerical_grad.shape == analytical_grad.shape
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-6)

    def test_gradient_independence(self, setup_constraint):
        """Test that gradients don't depend on unused variables"""
        c = setup_constraint["constraint"]
        inputs = {
            "body_vel": setup_constraint["body_vel"].clone().requires_grad_(True),
            "body_vel_prev": setup_constraint["body_vel_prev"].clone().requires_grad_(True),
            "lambda_n": setup_constraint["lambda_n"].clone().requires_grad_(True),
            "lambda_t": setup_constraint["lambda_t"].clone().requires_grad_(True),
            "lambda_j": setup_constraint["lambda_j"].clone().requires_grad_(True),
            "body_f": setup_constraint["body_f"].clone().requires_grad_(True),
            "J_n": setup_constraint["J_n"].clone(),
            "J_t": setup_constraint["J_t"].clone(),
            "J_j_p": setup_constraint["J_j_p"].clone(),
            "J_j_c": setup_constraint["J_j_c"].clone(),
            "dt": setup_constraint["dt"]
        }

        # Get analytical derivatives
        dvel, dln, dlt, dlj = c.get_derivatives(
            J_n=inputs["J_n"],
            J_t=inputs["J_t"],
            J_j_c=inputs["J_j_c"],
            J_j_p=inputs["J_j_p"]
        )

        # Modify unused variables
        inputs_modified = {
            "body_vel": setup_constraint["body_vel"].clone().requires_grad_(True),
            "body_vel_prev": torch.zeros_like(setup_constraint["body_vel_prev"]).requires_grad_(True),
            "lambda_n": setup_constraint["lambda_n"].clone().requires_grad_(True),
            "lambda_t": setup_constraint["lambda_t"].clone().requires_grad_(True),
            "lambda_j": setup_constraint["lambda_j"].clone().requires_grad_(True),
            "body_f": torch.zeros_like(setup_constraint["body_f"]).requires_grad_(True),
            "J_n": setup_constraint["J_n"].clone(),
            "J_t": setup_constraint["J_t"].clone(),
            "J_j_p": setup_constraint["J_j_p"].clone(),
            "J_j_c": setup_constraint["J_j_c"].clone(),
            "dt": setup_constraint["dt"]
        }

        dvel2, dln2, dlt2, dlj2 = c.get_derivatives(
            J_n=inputs_modified["J_n"],
            J_t=inputs_modified["J_t"],
            J_j_c=inputs_modified["J_j_c"],
            J_j_p=inputs_modified["J_j_p"]
        )

        # Compare
        assert torch.allclose(dvel, dvel2, atol=1e-6)
        assert torch.allclose(dln, dln2, atol=1e-6)
        assert torch.allclose(dlt, dlt2, atol=1e-6)
        assert torch.allclose(dlj, dlj2, atol=1e-6)


class TestContactConstraint:
    @pytest.fixture
    def setup_constraint(self):
        device = torch.device("cpu")
        body_count = 2
        max_contacts = 3
        dt = 0.01
        torch.random.manual_seed(0)

        constraint = ContactConstraint(device=device)

        body_vel = torch.randn(body_count, 6, 1)
        body_vel_prev = torch.randn(body_count, 6, 1)
        lambda_n = torch.randn(body_count, max_contacts, 1)
        body_trans = torch.randn(body_count, 7, 1)  # [pos, quat]
        body_trans[:, 3:] = normalize_quat_batch(body_trans[:, 3:])
        contact_points = torch.randn(body_count, max_contacts, 3, 1)
        ground_points = torch.randn(body_count, max_contacts, 3, 1)
        contact_normals = torch.randn(body_count, max_contacts, 3, 1)
        contact_mask = torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool)
        restitution = torch.ones(body_count, 1) * 0.8
        penetration_depth = constraint.get_penetration_depths(body_trans, contact_points, ground_points, contact_normals)
        J_n = constraint.compute_contact_jacobians(body_trans, contact_points, contact_normals, contact_mask)

        return {
            "constraint": constraint,
            "body_vel": body_vel,
            "body_vel_prev": body_vel_prev,
            "lambda_n": lambda_n,
            "J_n": J_n,
            "penetration_depth": penetration_depth,
            "contact_mask": contact_mask,
            "restitution": restitution,
            "dt": dt,
            "body_count": body_count,
            "max_contacts": max_contacts
        }

    def test_gradient_body_vel(self, setup_constraint):
        """Test derivative with respect to body_vel"""
        c = setup_constraint["constraint"]
        inputs = {
            "body_vel_prev": setup_constraint["body_vel_prev"].clone(),
            "lambda_n": setup_constraint["lambda_n"].clone(),
            "J_n": setup_constraint["J_n"].clone(),
            "penetration_depth": setup_constraint["penetration_depth"].clone(),
            "contact_mask": setup_constraint["contact_mask"].clone(),
            "restitution": setup_constraint["restitution"].clone(),
            "dt": setup_constraint["dt"]
        }
        body_vel = setup_constraint["body_vel"].clone().requires_grad_(True)

        # Analytical gradient
        analytical_grad, _ = c.get_derivatives(body_vel=body_vel, **inputs)

        # Numerical gradient using autograd (manual Jacobian)
        residuals = c.get_residuals(body_vel=body_vel, **inputs)  # [B, C, 1]
        numerical_grad_rows = []
        for i in range(setup_constraint["body_count"]):
            grad_rows_per_body = []
            for j in range(setup_constraint["max_contacts"]):  # Loop over residuals
                grad_output = torch.zeros_like(residuals)  # [B, C, 1]
                grad_output[i, j, 0] = 1.0
                grad = torch.autograd.grad(residuals, body_vel, grad_outputs=grad_output, create_graph=True)[0][i]  # [6, 1]
                grad_rows_per_body.append(grad.squeeze(-1))  # [6]
            numerical_grad_rows.append(torch.stack(grad_rows_per_body))  # [C, 6]
        numerical_grad = torch.stack(numerical_grad_rows)  # [B, C, 6]

        # Compare shapes and values
        assert analytical_grad.shape == (setup_constraint["body_count"], setup_constraint["max_contacts"], 6)
        assert numerical_grad.shape == analytical_grad.shape
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-6)

    def test_gradient_lambda_n(self, setup_constraint):
        """Test derivative with respect to lambda_n"""
        c = setup_constraint["constraint"]
        inputs = {
            "body_vel": setup_constraint["body_vel"].clone(),
            "body_vel_prev": setup_constraint["body_vel_prev"].clone(),
            "lambda_n": setup_constraint["lambda_n"].clone().requires_grad_(True),
            "J_n": setup_constraint["J_n"].clone(),
            "penetration_depth": setup_constraint["penetration_depth"].clone(),
            "contact_mask": setup_constraint["contact_mask"].clone(),
            "restitution": setup_constraint["restitution"].clone(),
            "dt": setup_constraint["dt"]
        }

        # Analytical gradient
        with torch.no_grad():
            _, analytical_grad = c.get_derivatives(**inputs)

        # Numerical gradient using autograd (manual Jacobian)
        residuals = c.get_residuals(**inputs)  # [B, C, 1]
        numerical_grad_rows = []
        for i in range(setup_constraint["body_count"]):
            grad_rows_per_body = []
            for j in range(setup_constraint["max_contacts"]):  # Loop over residuals
                grad_output = torch.zeros_like(residuals)  # [B, C, 1]
                grad_output[i, j, 0] = 1.0
                grad = torch.autograd.grad(residuals, inputs["lambda_n"], grad_outputs=grad_output, create_graph=True)[0][i]  # [C, 1]
                grad_rows_per_body.append(grad.squeeze(-1))  # [C]
            numerical_grad_rows.append(torch.stack(grad_rows_per_body))  # [C, C]
        numerical_grad = torch.stack(numerical_grad_rows)  # [B, C, C]

        # Compare shapes and values
        assert analytical_grad.shape == (setup_constraint["body_count"], setup_constraint["max_contacts"], setup_constraint["max_contacts"])
        assert numerical_grad.shape == analytical_grad.shape
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-6)


class TestFrictionConstraint:
    @pytest.fixture
    def setup_constraint(self):
        device = torch.device("cpu")
        body_count = 2
        max_contacts = 3
        constraint = FrictionConstraint(device=device)
        body_vel = torch.randn(body_count, 6, 1)
        lambda_n = torch.randn(body_count, max_contacts, 1)
        lambda_t = torch.randn(body_count, 2 * max_contacts, 1)
        gamma = torch.randn(body_count, max_contacts, 1)
        body_trans = torch.randn(body_count, 7, 1)
        contact_points = torch.randn(body_count, max_contacts, 3, 1)
        contact_normals = torch.randn(body_count, max_contacts, 3, 1)
        contact_mask = torch.tensor([[True, True, True], [True, True, True]], dtype=torch.bool)
        friction_coeff = torch.ones(body_count, 1) * 0.5
        J_t = constraint.compute_tangential_jacobians(body_trans, contact_points, contact_normals, contact_mask)
        return {
            "constraint": constraint,
            "body_vel": body_vel,
            "lambda_n": lambda_n,
            "lambda_t": lambda_t,
            "gamma": gamma,
            "J_t": J_t,
            "contact_mask": contact_mask,
            "friction_coeff": friction_coeff,
            "body_count": body_count,
            "max_contacts": max_contacts
        }

    def test_gradient_body_vel(self, setup_constraint):
        c = setup_constraint["constraint"]
        inputs = {
            "lambda_n": setup_constraint["lambda_n"].clone(),
            "lambda_t": setup_constraint["lambda_t"].clone(),
            "gamma": setup_constraint["gamma"].clone(),
            "J_t": setup_constraint["J_t"].clone(),
            "contact_mask": setup_constraint["contact_mask"].clone(),
            "friction_coeff": setup_constraint["friction_coeff"].clone()
        }
        body_vel = setup_constraint["body_vel"].clone().requires_grad_(True)
        analytical_grad, _, _, _ = c.get_derivatives(body_vel=body_vel, **inputs)
        residuals = c.get_residuals(body_vel=body_vel, **inputs)
        numerical_grad_rows = []
        for i in range(setup_constraint["body_count"]):
            grad_rows_per_body = []
            for j in range(3 * setup_constraint["max_contacts"]):
                grad_output = torch.zeros_like(residuals)
                grad_output[i, j, 0] = 1.0
                grad = torch.autograd.grad(residuals, body_vel, grad_outputs=grad_output, create_graph=True)[0][i]
                grad_rows_per_body.append(grad.squeeze(-1))
            numerical_grad_rows.append(torch.stack(grad_rows_per_body))
        numerical_grad = torch.stack(numerical_grad_rows)

        # Visualize differences
        # visualize_jacobian_difference(analytical_grad, numerical_grad, "Jacobian w.r.t. body_vel", "body_vel_jacobian")

        assert analytical_grad.shape == (setup_constraint["body_count"], 3 * setup_constraint["max_contacts"], 6)
        assert numerical_grad.shape == analytical_grad.shape
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-6)

    def test_gradient_lambda_n(self, setup_constraint):
        c = setup_constraint["constraint"]
        inputs = {
            "body_vel": setup_constraint["body_vel"].clone(),
            "lambda_t": setup_constraint["lambda_t"].clone(),
            "gamma": setup_constraint["gamma"].clone(),
            "J_t": setup_constraint["J_t"].clone(),
            "contact_mask": setup_constraint["contact_mask"].clone(),
            "friction_coeff": setup_constraint["friction_coeff"].clone()
        }
        lambda_n = setup_constraint["lambda_n"].clone().requires_grad_(True)
        with torch.no_grad():
            _, analytical_grad, _, _ = c.get_derivatives(lambda_n=lambda_n, **inputs)

        residuals = c.get_residuals(lambda_n=lambda_n, **inputs)
        numerical_grad_rows = []
        for i in range(setup_constraint["body_count"]):
            grad_rows_per_body = []
            for j in range(3 * setup_constraint["max_contacts"]):
                grad_output = torch.zeros_like(residuals)
                grad_output[i, j, 0] = 1.0
                grad = torch.autograd.grad(residuals, lambda_n, grad_outputs=grad_output, create_graph=True)[0][i]
                grad_rows_per_body.append(grad.squeeze(-1))
            numerical_grad_rows.append(torch.stack(grad_rows_per_body))
        numerical_grad = torch.stack(numerical_grad_rows)

        # Visualize differences
        # visualize_jacobian_difference(analytical_grad, numerical_grad, "Jacobian w.r.t. lambda_n", "lambda_n_jacobian")

        assert analytical_grad.shape == (setup_constraint["body_count"], 3 * setup_constraint["max_contacts"], setup_constraint["max_contacts"])
        assert numerical_grad.shape == analytical_grad.shape
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-6)

    def test_gradient_lambda_t(self, setup_constraint):
        c = setup_constraint["constraint"]
        inputs = {
            "body_vel": setup_constraint["body_vel"].clone(),
            "lambda_n": setup_constraint["lambda_n"].clone(),
            "gamma": setup_constraint["gamma"].clone(),
            "J_t": setup_constraint["J_t"].clone(),
            "contact_mask": setup_constraint["contact_mask"].clone(),
            "friction_coeff": setup_constraint["friction_coeff"].clone()
        }
        lambda_t = setup_constraint["lambda_t"].clone().requires_grad_(True)
        _, _, analytical_grad, _ = c.get_derivatives(lambda_t=lambda_t, **inputs)
        residuals = c.get_residuals(lambda_t=lambda_t, **inputs)
        numerical_grad_rows = []
        for i in range(setup_constraint["body_count"]):
            grad_rows_per_body = []
            for j in range(3 * setup_constraint["max_contacts"]):
                grad_output = torch.zeros_like(residuals)
                grad_output[i, j, 0] = 1.0
                grad = torch.autograd.grad(residuals, lambda_t, grad_outputs=grad_output, create_graph=True)[0][i]
                grad_rows_per_body.append(grad.squeeze(-1))
            numerical_grad_rows.append(torch.stack(grad_rows_per_body))
        numerical_grad = torch.stack(numerical_grad_rows)

        # Visualize differences
        # visualize_jacobian_difference(analytical_grad, numerical_grad, "Jacobian w.r.t. lambda_t", "lambda_t_jacobian")

        assert analytical_grad.shape == (setup_constraint["body_count"], 3 * setup_constraint["max_contacts"], 2 * setup_constraint["max_contacts"])
        assert numerical_grad.shape == analytical_grad.shape
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-6)

    def test_gradient_gamma(self, setup_constraint):
        c = setup_constraint["constraint"]
        inputs = {
            "body_vel": setup_constraint["body_vel"].clone(),
            "lambda_n": setup_constraint["lambda_n"].clone(),
            "lambda_t": setup_constraint["lambda_t"].clone(),
            "J_t": setup_constraint["J_t"].clone(),
            "contact_mask": setup_constraint["contact_mask"].clone(),
            "friction_coeff": setup_constraint["friction_coeff"].clone()
        }
        gamma = setup_constraint["gamma"].clone().requires_grad_(True)
        _, _, _, analytical_grad = c.get_derivatives(gamma=gamma, **inputs)
        residuals = c.get_residuals(gamma=gamma, **inputs)
        numerical_grad_rows = []
        for i in range(setup_constraint["body_count"]):
            grad_rows_per_body = []
            for j in range(3 * setup_constraint["max_contacts"]):
                grad_output = torch.zeros_like(residuals)
                grad_output[i, j, 0] = 1.0
                grad = torch.autograd.grad(residuals, gamma, grad_outputs=grad_output, create_graph=True)[0][i]
                grad_rows_per_body.append(grad.squeeze(-1))
            numerical_grad_rows.append(torch.stack(grad_rows_per_body))
        numerical_grad = torch.stack(numerical_grad_rows)

        # Visualize differences
        # visualize_jacobian_difference(analytical_grad, numerical_grad, "Jacobian w.r.t. gamma", "gamma_jacobian")

        assert analytical_grad.shape == (setup_constraint["body_count"], 3 * setup_constraint["max_contacts"], setup_constraint["max_contacts"])
        assert numerical_grad.shape == analytical_grad.shape
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-6)


class TestRevoluteConstraint:
    @pytest.fixture
    def setup_constraint(self):
        device = torch.device("cpu")
        body_count = 2
        joint_count = 1
        dt = 0.01
        torch.random.manual_seed(0)

        # Mock Model for joint data
        class MockModel:
            def __init__(self):
                self.joint_parent = torch.tensor([0], dtype=torch.int32, device=device)
                self.joint_child = torch.tensor([1], dtype=torch.int32, device=device)
                self.joint_X_p = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device=device).view(1, 7, 1)
                self.joint_X_c = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device=device).view(1, 7, 1)
                self.body_q = torch.randn(body_count, 7, 1, device=device)
                self.body_q[:, 3:] = normalize_quat_batch(self.body_q[:, 3:])

        model = MockModel()
        constraint = RevoluteConstraint(model)

        body_vel = torch.randn(body_count, 6, 1, device=device)
        body_trans = model.body_q.clone()
        lambda_j = torch.randn(5 * joint_count, 1, device=device)

        J_j_p, J_j_c = constraint.compute_jacobians(body_trans)

        return {
            "constraint": constraint,
            "body_vel": body_vel,
            "body_trans": body_trans,
            "lambda_j": lambda_j,
            "J_j_p": J_j_p,
            "J_j_c": J_j_c,
            "dt": dt,
            "body_count": body_count,
            "joint_count": joint_count
        }

    def test_gradient_body_vel(self, setup_constraint):
        """Test derivative with respect to body_vel"""
        c = setup_constraint["constraint"]
        body_vel = setup_constraint["body_vel"].clone().requires_grad_(True)
        body_trans = setup_constraint["body_trans"].clone()
        J_j_p = setup_constraint["J_j_p"].clone()
        J_j_c = setup_constraint["J_j_c"].clone()
        lambda_j = setup_constraint["lambda_j"].clone()
        dt = setup_constraint["dt"]

        # Analytical gradient
        with torch.no_grad():
            analytical_grad, _ = c.get_derivatives(body_vel=body_vel,
                                                body_trans=body_trans,
                                                J_j_p=J_j_p,
                                                J_j_c=J_j_c,
                                                dt=dt)  # [5 * D, 6 * B]

        # Numerical gradient
        res = c.get_residuals(body_vel=body_vel,
                              lambda_j=lambda_j,
                              body_trans=body_trans,
                              J_j_p=J_j_p,
                              J_j_c=J_j_c,
                              dt=dt)  # [B, 6, 1]

        numerical_grad = torch.zeros_like(analytical_grad)
        for i in range(5 * setup_constraint["joint_count"]):
            grad_output = torch.zeros_like(res)
            grad_output[i, 0] = 1.0
            grad = torch.autograd.grad(res, body_vel, grad_outputs=grad_output, create_graph=True)[0]  # [B, 6, 1]
            numerical_grad[i] = grad.flatten()

        # Compare
        assert analytical_grad.shape == (5 * setup_constraint["joint_count"], 6 * setup_constraint["body_count"])
        assert numerical_grad.shape == analytical_grad.shape
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-6)


    def test_revolute_joint_behavior(self, setup_constraint):
        lambda_j = torch.zeros( 5 * setup_constraint["joint_count"], 1, device=setup_constraint["constraint"].device)

        dt = 1.0
        c = setup_constraint["constraint"]
        body_trans = torch.zeros(setup_constraint["body_count"], 7, 1)
        body_trans[0, :3] = torch.tensor([0.0, 0.0, 0.0]).view(3, 1)  # Parent at origin
        body_trans[0, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(4, 1)  # Identity quaternion
        body_trans[1, :3] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1)  # Child at (1, 0, 0)
        body_trans[1, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(4, 1)  # Identity quaternion
        body_trans[:, 3:] = normalize_quat_batch(body_trans[:, 3:])

        # Get constraint Jacobians
        J_j_p, J_j_c = c.compute_jacobians(body_trans)

        tests = [
            ( # X-axis rotation (should be constrained),
                False,
                torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=c.device).view(6, 1),
                torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=c.device).view(6, 1)
            ),
            ( # Y-axis rotation (should be constrained)
                False,
                torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=c.device).view(6, 1),
                torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], device=c.device).view(6, 1)
            ),
            ( # Z-axis rotation (should be allowed)
                True,
             torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=c.device).view(6, 1),
             torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=c.device).view(6, 1)
            ),
            (  # X-axis translation (should be constrained)
                False,
                torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=c.device).view(6, 1),
                torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=c.device).view(6, 1)
            ),
            (  # Y-axis translation (should be constrained)
                False,
                torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=c.device).view(6, 1),
                torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=c.device).view(6, 1)
            ),
            (  # Z-axis translation (should be allowed)
                False,
                torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=c.device).view(6, 1),
                torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], device=c.device).view(6, 1)
            )

        ]

        for allowed, v_p, v_c in tests:
            body_vel = torch.stack([v_p, v_c], dim=0)  # [2, 6, 1]
            res_j = c.get_residuals(body_vel, lambda_j, body_trans, J_j_p, J_j_c, dt)  # [B, 6, 1]

            if allowed:
                assert torch.norm(res_j).item() < 1e-6, f"Residual not zero for allowed motion: {res_j}"
            else:
                assert torch.norm(res_j).item() > 1e-6, f"Residual should be non-zero for constrained motion: {res_j}"

    def test_jacobian_reshaping(self, setup_constraint):
        """Test whether the Jacobian reshaping in DynamicsConstraint works correctly."""
        c = setup_constraint["constraint"]
        body_trans = setup_constraint["body_trans"].clone()

        # Compute the Jacobians
        J_j_p, J_j_c = c.compute_jacobians(body_trans)
        D = c.joint_parent.shape[0]  # Number of joints

        # Test reshaping - this should work if dimensions are consistent
        try:
            J_j_p_batch = J_j_p.view(D, 5, 6)
            J_j_c_batch = J_j_c.view(D, 5, 6)

            # Test calculation with parent body velocity
            parent_idx = c.joint_parent[0]
            v_p = setup_constraint["body_vel"][parent_idx]  # [6, 1]

            # Method 1: Using original shape
            result1 = torch.matmul(J_j_p, v_p)  # [5*D, 1]

            # Method 2: Using reshaped Jacobians
            result2_batch = torch.matmul(J_j_p_batch, v_p)  # [D, 5, 1]
            result2 = result2_batch.view(-1, 1)  # [5*D, 1]

            # Compare results
            assert torch.allclose(result1, result2, atol=1e-6), "Reshaping changes calculation results"

        except RuntimeError as e:
            print(f"Reshaping failed: {e}")
            print(f"J_j_p shape: {J_j_p.shape}, expected: [{D}, 5, 6]")
            print(f"Expected elements: {D * 5 * 6}, actual: {J_j_p.numel()}")

if __name__ == "__main__":
    pytest.main([__file__])
