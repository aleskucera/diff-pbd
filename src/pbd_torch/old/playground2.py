import torch
from pbd_torch.model import Model, State, Control
from pbd_torch.constraints import RevoluteConstraint
from pbd_torch.transform import normalize_quat_batch
from newton_engine import NonSmoothNewtonEngine  # Adjust import path as needed
from pbd_torch.constants import *

device = torch.device("cpu")
dt = 1.0
B = 2  # Two bodies
D = 1  # One joint
C = 0  # No contacts

# Mock Model
class MockModel(Model):
    def __init__(self):
        super().__init__(device=device)
        self.body_q = torch.zeros(B, 7, 1, device=device)  # 7D state (3 pos + 4 quat)
        self.joint_parent = torch.tensor([0], dtype=torch.long, device=device)  # Body 0 is parent
        self.joint_child = torch.tensor([1], dtype=torch.long, device=device)   # Body 1 is child
        self.joint_X_p = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device=device).view(D, 7, 1)
        self.joint_X_c = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device=device).view(D, 7, 1)
        self.body_mass = torch.ones(B, 1, 1, device=device) * 10.0
        self.body_inv_mass = torch.ones(B, 1, 1, device=device) * 0.1
        self.body_inertia = torch.ones(B, 3, 3, device=device) * 10.0
        self.body_inv_inertia = torch.ones(B, 3, 3, device=device) * 0.1
        self.g_accel = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device).view(6, 1)
        self.max_contacts_per_body = C  # Explicitly zero contacts
        self.restitution = torch.zeros(B, 1, device=device)
        self.dynamic_friction = torch.zeros(B, 1, device=device)

# Initialize state
model = MockModel()
state = model.state()  # Use model's state constructor
state.body_q = torch.zeros(B, 7, 1, device=device)
state.body_q[0, :3] = torch.tensor([0.0, 0.0, 0.0], device=device).view(3, 1)  # Parent at origin
state.body_q[0, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).view(4, 1)  # Identity quaternion
state.body_q[1, :3] = torch.tensor([1.0, 0.0, 0.0], device=device).view(3, 1)  # Child at (1, 0, 0)
state.body_q[1, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).view(4, 1)  # Identity quaternion
state.body_q[:, 3:] = normalize_quat_batch(state.body_q[:, 3:])
state.body_qd = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device).view(B, 6, 1)
state.body_f = torch.zeros(B, 6, 1, device=device)  # No external forces
state.contact_points = torch.empty(B, C, 3, 1, device=device)  # Empty tensor for no contacts
state.contact_normals = torch.empty(B, C, 3, 1, device=device)
state.contact_points_ground = torch.empty(B, C, 3, 1, device=device)
state.contact_mask = torch.zeros(B, C, device=device, dtype=torch.bool)
state.time = 0.0

# Initialize control
control = model.control()
control.body_f = torch.zeros(B, 6, 1, device=device)
control.contact_points = torch.empty(B, C, 3, 1, device=device)
control.contact_normals = torch.empty(B, C, 3, 1, device=device)
control.contact_points_ground = torch.empty(B, C, 3, 1, device=device)
control.contact_mask = torch.zeros(B, C, device=device, dtype=torch.bool)

# Initialize solver
engine = NonSmoothNewtonEngine(model, iterations=10, device=device)
state_out = model.state()
state_out.body_q = state.body_q.clone()
state_out.body_qd = state.body_qd.clone()
state_out.body_f = state.body_f.clone()
state_out.contact_points = state.contact_points.clone()
state_out.contact_normals = state.contact_normals.clone()
state_out.contact_points_ground = state.contact_points_ground.clone()
state_out.contact_mask = state.contact_mask.clone()
state_out.time = 0.0

# Compute initial Jacobians for diagnostics
revolute = RevoluteConstraint(model)
J_j_p, J_j_c = revolute.compute_jacobians(state.body_q)
print(f"J_j_p: {J_j_p}")
print(f"J_j_c: {J_j_c}")

# Test case 1: Translation only (should be constrained)
v_parent = torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], device=device).view(6, 1)
v_child = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device).view(6, 1)
v_rel = torch.matmul(J_j_p, v_parent) + torch.matmul(J_j_c, v_child)
print("Test 1 - Translation (expect near zero):", v_rel.squeeze().tolist())

# Test case 2: Allowed rotation (z-axis, should be non-zero)
v_parent = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=device).view(6, 1)
v_child = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=device).view(6, 1)
v_rel = torch.matmul(J_j_p, v_parent) + torch.matmul(J_j_c, v_child)
print("Test 2 - Z-rotation (expect non-zero):", v_rel.squeeze().tolist())

# Simulate
num_steps = 20
for step in range(num_steps):
    prev_body_q = state.body_q.clone()
    prev_body_qd = state.body_qd.clone()

    # Update body_trans and run simulation
    engine._body_trans = state.body_q.clone()
    engine.simulate(state, state_out, control, dt)

    # Update state for next iteration
    state = state_out
    state_out = model.state()
    state_out.body_q = state.body_q.clone()
    state_out.body_qd = state.body_qd.clone()
    state_out.body_f = state.body_f.clone()
    state_out.contact_points = state.contact_points.clone()
    state_out.contact_normals = state.contact_normals.clone()
    state_out.contact_points_ground = state.contact_points_ground.clone()
    state_out.contact_mask = state.contact_mask.clone()
    state_out.time = state.time

    # Diagnostics
    v_j_p = torch.matmul(J_j_p.view(D, 5, 6), state.body_qd[0]).view(5 * D, 1)
    v_j_c = torch.matmul(J_j_c.view(D, 5, 6), state.body_qd[1]).view(5 * D, 1)
    v_rel = v_j_p + v_j_c
    constraint_violation = torch.norm(v_rel).item()

    residual_norm = torch.sum(engine._debug_F_x[-1] ** 2).item() if engine._debug_F_x else float('nan')

    joint_pos_p = state.body_q[0, :3] + torch.tensor([1.0, 0.0, 0.0], device=device).view(3, 1)
    joint_pos_c = state.body_q[1, :3]
    X_p, X_c, _, _ = revolute._get_joint_frames(state.body_q)
    drift = torch.norm(X_p[0, :3] - X_c[0, :3]).item()

    vel_norm = torch.norm(state.body_qd).item()

    print(f"Step {step}:")
    print(f"  Constraint violation norm: {constraint_violation:.6f}")
    print(f"  Residual norm: {residual_norm:.6f}")
    print(f"  Joint position drift: {drift:.6f}")
    print(f"  Velocity norm: {vel_norm:.6f}")
    print(f"  Parent pos: {state.body_q[0, :3].squeeze().tolist()}")
    print(f"  Parent rot: {state.body_q[0, 3:].squeeze().tolist()}")
    print(f"  Child pos: {state.body_q[1, :3].squeeze().tolist()}")
    print(f"  Child rot: {state.body_q[1, 3:].squeeze().tolist()}")
    print(f"  Parent vel: {state.body_qd[0, 3:].squeeze().tolist()}")
    print(f"  Child vel: {state.body_qd[1, 3:].squeeze().tolist()}")