# Differentiable Position-Based Dynamics

- [ ]  Create tests for the code
- [ ]  Document the code
- [ ]  Add demos for the correction methods
- [ ]  Visualizer nicer css style
- [ ]  Comparison between

## JSON Format

To use the visualization tool, you need to provide a JSON representation of the simulation state. This JSON can be generated using the serialize methods in the Model and State classes. Below is the structure of the expected JSON format.

### Model JSON Format

```json
{
    "body_count": int,                        // Total number of bodies in the model
    "joint_count": int,                       // Total number of joints in the model
    "flat_ground": bool,                      // Whether to display the ground as a flat plane
    "ground_points": [                       // Array of 3D points forming the heightmap grid
        {
          "x": float,                   // x-coordinate
          "y": float,                   // y-coordinate
          "z": float                    // Height at this point
        },
        // ... more points
      ],
    "ground_normals": [                      // Optional: Array of normal vectors for each point
        {
          "x": float,                   // x-component of normal
          "y": float,                   // y-component of normal
          "z": float                    // z-component of normal
        },
        // ... more normals (same length as points array)
    ],
    "bodies": [                               // List of body configurations
      {
            "name": str,                        // Name of the body
            "shape": {                          // Shape of the body (Box, Sphere, or Cylinder)
              "type": int,                        // 0: Box, 1: Sphere, 2: Cylinder
              "hx": float,                        // Box: half-length in x (only for Box)
              "hy": float,                        // Box: half-length in y (only for Box)
              "hz": float,                        // Box: half-length in z (only for Box)
              "radius": float,                    // Sphere/Cylinder: radius (only for Sphere/Cylinder)
              "height": float                     // Cylinder: height (only for Cylinder)
          },
        },
    ...
  ],
}
```

### State JSON Format

```json
{
  "time": float,                            // Current simulation time
  "bodies": [                               // List of body states
    {
      "name": str,                          // Name of the body
      "q": [float, ...],                    // Position and orientation (quaternion) [x, y, z, w, i, j, k]
      "qd": [float, ...],                   // Linear and angular velocity [vx, vy, vz, wx, wy, wz]
      "f": [float, ...],                    // Forces and torques [fx, fy, fz, tx, ty, tz]
      "contacts": [int, ...]                // Indices of contact points (if applicable)
    },
    ...
  ],
}
```


To perform a Newton root-finding optimization for simulating the system as outlined in your query, we’ll follow the four steps you’ve specified: integrating the generalized positions and velocities, computing the residual function \( F(x) \), calculating the Jacobian, and performing the Newton step. Below, I’ll provide a detailed explanation and a complete implementation based on the provided documents and your requirements.

---

### Step 1: Integrate Generalized Positions and Velocities

We start by updating the generalized positions (\( \text{body\_q} \)) and velocities (\( \text{body\_qd} \)) using a semi-implicit Euler integrator. This method computes the new positions and velocities based on the current state, external forces, and time step.

- **Inputs:**
  - \( \text{body\_q} \): Current positions [body_count, 7]
  - \( \text{body\_qd} \): Current velocities [body_count, 6]
  - \( \text{body\_f} \): External forces [body_count, 6]
  - \( \text{body\_inv\_mass} \): Inverse masses [body_count]
  - \( \text{body\_inv\_inertia} \): Inverse inertia tensors [body_count, 3, 3]
  - \( \text{gravity} \): Gravity vector [3]
  - \( \text{dt} \): Time step

- **Outputs:**
  - \( \text{new\_body\_q} \): Updated positions
  - \( \text{new\_body\_qd} \): Updated velocities

The semi-implicit Euler integrator handles this integration, and we’ll assume it’s correctly implemented in the `SemiImplicitEulerIntegrator` class.

---

### Step 2: Compute the Residual Function \( F(x) \)

The residual function \( F(x) \) has two components, where \( x = [\text{body\_qd}, \lambda_n] \):
- \( \text{body\_qd} \): Velocities [body_count, 6]
- \( \lambda_n \): Normal impulses (constraint forces) [body_count, max_contacts]

#### Part a: Newton-Euler Equations

The Newton-Euler equations balance the forces and accelerations:
\[ M \cdot a - J_n \cdot \lambda_n - \text{body\_f} = 0 \]
Where:
- \( M \): Mass matrix [body_count, 6, 6]
- \( a \): Acceleration, computed as \( a = \frac{\text{new\_body\_qd} - \text{body\_qd}}{\text{dt}} \) [body_count, 6]
- \( J_n \): Contact Jacobian [body_count, max_contacts, 6]
- \( \lambda_n \): Normal impulses [body_count, max_contacts]
- \( \text{body\_f} \): External forces [body_count, 6]

The residual for this part is:
\[ \text{res\_1} = M \cdot a - J_n \cdot \lambda_n - \text{body\_f} \]

#### Part b: Complementary Condition

The complementary condition ensures that either the normal impulse (\( \lambda_n \)) or the relative velocity at the contact point (\( J_n \cdot \text{body\_qd} \)) is zero, while both are non-negative:
\[ \lambda_n \geq 0, \quad J_n \cdot \text{body\_qd} \geq 0, \quad \lambda_n \cdot (J_n \cdot \text{body\_qd}) = 0 \]

This is approximated using the Fisher-Burmeister function:
\[ \phi(\lambda_n, J_n \cdot \text{body\_qd}) = \lambda_n + (J_n \cdot \text{body\_qd}) - \sqrt{\lambda_n^2 + (J_n \cdot \text{body\_qd})^2 + \epsilon} \]
Where \( \epsilon \) (e.g., 1e-6) ensures numerical stability. The residual for this part is:
\[ \text{res\_2} = \phi(\lambda_n, J_n \cdot \text{body\_qd}) \]

The full residual is:
\[ F(x) = \begin{bmatrix} \text{res\_1} \\ \text{res\_2} \end{bmatrix} \]

---

### Step 3: Compute the Jacobian for the Newton Step

The Jacobian \( J_F = \frac{\partial F}{\partial x} \) is needed for the Newton iteration, where \( x = [\text{body\_qd}, \lambda_n] \). It’s a block matrix based on the derivatives of \( F(x) \).

#### Derivatives of Part a (Newton-Euler):
- **With respect to \( \text{body\_qd} \):**
  \[ \frac{\partial}{\partial \text{body\_qd}} (M \cdot a) = M \cdot \frac{\partial a}{\partial \text{body\_qd}} = M \cdot \frac{1}{\text{dt}} \]
  Since \( a = \frac{\text{new\_body\_qd} - \text{body\_qd}}{\text{dt}} \), and \( \text{body\_qd} \) is the previous velocity (constant here).

- **With respect to \( \lambda_n \):**
  \[ \frac{\partial}{\partial \lambda_n} (-J_n \cdot \lambda_n) = -J_n \]

#### Derivatives of Part b (Fisher-Burmeister):
Let \( a = \lambda_n \), \( b = J_n \cdot \text{body\_qd} \).
- **With respect to \( \text{body\_qd} \):**
  \[ \frac{\partial \phi}{\partial \text{body\_qd}} = \frac{\partial \phi}{\partial b} \cdot \frac{\partial b}{\partial \text{body\_qd}} = \frac{\partial \phi}{\partial b} \cdot J_n \]
  Where \( \frac{\partial \phi}{\partial b} = 1 - \frac{b}{\sqrt{a^2 + b^2 + \epsilon}} \).

- **With respect to \( \lambda_n \):**
  \[ \frac{\partial \phi}{\partial \lambda_n} = \frac{\partial \phi}{\partial a} = 1 - \frac{a}{\sqrt{a^2 + b^2 + \epsilon}} \]

The Jacobian is:
\[ J_F = \begin{bmatrix}
\frac{M}{\text{dt}} & -J_n \\
\frac{\partial \phi}{\partial b} \cdot J_n & \frac{\partial \phi}{\partial a}
\end{bmatrix} \]

---

### Step 4: Perform the Newton Step

The Newton update is:
\[ x_{\text{new}} = x - J_F^{-1} \cdot F(x) \]
In practice, we solve:
\[ J_F \cdot \Delta x = -F(x) \]
Then update:
\[ x \leftarrow x + \Delta x \]
Where \( \Delta x = [\Delta \text{body\_qd}, \Delta \lambda_n] \).

---

### Implementation

Here’s how to implement this in Python using PyTorch, based on the provided classes:

```python
import torch
from pbd_torch.constraints import compute_contact_jacobian
from pbd_torch.integrator import SemiImplicitEulerIntegrator
from pbd_torch.model import Model, State, Control
from pbd_torch.ncp import FisherBurmeister

class NonSmoothNewtonEngine:
    def __init__(self, iterations: int = 10, device: torch.device = torch.device("cpu")):
        self.device = device
        self.iterations = iterations
        self.integrator = SemiImplicitEulerIntegrator(use_local_omega=True, device=device)
        self.fb = FisherBurmeister(epsilon=1e-6)

    def simulate(self, model: Model, state_in: State, state_out: State, control: Control, dt: float):
        max_contacts = state_in.contact_points.shape[1]
        body_count = model.body_count

        # Step 1: Integrate positions and velocities
        new_body_q, new_body_qd = self.integrator.integrate(
            body_q=state_in.body_q,
            body_qd=state_in.body_qd,
            body_f=state_in.body_f,
            body_inv_mass=model.body_inv_mass,
            body_inv_inertia=model.body_inv_inertia,
            gravity=model.gravity,
            dt=dt,
        )

        # Initialize lambda_n
        lambda_n = torch.zeros((body_count, max_contacts), device=self.device)

        for _ in range(self.iterations):
            # Compute mass matrix and contact Jacobian
            M = model.mass_matrix  # [body_count, 6, 6]
            J_n = compute_contact_jacobian(state_in)  # [body_count, max_contacts, 6]

            # Compute acceleration
            a = (new_body_qd - state_in.body_qd) / dt  # [body_count, 6]

            # Compute J_n @ body_qd
            J_n_dot_qd = torch.bmm(J_n, new_body_qd.unsqueeze(-1)).squeeze(-1)  # [body_count, max_contacts]

            # Step 2: Compute F(x)
            # Newton-Euler residual
            res_1 = torch.bmm(M, a.unsqueeze(-1)).squeeze(-1) - \
                    torch.bmm(J_n.transpose(1, 2), lambda_n.unsqueeze(-1)).squeeze(-1) - \
                    state_in.body_f  # [body_count, 6]
            # Fisher-Burmeister residual
            res_2 = self.fb.evaluate(lambda_n, J_n_dot_qd)  # [body_count, max_contacts]
            F_x = torch.cat((res_1, res_2), dim=1)  # [body_count, 6 + max_contacts]

            # Step 3: Compute Jacobian
            da, db = self.fb.derivatives(lambda_n, J_n_dot_qd)  # [body_count, max_contacts]

            # Assemble Jacobian per body (for simplicity, we’ll do it in a loop)
            for b in range(body_count):
                M_b = M[b]  # [6, 6]
                J_n_b = J_n[b]  # [max_contacts, 6]
                da_b = da[b]  # [max_contacts]
                db_b = db[b]  # [max_contacts]

                # Top-left: M/dt
                top_left = M_b / dt  # [6, 6]
                # Top-right: -J_n
                top_right = -J_n_b.transpose(0, 1)  # [6, max_contacts]
                # Bottom-left: db * J_n
                bottom_left = (db_b.unsqueeze(-1) * J_n_b).transpose(0, 1)  # [max_contacts, 6]
                # Bottom-right: diag(da)
                bottom_right = torch.diag(da_b)  # [max_contacts, max_contacts]

                J_F_b = torch.cat(
                    [
                        torch.cat([top_left, top_right], dim=1),
                        torch.cat([bottom_left, bottom_right], dim=1),
                    ],
                    dim=0,
                )  # [6 + max_contacts, 6 + max_contacts]

                # Step 4: Newton step
                delta_x_b = torch.linalg.solve(J_F_b, -F_x[b])
                new_body_qd[b] += delta_x_b[:6]
                lambda_n[b] += delta_x_b[6:]

        # Update output state
        state_out.body_q = new_body_q
        state_out.body_qd = new_body_qd
```

---

### Explanation of the Code

1. **Integration:**
   - The `integrator.integrate` method computes initial \( \text{new\_body\_q} \) and \( \text{new\_body\_qd} \).

2. **Residual \( F(x) \):**
   - `res_1`: Computes the Newton-Euler residual using matrix multiplications.
   - `res_2`: Uses the `FisherBurmeister.evaluate` method to compute the complementarity residual.

3. **Jacobian:**
   - Derivatives are computed using `FisherBurmeister.derivatives`.
   - The Jacobian is assembled block-wise for each body to handle tensor dimensions correctly.

4. **Newton Step:**
   - Solves the linear system using `torch.linalg.solve` and updates \( \text{body\_qd} \) and \( \lambda_n \).

This implementation iterates the Newton steps multiple times (controlled by `iterations`) to refine the solution, ensuring the system satisfies both the Newton-Euler equations and the complementarity conditions. Adjust `iterations` based on convergence needs.
