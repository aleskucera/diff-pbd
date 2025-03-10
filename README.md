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
