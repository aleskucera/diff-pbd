import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pbd_torch.environment import generate_heightmaps, make_x_y_grids, compute_heightmap_gradients
from pbd_torch.heightmap_generators import MultiGaussianHeightmapGenerator
import torch

import matplotlib
matplotlib.use('TkAgg')

def main():
    grid_res = 0.05  # [m] 5cm per grid cell
    max_coord = 3.2  # [m]
    heighmap_gen = MultiGaussianHeightmapGenerator(
        min_gaussians=400,
        max_gaussians=600,
        min_height_fraction=0.03,
        max_height_fraction=0.12,
        min_std_fraction=0.03,
        max_std_fraction=0.08,
        min_sigma_ratio=0.6,
    )
    x_grid, y_grid = make_x_y_grids(max_coord, grid_res, 1)
    z_grid, suit_mask = generate_heightmaps(x_grid, y_grid, heighmap_gen)

    # Print shapes
    print("x_grid shape:", x_grid.shape)  # (1, 128, 128)
    print("y_grid shape:", y_grid.shape)  # (1, 128, 128)
    print("z_grid shape:", z_grid.shape)  # (1, 128, 128)
    print("suit_mask shape:", suit_mask.shape)  # (1, 128, 128)

    # Compute gradients
    z_grid_grads = compute_heightmap_gradients(z_grid, grid_res)  # (1, 2, 128, 128)

    # Compute normals
    ones = torch.ones_like(z_grid_grads[:, 0]).unsqueeze(1)  # (1, 1, 128, 128)
    normals = torch.cat((-z_grid_grads, ones), dim=1)  # (1, 3, 128, 128)
    normals /= torch.linalg.norm(normals, dim=1, keepdim=True)  # Normalize

    # Visualize the heightmap and normals
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = x_grid.squeeze(), y_grid.squeeze()
    Z = z_grid.squeeze()
    ax.plot_surface(X, Y, Z, cmap='terrain', edgecolor='none', alpha=0.8)

    # Plot normals as arrows
    downsample_factor = 10  # Downsample for better visualization
    X_downsampled = X[::downsample_factor, ::downsample_factor]
    Y_downsampled = Y[::downsample_factor, ::downsample_factor]
    Z_downsampled = Z[::downsample_factor, ::downsample_factor]
    normals_downsampled = normals[0, :, ::downsample_factor, ::downsample_factor].permute(1, 2, 0)

    for i in range(X_downsampled.shape[0]):
        for j in range(X_downsampled.shape[1]):
            ax.quiver(
                X_downsampled[i, j], Y_downsampled[i, j], Z_downsampled[i, j],
                normals_downsampled[i, j, 0], normals_downsampled[i, j, 1], normals_downsampled[i, j, 2],
                length=0.1, color='red', normalize=True
            )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Height (m)')

    # Compute limits
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    z_min, z_max = Z.min(), Z.max()

    # Make the plot visible with equal axis scaling
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5

    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

    plt.show()

if __name__ == "__main__":
    main()

