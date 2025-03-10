import Imath  # For OpenEXR data types
import matplotlib.pyplot as plt
import numpy as np
import OpenEXR  # For loading .exr files
import torch


def load_exr_file(file_path: str, channel: str = 'R') -> torch.Tensor:
    """
    Loads a specific channel from an .exr file.

    Args:
        file_path: Path to the .exr file.
        channel: The channel to load (e.g., 'R', 'G', 'B').

    Returns:
        A PyTorch tensor of the channel data.
    """
    # Open the .exr file
    exr_file = OpenEXR.InputFile(file_path)
    # Get the pixel data
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    # Read the specified channel
    channel_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
    # Convert to a numpy array and then to a PyTorch tensor
    data = np.frombuffer(channel_data, dtype=np.float32).reshape(size[1], size[0])
    return torch.from_numpy(data)


def load_ground(heightmap_path: str, size_x: float, size_y: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loads a heightmap from an .exr file and computes its point cloud and normals.

    Args:
        heightmap_path: Path to the .exr file.
        size_x: Size of the ground in the x-direction (meters).
        size_y: Size of the ground in the y-direction (meters).

    Returns:
        A tuple containing:
        - point_cloud: Tensor of shape (H * W, 3) where each row is (x, y, z).
        - normals: Tensor of shape (H * W, 3) where each row is the normal vector at the corresponding point.
    """
    heightmap = load_exr_file(heightmap_path, 'R')
    H, W = heightmap.shape
    res_x = size_x / W  # Resolution in meters per pixel (x-direction)
    res_y = size_y / H  # Resolution in meters per pixel (y-direction)

    # Create X and Y grids (assuming zero-centered coordinates)
    x = torch.linspace(-size_x / 2, size_x / 2, W)
    y = torch.linspace(-size_y / 2, size_y / 2, H)
    X, Y = torch.meshgrid(x, y, indexing="xy")

    # Flatten X, Y, and Z into a point cloud
    point_cloud = torch.stack([X.reshape(-1), Y.reshape(-1), heightmap.reshape(-1)], dim=1)  # Shape: (H * W, 3)

    # Compute gradients using torch.gradient
    dz_dx, dz_dy = torch.gradient(heightmap, spacing=(res_y, res_x), dim=(0, 1))
    # Stack gradients and compute normals
    grad_x = dz_dx.reshape(-1)
    grad_y = dz_dy.reshape(-1)
    normals = torch.stack([-grad_x, -grad_y, torch.ones_like(grad_x)], dim=1)  # Shape: (H * W, 3)
    normals /= torch.linalg.norm(normals, dim=1, keepdim=True)  # Normalize

    return point_cloud, normals


def visualize_point_cloud(point_cloud: torch.Tensor, normals: torch.Tensor, downsample_factor: int = 10) -> None:
    """
    Visualizes a point cloud and its computed normals.

    Args:
        point_cloud: Tensor of shape (N, 3) where each row is (x, y, z).
        normals: Tensor of shape (N, 3) where each row is the normal vector at the corresponding point.
        downsample_factor: Downsample factor for normals visualization.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=point_cloud[:, 2], cmap='terrain', s=1)

    # Downsample normals for visualization
    points_downsampled = point_cloud[::downsample_factor]
    normals_downsampled = normals[::downsample_factor]

    # Plot normals as arrows
    ax.quiver(
        points_downsampled[:, 0], points_downsampled[:, 1], points_downsampled[:, 2],
        normals_downsampled[:, 0], normals_downsampled[:, 1], normals_downsampled[:, 2],
        length=0.1, color='red', normalize=True
    )

    # Set labels and equal axis scaling
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Height (m)')

    # Compute limits for equal axis scaling
    x_min, x_max = point_cloud[:, 0].min(), point_cloud[:, 0].max()
    y_min, y_max = point_cloud[:, 1].min(), point_cloud[:, 1].max()
    z_min, z_max = point_cloud[:, 2].min(), point_cloud[:, 2].max()
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5

    ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
    ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

    plt.show()


def main():
    """
    Main function to load and visualize the point cloud and normals.
    """
    # Load the .exr heightmap file
    file_path = "/home/kuceral4/school/diff-pbd/data/blender/ground1/textures/ant01_heightmap.exr"  # Replace with your .exr file path
    size_x = 2  # Size of the ground in the x-direction (meters)
    size_y = 2  # Size of the ground in the y-direction (meters)

    # Load the point cloud and compute normals
    point_cloud, normals = load_ground(file_path, size_x, size_y)

    # Visualize the point cloud and normals
    visualize_point_cloud(point_cloud, normals, downsample_factor=100)


if __name__ == "__main__":
    main()
