import Imath  # For OpenEXR data types
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
    # Make sure the array is writable by creating a copy
    data = np.copy(data)
    return torch.from_numpy(data)


def load_terrain_from_exr_file(heightmap_path: str, size_x: float, size_y: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
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
    num_points_y, num_points_x = heightmap.shape
    res_x = size_x / num_points_x  # Resolution in meters per pixel (x-direction)
    res_y = size_y / num_points_y  # Resolution in meters per pixel (y-direction)

    # Create X and Y grids (assuming zero-centered coordinates)
    x = torch.linspace(-size_x / 2, size_x / 2, num_points_x)
    y = torch.linspace(-size_y / 2, size_y / 2, num_points_y)
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
