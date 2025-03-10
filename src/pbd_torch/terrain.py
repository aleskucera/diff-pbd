from dataclasses import dataclass
from dataclasses import field

import Imath  # For OpenEXR data types
import numpy as np
import OpenEXR  # For loading .exr files
import torch


@dataclass
class Terrain:
    """
    Represents a terrain heightmap with associated properties.
    
    Attributes:
        size_x: Size of terrain in x direction (meters)
        size_y: Size of terrain in y direction (meters)
        height_data: 2D tensor of height values
        normals: Optional pre-computed normal vectors
        max_coord: Maximum coordinate value for interpolation (defaults to max of size_x/2, size_y/2)
    """
    size_x: float
    size_y: float
    height_data: torch.Tensor
    normals: torch.Tensor = None
    max_coord: float = None
    
    # Bounds are computed during initialization
    bounds: dict = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate inputs
        if len(self.height_data.shape) != 2:
            raise ValueError("Height data must be a 2D tensor")
            
        # Store dimensions for convenience
        self.resolution_y, self.resolution_x = self.height_data.shape
        
        # Set max_coord if not provided
        if self.max_coord is None:
            self.max_coord = max(self.size_x/2, self.size_y/2)
            
        # Compute bounds
        self.bounds = {
            "minZ": float(self.height_data.min()),
            "maxZ": float(self.height_data.max()),
            "minX": -self.size_x/2,
            "maxX": self.size_x/2,
            "minY": -self.size_y/2,
            "maxY": self.size_y/2
        }
        
        # Compute normals if not provided
        if self.normals is None:
            self.compute_normals()
    
    def compute_normals(self):
        """Computes normal vectors for each point in the height field."""
        device = self.height_data.device
        
        # Calculate grid spacings
        dx = self.size_x / (self.resolution_x - 1)  
        dy = self.size_y / (self.resolution_y - 1)
        
        # Compute gradients using torch.gradient
        dz_dy, dz_dx = torch.gradient(self.height_data, spacing=(dy, dx))
        
        # Create normal vectors (-dz_dx, -dz_dy, 1) and normalize
        ones = torch.ones_like(self.height_data)
        normals = torch.stack([-dz_dx, -dz_dy, ones], dim=-1)
        
        # Normalize each normal vector
        norm = torch.norm(normals, dim=-1, keepdim=True)
        self.normals = normals / norm
    
    def get_height_at_point(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Gets interpolated height at given x,y coordinates.
        
        Args:
            x: Tensor of x coordinates
            y: Tensor of y coordinates
            
        Returns:
            Tensor of interpolated heights at requested points
        """
        # Ensure inputs are properly shaped tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.height_data.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.height_data.device)
            
        # Reshape inputs for interpolation
        batch_size = 1
        num_points = x.numel()
        
        # Combine x and y into query points
        query = torch.stack([x.flatten(), y.flatten()], dim=-1).reshape(batch_size, num_points, 2)
        
        # Add batch dimension to height data if needed
        grid = self.height_data.unsqueeze(0) if self.height_data.dim() == 2 else self.height_data
        
        # Use the interpolation function
        heights = interpolate_grid(grid, query, self.max_coord).squeeze()
        
        # Return appropriately shaped output
        return heights.reshape_as(x) if x.numel() > 1 else heights.item()
    
    def get_normal_at_point(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Gets interpolated normal vector at given x,y coordinates.
        
        Args:
            x: Tensor of x coordinates
            y: Tensor of y coordinates
            
        Returns:
            Tensor of interpolated normal vectors at requested points
        """
        # Ensure inputs are properly shaped tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.height_data.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.height_data.device)
            
        # Reshape inputs for interpolation
        batch_size = 1
        num_points = x.numel()
        
        # Combine x and y into query points
        query = torch.stack([x.flatten(), y.flatten()], dim=-1).reshape(batch_size, num_points, 2)
        
        # Add batch dimension to normals and permute to match expected format
        normals_batch = self.normals.unsqueeze(0).permute(0, 3, 1, 2)
        
        # Use the interpolation function
        result_normals = interpolate_normals(normals_batch, query, self.max_coord)
        
        # Return appropriately shaped output
        if x.numel() == 1:
            return result_normals.squeeze()
        else:
            return result_normals.reshape(*x.shape, 3)
    
    def serialize(self):
        """
        Serializes the terrain data for transmission to the client.
        Returns a dictionary representation suitable for JSON.
        """
        # Flatten tensors for JSON serialization
        height_data_flat = self.height_data.flatten().tolist()
        
        # Prepare normals if available
        normals_flat = None
        if self.normals is not None:
            # Reshape to [rows*cols, 3] and flatten to [x1,y1,z1,x2,y2,z2,...]
            normals_flat = self.normals.reshape(-1, 3).flatten().tolist()
        
        return {
            "dimensions": {
                "size_x": float(self.size_x),
                "size_y": float(self.size_y),
                "resolution_x": int(self.resolution_x),
                "resolution_y": int(self.resolution_y)
            },
            "bounds": self.bounds,
            "max_coord": float(self.max_coord),
            "heightData": height_data_flat,
            "normals": normals_flat
        }

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

def create_terrain_from_exr_file(heightmap_path: str, size_x: float, size_y: float) -> Terrain:
    """
    Loads a heightmap file and creates a Terrain object.
    
    Args:
        heightmap_path: Path to the .exr file
        size_x: Size of the ground in the x-direction (meters)
        size_y: Size of the ground in the y-direction (meters)
        
    Returns:
        A Terrain object representing the heightmap
    """
    heightmap = load_exr_file(heightmap_path, 'R')
    
    # Create the Terrain object
    terrain = Terrain(
        size_x=float(size_x),
        size_y=float(size_y),
        height_data=heightmap,
        max_coord=max(size_x/2, size_y/2)  # Set max_coord explicitly
    )
    
    return terrain

def normalized(x, eps=1e-6):
    """
    Normalizes the input tensor.

    Parameters:
    - x: Input tensor.
    - eps: Small value to avoid division by zero.

    Returns:
    - Normalized tensor.
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    norm.clamp_(min=eps)
    return x / norm

def interpolate_normals(normals: torch.Tensor, query: torch.Tensor, max_coord: float) -> torch.Tensor:
    """
    Interpolates the normals at the desired (query[0], query[1]]) coordinates.

    Parameters:
    - normals: Tensor of normals corresponding to the x and y coordinates (3D array), (B, 3, D, D). Top-left corner is (-max_coord, -max_coord). The indexing order follows the "xy" convention, meaning the first dimension is the y-axis and the second dimension is the x-axis.
    - query: Tensor of desired point coordinates for interpolation (3D array), (B, N, 2). Range is from -max_coord to max_coord.
    Returns:
    - Interpolated normals at the queried coordinates in shape (B, N, 3).
    """
    norm_query = query / max_coord  # Normalize to [-1, 1]
    # Clamp the query coordinates to the grid's valid range
    norm_query.clamp_(-1, 1)
    # Query coordinates of shape (B, N, 1, 2)
    grid_coords = norm_query.unsqueeze(2)
    # Interpolate the normals into shape (B, 3, N)
    interpolated_normals = torch.nn.functional.grid_sample(normals, grid_coords, align_corners=True, mode="bilinear").squeeze(3)
    return normalized(interpolated_normals.transpose(1, 2))


def interpolate_grid(grid: torch.Tensor, query: torch.Tensor, max_coord: float | torch.Tensor) -> torch.Tensor:
    """
    Interpolates the height at the desired (query[0], query[1]]) coordinates.

    Parameters:
    - grid: Tensor of grid values corresponding to the x and y coordinates (3D array), (B, D, D). Top-left corner is (-max_coord, -max_coord). The indexing order follows the "xy" convention, meaning the first dimension is the y-axis and the second dimension is the x-axis.
    - query: Tensor of desired point coordinates for interpolation (3D array), (B, N, 2). Range is from -max_coord to max_coord.
    Returns:
    - Interpolated grid values at the queried coordinates in shape (B, N, 1).
    """
    norm_query = query / max_coord  # Normalize to [-1, 1]
    # Clamp the query coordinates to the grid's valid range
    norm_query.clamp_(-1, 1)
    # Query coordinates of shape (B, N, 1, 2)
    grid_coords = norm_query.unsqueeze(2)
    # Grid of shape (B, 1, H, W)
    grid_w_c = grid.unsqueeze(1)
    # Interpolate the grid values into shape (B, 1, N, 1)
    z_query = torch.nn.functional.grid_sample(grid_w_c, grid_coords, align_corners=True, mode="bilinear")
    return z_query.squeeze(1)
