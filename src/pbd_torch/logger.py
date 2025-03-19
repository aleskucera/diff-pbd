import os
from typing import Any

import numpy as np
import torch

os.environ["DEBUG"] = "true"


class DebugLogger:
    def __init__(self):
        self.debug_enabled = os.getenv("DEBUG", "false").lower() == "true"
        self.indent_level = 0
        self.indent_str = "    "
        self.section_width = 80  # Total width of section header
        self.subsection_width = 60  # Total width of subsection header

    def _process_tensor(self, tensor: torch.Tensor) -> Any:
        """Process tensor for prettier printing."""
        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.detach().cpu().numpy()
            return np.round(tensor_np, 4)
        return tensor

    def _format_value(self, value: Any) -> str:
        """Format different types of values."""
        if isinstance(value, torch.Tensor):
            return str(self._process_tensor(value))
        elif isinstance(value, (list, tuple)):
            return str([self._format_value(v) for v in value])
        elif isinstance(value, dict):
            return str({k: self._format_value(v) for k, v in value.items()})
        return str(value)

    def _create_centered_header(self, title: str, width: int, fill_char: str) -> str:
        """Create a centered header with fixed width."""
        if len(title) > width - 4:  # Leave at least 2 fill_char on each side
            title = title[: (width - 7)] + "..."

        total_fill = width - len(title) - 2  # -2 for spaces around title
        left_fill = total_fill // 2
        right_fill = total_fill - left_fill

        return f"{fill_char * left_fill} {title} {fill_char * right_fill}"

    def print(self, *args, **kwargs):
        if not self.debug_enabled:
            return
        indent = self.indent_str * self.indent_level
        processed_args = [self._format_value(arg) for arg in args]
        print(indent + " ".join(map(str, processed_args)), **kwargs)

    def section(self, title: str):
        if not self.debug_enabled:
            return
        header = self._create_centered_header(title, self.section_width, "=")
        print(f"\n{header}\n")

    def subsection(self, title: str):
        if not self.debug_enabled:
            return
        header = self._create_centered_header(title, self.subsection_width, "-")
        print(f"\n{header}\n")

    def indent(self):
        self.indent_level += 1

    def undent(self):
        self.indent_level = max(0, self.indent_level - 1)


# Create global instance
debug = DebugLogger()


# Example usage
def demo_debug_printer():
    # Create some sample tensors
    pos = torch.tensor([1.23456, 2.34567, 3.45678])
    vel = torch.tensor([0.12345, -0.23456, 0.34567])
    matrix = torch.randn(2, 3)

    # Dictionary with mixed types
    state = {"position": pos, "velocity": vel, "name": "object_1", "active": True}

    # Basic printing
    debug.section("BASIC PRINTING")
    debug.print("Position:", pos)
    debug.print("Velocity:", vel)
    debug.print("Matrix:", matrix)

    # Nested sections
    debug.section("NESTED SECTIONS")
    debug.print("Starting simulation...")

    debug.subsection("PHYSICS UPDATE")
    debug.indent()
    debug.print("Computing forces...")
    debug.print("State:", state)

    debug.subsection("COLLISION DETECTION")
    debug.print("Checking collisions...")
    debug.indent()
    debug.print("Object 1 vs Object 2")
    debug.print("Collision points:", torch.randn(3, 3))
    debug.undent()

    # Lists and dictionaries
    debug.section("COMPLEX DATA STRUCTURES")
    debug.print("List of tensors:", [pos, vel])
    debug.print(
        "Nested dict:",
        {"obj1": {"pos": pos, "vel": vel}, "obj2": {"pos": -pos, "vel": -vel}},
    )


if __name__ == "__main__":
    demo_debug_printer()
