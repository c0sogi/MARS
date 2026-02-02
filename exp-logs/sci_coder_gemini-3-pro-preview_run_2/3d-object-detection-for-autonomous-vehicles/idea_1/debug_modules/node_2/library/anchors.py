import torch
import numpy as np
from library.config import Config


class AnchorGenerator:
    """
    Generates 3D anchors for the Single Shot Detector (SSD) head.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else Config
        self.class_names = self.config.CLASS_NAMES
        self.anchor_sizes = self.config.ANCHOR_SIZES
        self.anchor_rotations = self.config.ANCHOR_ROTATIONS
        self.pc_range = self.config.PC_RANGE

        # Approximate ground height in sensor frame for NuScenes
        # (Lidar is typically ~1.84m above the ground)
        self.ground_z = -1.84

    @property
    def num_anchors_per_location(self):
        """
        Returns the number of anchors generated per spatial grid location.
        """
        return len(self.class_names) * len(self.anchor_rotations)

    def generate(self, feature_map_size, device="cpu"):
        """
        Generates 3D anchors for a given feature map size.

        Args:
            feature_map_size (tuple): Dimensions of the feature map (H, W) or (C, H, W).
            device (str or torch.device): The device to place the anchor tensors on.

        Returns:
            torch.Tensor: A tensor of anchors with shape (H, W, Num_Anchors_Per_Loc, 7).
                          The last dimension contains [x, y, z, w, l, h, yaw].
        """
        # Handle input shape variations
        if len(feature_map_size) == 3:
            _, H, W = feature_map_size
        else:
            H, W = feature_map_size

        # Unpack Point Cloud Range
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range

        # Calculate Stride
        x_span = x_max - x_min
        y_span = y_max - y_min
        x_stride = x_span / W
        y_stride = y_span / H

        # Generate Grid Centers
        # Centers are offset by half a stride from the grid edges
        x_centers = (
            torch.arange(W, device=device, dtype=torch.float32) * x_stride
            + x_min
            + (x_stride / 2)
        )
        y_centers = (
            torch.arange(H, device=device, dtype=torch.float32) * y_stride
            + y_min
            + (y_stride / 2)
        )

        # Create Meshgrid
        # indexing='ij' ensures y varies along dim 0 and x along dim 1
        y_grid, x_grid = torch.meshgrid(y_centers, x_centers, indexing="ij")

        anchors_list = []

        # Iterate over classes and rotations to generate specific anchors
        # The order here determines the channel mapping in the detection head
        for class_name in self.class_names:
            # Retrieve dimensions [w, l, h]
            dims = self.anchor_sizes.get(class_name, [1.0, 1.0, 1.0])
            w, l, h = dims

            # Calculate Z center to align bottom of anchor with ground
            z_center = self.ground_z + (h / 2.0)

            for rot in self.anchor_rotations:
                # Create parameter grids for this anchor type
                z_grid = torch.full_like(x_grid, z_center)
                w_grid = torch.full_like(x_grid, w)
                l_grid = torch.full_like(x_grid, l)
                h_grid = torch.full_like(x_grid, h)
                r_grid = torch.full_like(x_grid, rot)

                # Stack parameters: [x, y, z, w, l, h, yaw]
                # Shape: (H, W, 7)
                anchor = torch.stack(
                    [x_grid, y_grid, z_grid, w_grid, l_grid, h_grid, r_grid], dim=-1
                )
                anchors_list.append(anchor)

        # Stack all anchor types
        # Shape: (Num_Types, H, W, 7)
        anchors = torch.stack(anchors_list, dim=0)

        # Permute to (H, W, Num_Types, 7) to align with spatial feature maps
        anchors = anchors.permute(1, 2, 0, 3)

        return anchors
