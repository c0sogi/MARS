import torch
import torch.nn as nn
import numpy as np


class StainDeconvolution(nn.Module):
    """
    A PyTorch module for GPU-accelerated Stain Deconvolution.

    This layer converts RGB images into Hematoxylin and Eosin (H&E) optical density maps
    using a fixed color deconvolution matrix. It outputs a 5-channel tensor consisting
    of the original RGB channels followed by the H and E channels.
    """

    def __init__(self):
        super(StainDeconvolution, self).__init__()

        # Standard H&E OD matrix (Ruifrok and Johnston)
        # Rows represent the absorption of Red, Green, Blue for a specific stain.
        # Row 0: Hematoxylin
        # Row 1: Eosin
        # Row 2: Residual (calculated as cross product or orthogonal vector)

        # Normalized OD vectors
        h_vector = np.array([0.644211, 0.716556, 0.266844])
        e_vector = np.array([0.092789, 0.954111, 0.283111])

        # Calculate residual vector (normalized cross product)
        r_vector = np.cross(h_vector, e_vector)
        r_vector = r_vector / np.linalg.norm(r_vector)

        # Stack to form the Stain Matrix M where OD = C * M
        # Shape: (3, 3)
        stain_matrix = np.vstack((h_vector, e_vector, r_vector))

        # We need to solve for C (Concentrations): C = OD * M^-1
        try:
            inverse_matrix = np.linalg.inv(stain_matrix)
        except np.linalg.LinAlgError:
            # Fallback identity if singular (unlikely with standard vectors)
            inverse_matrix = np.eye(3)

        # Convert to torch tensor
        # We use float32 for standard training
        self.register_buffer("inverse_matrix", torch.from_numpy(inverse_matrix).float())

        # Epsilon to avoid log(0)
        self.epsilon = 1e-6

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input RGB tensor of shape (B, 3, H, W).
                              Expected value range is [0, 1].

        Returns:
            torch.Tensor: Output tensor of shape (B, 5, H, W).
                          Channels: R, G, B, Hematoxylin, Eosin.
        """
        # Ensure input is clamped to avoid log(0) and handle slight overshoots
        # x is (B, 3, H, W)
        x_clamped = torch.clamp(x, self.epsilon, 1.0)

        # 1. Convert RGB to Optical Density (OD)
        # Beer-Lambert law: OD = -log(I / I0)
        # Assuming I0 = 1.0 (since data is normalized 0-1)
        od = -torch.log(x_clamped)

        # 2. Prepare for Matrix Multiplication
        # We need shape (B, H, W, 3) to multiply with (3, 3) inverse matrix
        # Permute: (B, 3, H, W) -> (B, H, W, 3)
        od_permuted = od.permute(0, 2, 3, 1)

        # 3. Deconvolve
        # C = OD * M^-1
        # (B, H, W, 3) @ (3, 3) -> (B, H, W, 3)
        # Note: inverse_matrix is (3, 3). matmul broadcasts over B, H, W.
        # Since inverse_matrix is registered as a buffer, it's on the correct device.
        stains = torch.matmul(od_permuted, self.inverse_matrix.t())

        # 4. Extract H and E
        # Channel 0: Hematoxylin
        # Channel 1: Eosin
        # Channel 2: Residual (discarded)

        # Permute back to (B, 3, H, W)
        stains_permuted = stains.permute(0, 3, 1, 2)

        # Extract H and E channels
        h_channel = stains_permuted[:, 0:1, :, :]
        e_channel = stains_permuted[:, 1:2, :, :]

        # Optional: Apply a non-linearity or normalization to H/E if needed.
        # Typically, we keep them as raw density values or clamp negative values (noise).
        h_channel = torch.relu(h_channel)
        e_channel = torch.relu(e_channel)

        # 5. Concatenate with original RGB
        # Result shape: (B, 5, H, W)
        out = torch.cat([x, h_channel, e_channel], dim=1)

        return out
