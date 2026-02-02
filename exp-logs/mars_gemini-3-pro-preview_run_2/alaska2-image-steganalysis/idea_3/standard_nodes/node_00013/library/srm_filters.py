import numpy as np
import torch
import torch.nn as nn


def get_srm_layer():
    """
    Creates a fixed Conv2d layer initialized with 30 SRM (Spatial Rich Model) high-pass filters.

    Returns:
        nn.Conv2d: A convolutional layer with 3 input channels, 30 output channels,
                   kernel size 5x5, and fixed weights.
    """
    # 1. Define Basic 1D Kernels
    # Note: We pad them to length 5 to simplify 2D construction
    # 1st order: [-1, 1]
    k1 = np.array([0, 0, -1, 1, 0])
    # 2nd order: [-1, 2, -1]
    k2 = np.array([0, -1, 2, -1, 0])
    # 3rd order: [-1, 3, -3, 1]
    k3 = np.array([0, -1, 3, -3, 1])
    # 4th order: [-1, 4, -6, 4, -1]
    k4 = np.array([-1, 4, -6, 4, -1])

    filters = []

    # Helper to rotate a 5x5 grid
    def rotate_filters(base_kernel_2d):
        # Returns list of [Original, Rot90, Rot180, Rot270]
        # Note: For symmetric kernels, some of these are duplicates,
        # but we filter duplicates later or just accept redundancy for the fixed stem.
        # For derivatives, we usually want H, V, D1, D2.

        # H (Original)
        f_h = base_kernel_2d
        # V (Transposed)
        f_v = base_kernel_2d.T

        # For diagonals, we construct them manually or rotate.
        # Simple 90 deg rotation of H gives V.
        # We need specific diagonal construction for derivatives.
        return [f_h, f_v]

    # --- Group 1: Basic Derivatives (H and V) ---
    # Construct 2D kernels by outer product with a "smoothing" or "identity" kernel?
    # SRM usually applies the 1D kernel in one direction and a specific profile in the other.
    # Here we use the standard "spam" approach: Kernel in one dir, Identity (1) in other (center).

    center_pulse = np.array([0, 0, 1, 0, 0])

    # 1st Order (H, V)
    f1_h = np.outer(center_pulse, k1)
    filters.extend(rotate_filters(f1_h))  # Adds H, V

    # 2nd Order (H, V)
    f2_h = np.outer(center_pulse, k2)
    filters.extend(rotate_filters(f2_h))

    # 3rd Order (H, V)
    f3_h = np.outer(center_pulse, k3)
    filters.extend(rotate_filters(f3_h))

    # 4th Order (H, V)
    f4_h = np.outer(center_pulse, k4)
    filters.extend(rotate_filters(f4_h))

    # Current count: 8

    # --- Group 2: Diagonal Derivatives ---
    # We construct these by placing the 1D kernel along the diagonal

    def make_diag(k_1d):
        f = np.zeros((5, 5))
        # k_1d is length 5.
        for i in range(5):
            f[i, i] = k_1d[i]
        return f

    def make_anti_diag(k_1d):
        f = np.zeros((5, 5))
        for i in range(5):
            f[i, 4 - i] = k_1d[i]
        return f

    # 1st Order Diagonals
    filters.append(make_diag(k1))
    filters.append(make_anti_diag(k1))

    # 2nd Order Diagonals
    filters.append(make_diag(k2))
    filters.append(make_anti_diag(k2))

    # 3rd Order Diagonals
    filters.append(make_diag(k3))
    filters.append(make_anti_diag(k3))

    # 4th Order Diagonals
    filters.append(make_diag(k4))
    filters.append(make_anti_diag(k4))

    # Current count: 8 + 8 = 16

    # --- Group 3: Mixed Derivatives (Tensor Products) ---
    # k1 x k1 (Checkerboard-ish)
    filters.append(np.outer(k1, k1))

    # k1 x k2 (H and V variants)
    f_12 = np.outer(k1, k2)
    filters.extend(rotate_filters(f_12))

    # k1 x k3
    f_13 = np.outer(k1, k3)
    filters.extend(rotate_filters(f_13))

    # k2 x k2
    filters.append(np.outer(k2, k2))

    # k2 x k3
    f_23 = np.outer(k2, k3)
    filters.extend(rotate_filters(f_23))

    # k3 x k3
    filters.append(np.outer(k3, k3))

    # Current count: 16 + 1 + 2 + 2 + 1 + 2 + 1 = 25

    # --- Group 4: Area / Edge Kernels ---

    # Square 3x3 (Center 1, neighbors -1/8)
    s3 = np.zeros((5, 5))
    s3[1:4, 1:4] = -1 / 8
    s3[2, 2] = 1
    filters.append(s3)

    # Square 5x5 (Center 1, neighbors -1/24)
    s5 = -1 / 24 * np.ones((5, 5))
    s5[2, 2] = 1
    filters.append(s5)

    # Edge 3x3 (Laplacian)
    e3 = np.zeros((5, 5))
    e3[1:4, 1:4] = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    filters.append(e3)

    # Edge 5x5 type
    # A simple 5x5 Laplacian approximation
    e5 = np.zeros((5, 5))
    e5[2, 2] = 4
    e5[1, 2] = -1
    e5[3, 2] = -1
    e5[2, 1] = -1
    e5[2, 3] = -1
    # Add a wider ring for 5x5 effect
    e5[0, 2] = -0.25
    e5[4, 2] = -0.25
    e5[2, 0] = -0.25
    e5[2, 4] = -0.25
    e5[2, 2] += 1  # Balance the center
    # Normalize sum to 0
    e5 = e5 - e5.mean()
    filters.append(e5)

    # Current count: 25 + 4 = 29.

    # Add one more: k4 x k4 (High freq corner)
    filters.append(np.outer(k4, k4))

    # Total: 30 filters.

    # 2. Stack and Format
    # Shape: (30, 5, 5)
    filters_np = np.array(filters, dtype=np.float32)

    # 3. Create Conv2d Layer
    # We want to apply these 30 filters to the RGB input.
    # To do this using standard Conv2d(3, 30), we need weights of shape (30, 3, 5, 5).
    # We replicate the 5x5 filter across the 3 input channels.
    # This effectively computes: Output = Filter * (R + G + B)

    filters_np = np.stack([filters_np, filters_np, filters_np], axis=1)  # (30, 3, 5, 5)

    srm_layer = nn.Conv2d(
        in_channels=3, out_channels=30, kernel_size=5, stride=1, padding=2, bias=False
    )

    # Assign weights
    srm_layer.weight.data = torch.from_numpy(filters_np)

    # Freeze weights
    srm_layer.weight.requires_grad = False

    return srm_layer
