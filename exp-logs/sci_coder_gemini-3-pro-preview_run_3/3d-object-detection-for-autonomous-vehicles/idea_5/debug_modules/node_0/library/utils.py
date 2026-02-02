import torch
import torch.nn.functional as F
import numpy as np
from library.config import POINT_CLOUD_RANGE, VOXEL_SIZE, OUT_SIZE_FACTOR


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius for the Gaussian kernel based on object dimensions and min IoU overlap.
    Derived from the focal loss penalty in CornerNet/CenterNet.

    Args:
        det_size (tuple): (height, width) of the object on the heatmap grid.
        min_overlap (float): Minimum IoU overlap required.

    Returns:
        float: The calculated radius.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    """
    Generates a 2D Gaussian kernel.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draws a 2D Gaussian on the heatmap in-place.

    Args:
        heatmap (np.ndarray): The heatmap to update (H, W).
        center (tuple): (x, y) integer coordinates on the grid.
        radius (float): Radius of the Gaussian.
        k (float): Scaling factor (usually 1).

    Returns:
        np.ndarray: The updated heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def _gather_feat(feat, ind, mask=None):
    """
    Gathers features from a tensor using indices.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    """
    Transposes feature map and gathers values at specific spatial indices.
    Args:
        feat: (B, C, H, W)
        ind: (B, K)
    Returns:
        (B, K, C)
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def decode_predictions(heatmap, dim, rot, reg, z_map, K=100):
    """
    Decodes the dense outputs of the CenterNet head into 3D bounding boxes.

    Args:
        heatmap (Tensor): Class heatmaps (B, C, H, W). Assumed to be logits.
        dim (Tensor): Dimension predictions (B, 3, H, W) -> (log l, log w, log h).
        rot (Tensor): Orientation predictions (B, 2, H, W) -> (sin, cos).
        reg (Tensor): Position offset predictions (B, 2, H, W) -> (dx, dy).
        z_map (Tensor): Height/Z coordinate predictions (B, 1, H, W).
        K (int): Number of top objects to select.

    Returns:
        Tensor: Decoded boxes of shape (B, K, 9).
                Format: [x, y, z, w, l, h, yaw, score, class_id]
                Coordinates are in the Ego-Sensor frame (meters).
    """
    batch_size, num_classes, height, width = heatmap.size()

    # Apply sigmoid to convert logits to probabilities
    heatmap = torch.sigmoid(heatmap)

    # 1. Max Pooling NMS: Find local peaks
    hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    keep = (hmax == heatmap).float()
    heatmap = heatmap * keep

    # 2. Select Top K peaks
    # Flatten to (B, C*H*W) to find top scores globally
    topk_scores, topk_inds = torch.topk(heatmap.view(batch_size, -1), K)

    # Convert flattened indices to class, y, x
    topk_clses = (topk_inds // (height * width)).float()
    topk_inds = topk_inds % (height * width)

    topk_ys = (topk_inds.div(width, rounding_mode="floor")).float()
    topk_xs = (topk_inds % width).float()

    # 3. Gather values from regression maps at peak locations
    # reg: (B, 2, H, W) -> (B, K, 2)
    topk_reg = _transpose_and_gather_feat(reg, topk_inds)
    topk_reg = topk_reg.view(batch_size, K, 2)

    # dim: (B, 3, H, W) -> (B, K, 3)
    topk_dim = _transpose_and_gather_feat(dim, topk_inds)
    topk_dim = topk_dim.view(batch_size, K, 3)

    # rot: (B, 2, H, W) -> (B, K, 2)
    topk_rot = _transpose_and_gather_feat(rot, topk_inds)
    topk_rot = topk_rot.view(batch_size, K, 2)

    # z_map: (B, 1, H, W) -> (B, K, 1)
    topk_z = _transpose_and_gather_feat(z_map, topk_inds)
    topk_z = topk_z.view(batch_size, K, 1)

    # 4. Reconstruct Geometry

    # Center X, Y (Apply grid stride and offset)
    # xs = grid_x + offset_x
    xs = topk_xs + topk_reg[..., 0]
    ys = topk_ys + topk_reg[..., 1]

    # Scale to world meters (Ego frame)
    stride_x = VOXEL_SIZE[0] * OUT_SIZE_FACTOR
    stride_y = VOXEL_SIZE[1] * OUT_SIZE_FACTOR

    final_x = xs * stride_x + POINT_CLOUD_RANGE[0]
    final_y = ys * stride_y + POINT_CLOUD_RANGE[1]
    final_z = topk_z[..., 0]

    # Dimensions (Apply exponential to log-dims)
    # Model output: (log l, log w, log h) -> index 0=l, 1=w, 2=h
    # Submission format: width, length, height
    final_dims = torch.exp(topk_dim)
    final_l = final_dims[..., 0]
    final_w = final_dims[..., 1]
    final_h = final_dims[..., 2]

    # Rotation (Yaw)
    # rot = (sin, cos)
    final_rot = torch.atan2(topk_rot[..., 0], topk_rot[..., 1])

    # 5. Stack Final Results
    # Shape: (B, K, 9)
    # [x, y, z, w, l, h, yaw, score, class_id]
    # Note: We map L, W, H to W, L, H order for output consistency with submission requirements if needed,
    # but here we return a structured tensor.
    # Let's return: x, y, z, w, l, h, yaw, score, class

    detections = torch.stack(
        [
            final_x,  # 0: center_x
            final_y,  # 1: center_y
            final_z,  # 2: center_z
            final_w,  # 3: width
            final_l,  # 4: length
            final_h,  # 5: height
            final_rot,  # 6: yaw
            topk_scores,  # 7: score
            topk_clses,  # 8: class_id
        ],
        dim=2,
    )

    return detections
