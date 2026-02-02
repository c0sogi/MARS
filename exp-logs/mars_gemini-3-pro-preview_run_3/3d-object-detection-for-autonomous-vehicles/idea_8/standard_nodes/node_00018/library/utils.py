import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import cv2
from library.config import Config


def load_point_cloud(path, load_cached_data=False):
    """
    Loads a point cloud from a .bin file.
    Assumes data is float32. Reshapes to (N, 4) or (N, 5).
    """
    if not os.path.exists(path):
        # Try relative to input dir
        path = os.path.join(Config.DATA_ROOT, path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Point cloud file not found: {path}")

    points = np.fromfile(path, dtype=np.float32)

    # Reshape based on number of features
    # Standard formats are x,y,z,intensity (4) or x,y,z,intensity,ring (5)
    if points.shape[0] % 5 == 0:
        points = points.reshape(-1, 5)
    elif points.shape[0] % 4 == 0:
        points = points.reshape(-1, 4)
    else:
        # Fallback for irregular shapes, assume 4 and truncate
        n_points = points.shape[0] // 4
        points = points[: n_points * 4].reshape(-1, 4)

    # We only need first 4 features: x, y, z, intensity
    return points[:, :4]


class Voxelizer:
    def __init__(self):
        self.voxel_size = np.array(Config.VOXEL_SIZE, dtype=np.float32)
        self.point_cloud_range = np.array(Config.POINT_CLOUD_RANGE, dtype=np.float32)
        self.grid_size = np.array(Config.GRID_SIZE, dtype=np.int32)
        self.max_points = Config.MAX_POINTS_PER_VOXEL
        self.max_voxels = Config.MAX_VOXELS_TRAIN

    def __call__(self, points, training=True):
        """
        Converts points (N, 4) to pillars.
        Returns:
            pillar_features: (M, max_points, D)
            pillar_coords: (M, 3) [z, y, x] (indices)
            num_points: (M,)
        """
        device = points.device

        # 1. Filter points outside range
        mask = (
            (points[:, 0] >= self.point_cloud_range[0])
            & (points[:, 0] < self.point_cloud_range[3])
            & (points[:, 1] >= self.point_cloud_range[1])
            & (points[:, 1] < self.point_cloud_range[4])
            & (points[:, 2] >= self.point_cloud_range[2])
            & (points[:, 2] < self.point_cloud_range[5])
        )
        points = points[mask]

        if points.shape[0] == 0:
            return None, None, None

        # 2. Calculate coordinates
        # (P - min) / size
        coords = (
            points[:, :3] - torch.tensor(self.point_cloud_range[:3], device=device)
        ) / torch.tensor(self.voxel_size, device=device)
        coords = coords.long()

        # Clamp coordinates to ensure they are within grid bounds
        # Cite debug_lesson_8
        grid_size_tensor = torch.tensor(self.grid_size, device=device)
        coords = torch.clamp(
            coords, min=torch.zeros_like(grid_size_tensor), max=grid_size_tensor - 1
        )

        # 3. Create 1D keys for unique pillars (y, x)
        # We ignore Z for pillars (Z=0)
        keys = coords[:, 1] * self.grid_size[0] + coords[:, 0]

        # 4. Group points
        # Sort by key to group points in same pillar
        sorted_idx = torch.argsort(keys)
        points = points[sorted_idx]
        coords = coords[sorted_idx]
        keys = keys[sorted_idx]

        # Find unique pillars and their counts
        unique_keys, unique_counts = torch.unique_consecutive(keys, return_counts=True)

        # Limit number of voxels
        max_vox = self.max_voxels if training else Config.MAX_VOXELS_TEST
        if len(unique_keys) > max_vox:
            # Subsample voxels (deterministic first K)
            unique_keys = unique_keys[:max_vox]
            unique_counts = unique_counts[:max_vox]
            # Slice points corresponding to these keys
            total_points = unique_counts.sum().item()
            points = points[:total_points]
            coords = coords[:total_points]

        num_pillars = len(unique_keys)

        # 5. Scatter to dense tensor (M, max_points, C)
        # Create an index for each point within its pillar [0, 1, 2, ..., count-1]
        cumsum = torch.cumsum(unique_counts, dim=0)
        starts = torch.cat(
            (torch.zeros(1, device=device, dtype=torch.long), cumsum[:-1])
        )

        # Expand starts to match points
        repeated_starts = torch.repeat_interleave(starts, unique_counts)

        # internal_idx = global_idx - start_idx
        global_idx = torch.arange(len(points), device=device)
        internal_idx = global_idx - repeated_starts

        # Filter points that exceed max_points per voxel
        valid_point_mask = internal_idx < self.max_points

        points = points[valid_point_mask]
        internal_idx = internal_idx[valid_point_mask]

        # Map which pillar each point belongs to
        pillar_indices = torch.repeat_interleave(
            torch.arange(num_pillars, device=device), unique_counts
        )
        pillar_indices = pillar_indices[valid_point_mask]

        # Initialize output tensors
        # Features: [x, y, z, i, x_c, y_c, z_c, x_p, y_p] -> 9 channels
        pillar_features = torch.zeros((num_pillars, self.max_points, 9), device=device)

        # Compute geometric features
        # 1. Mean of points in pillar
        actual_counts = torch.zeros(num_pillars, device=device, dtype=torch.long)
        actual_counts.scatter_add_(0, pillar_indices, torch.ones_like(pillar_indices))

        pillar_sum = torch.zeros((num_pillars, 3), device=device)
        pillar_sum.scatter_add_(
            0, pillar_indices.unsqueeze(1).expand(-1, 3), points[:, :3]
        )
        pillar_mean = pillar_sum / actual_counts.unsqueeze(1).clamp(min=1)

        # 2. Pillar center (geometric center of voxel)
        unique_y = unique_keys // self.grid_size[0]
        unique_x = unique_keys % self.grid_size[0]

        x_p = (
            unique_x.float() * self.voxel_size[0]
            + self.voxel_size[0] / 2
            + self.point_cloud_range[0]
        )
        y_p = (
            unique_y.float() * self.voxel_size[1]
            + self.voxel_size[1] / 2
            + self.point_cloud_range[1]
        )

        # Expand mean and center to match points
        points_mean = pillar_mean[pillar_indices]
        points_center = torch.stack([x_p, y_p], dim=1)[pillar_indices]

        # Compile features
        f_xyz = points[:, :3]
        f_i = points[:, 3:4]
        f_cluster = points[:, :3] - points_mean
        f_center = points[:, :2] - points_center

        features_flat = torch.cat(
            [f_xyz, f_i, f_cluster, f_center], dim=1
        )  # (N_valid, 9)

        # Scatter into dense tensor
        pillar_features[pillar_indices, internal_idx] = features_flat

        # Coords: (M, 3) -> (z, y, x) format
        z_coords = torch.zeros_like(unique_x)
        pillar_coords = torch.stack([z_coords, unique_y, unique_x], dim=1).int()

        return pillar_features, pillar_coords, actual_counts


def gaussian_radius(det_size, min_overlap=0.5):
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


def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        masked_gaussian_tensor = torch.tensor(
            masked_gaussian, device=heatmap.device, dtype=heatmap.dtype
        )
        torch.maximum(masked_heatmap, masked_gaussian_tensor * k, out=masked_heatmap)

    return heatmap


def get_corners_3d(boxes):
    """
    boxes: (N, 7) [x, y, z, w, l, h, yaw]
    Returns: (N, 8, 3)
    """
    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = boxes

    n = boxes_np.shape[0]
    if n == 0:
        return np.zeros((0, 8, 3))

    w, l, h = boxes_np[:, 3], boxes_np[:, 4], boxes_np[:, 5]
    x, y, z = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2]
    yaw = boxes_np[:, 6]

    # 3D bounding box corners relative to center
    # x signs: 1, 1, -1, -1, 1, 1, -1, -1
    x_signs = np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_signs = np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_signs = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    # (N, 8)
    xs = w[:, None] / 2 * x_signs[None, :]
    ys = l[:, None] / 2 * y_signs[None, :]
    zs = h[:, None] / 2 * z_signs[None, :]

    # Rotate
    c = np.cos(yaw)
    s = np.sin(yaw)

    # x_new = x*c - y*s
    # y_new = x*s + y*c
    xs_rot = xs * c[:, None] - ys * s[:, None]
    ys_rot = xs * s[:, None] + ys * c[:, None]

    # Translate
    xs_final = xs_rot + x[:, None]
    ys_final = ys_rot + y[:, None]
    zs_final = zs + z[:, None]

    return np.stack([xs_final, ys_final, zs_final], axis=-1)


def nms_3d(boxes, scores, threshold=0.1):
    """
    Applies NMS on 3D boxes using BEV approximation (Axis-Aligned Bounding Box).
    boxes: (N, 7)
    scores: (N,)
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Convert to axis aligned BEV boxes for fast NMS
    x, y, w, l, yaw = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]

    c = torch.cos(yaw)
    s = torch.sin(yaw)

    # Corners relative to center
    x_corners = torch.stack([w / 2, w / 2, -w / 2, -w / 2], dim=1)
    y_corners = torch.stack([l / 2, -l / 2, -l / 2, l / 2], dim=1)

    # Rotate
    x_rot = x_corners * c.unsqueeze(1) - y_corners * s.unsqueeze(1)
    y_rot = x_corners * s.unsqueeze(1) + y_corners * c.unsqueeze(1)

    # Translate
    x_final = x_rot + x.unsqueeze(1)
    y_final = y_rot + y.unsqueeze(1)

    # AABB
    x_min = x_final.min(dim=1)[0]
    y_min = y_final.min(dim=1)[0]
    x_max = x_final.max(dim=1)[0]
    y_max = y_final.max(dim=1)[0]

    boxes_bev = torch.stack([x_min, y_min, x_max, y_max], dim=1)

    keep = torchvision.ops.nms(boxes_bev, scores, threshold)
    return keep


def extract_roi_features(features, boxes, map_stride=1, output_size=7):
    """
    Extracts features for rotated boxes using grid_sample.
    features: (B, C, H, W)
    boxes: (B, N, 7) [x, y, z, w, l, h, yaw]
    Returns: (B, N, C, output_size, output_size)
    """
    B, C, H, W = features.shape
    N = boxes.shape[1]

    # Create base grid (output_size, output_size, 2)
    range_ = torch.linspace(-0.5, 0.5, output_size, device=features.device)
    grid_y, grid_x = torch.meshgrid(range_, range_, indexing="ij")

    # Flatten grid: (1, 1, K, 2) where K = output_size^2
    base_grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, 1, -1, 2)
    base_grid = base_grid.repeat(B, N, 1, 1)

    # Scale by dimensions (w, l)
    w = boxes[..., 3:4].unsqueeze(2)  # (B, N, 1, 1)
    l = boxes[..., 4:5].unsqueeze(2)
    grid_scaled = base_grid * torch.cat([w, l], dim=-1)

    # Rotate
    yaw = boxes[..., 6:7].unsqueeze(2)
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    x_local = grid_scaled[..., 0]
    y_local = grid_scaled[..., 1]

    x_rot = x_local * c[..., 0] - y_local * s[..., 0]
    y_rot = x_local * s[..., 0] + y_local * c[..., 0]

    # Translate
    x_world = x_rot + boxes[..., 0:1]
    y_world = y_rot + boxes[..., 1:2]

    # Normalize to feature map coordinates [-1, 1]
    pc_range = torch.tensor(Config.POINT_CLOUD_RANGE, device=features.device)
    min_x, min_y = pc_range[0], pc_range[1]
    max_x, max_y = pc_range[3], pc_range[4]

    u = 2 * (x_world - min_x) / (max_x - min_x) - 1
    v = 2 * (y_world - min_y) / (max_y - min_y) - 1

    # Reshape grid to (B, N * output_size * output_size, 1, 2) for sampling
    grid_final = torch.stack([u, v], dim=-1).reshape(
        B, N * output_size * output_size, 1, 2
    )

    # Sample
    sampled = F.grid_sample(features, grid_final, align_corners=False)

    # Reshape back to (B, N, C, output_size, output_size)
    sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, L, C)
    sampled = sampled.reshape(B, N, output_size, output_size, C).permute(0, 1, 4, 2, 3)

    return sampled


def encode_refinement_targets(proposals, gt_boxes):
    """
    Encodes residuals between proposals and GT.
    proposals: (B, N, 7)
    gt_boxes: (B, N, 7)
    """
    # xyz residuals
    d_xyz = gt_boxes[..., :3] - proposals[..., :3]

    # dim residuals (log space)
    d_dim = torch.log(gt_boxes[..., 3:6] / proposals[..., 3:6].clamp(min=1e-6))

    # yaw residuals (sin/cos of difference)
    d_yaw = gt_boxes[..., 6] - proposals[..., 6]
    d_yaw_sin = torch.sin(d_yaw).unsqueeze(-1)
    d_yaw_cos = torch.cos(d_yaw).unsqueeze(-1)

    return torch.cat([d_xyz, d_dim, d_yaw_sin, d_yaw_cos], dim=-1)


def decode_refinement(proposals, residuals):
    """
    Decodes residuals to get final boxes.
    residuals: (B, N, 8) [dx, dy, dz, dw, dl, dh, sin_d, cos_d]
    """
    xyz = proposals[..., :3] + residuals[..., :3]
    dim = proposals[..., 3:6] * torch.exp(residuals[..., 3:6])

    delta_yaw = torch.atan2(residuals[..., 6], residuals[..., 7])
    yaw = proposals[..., 6] + delta_yaw

    return torch.cat([xyz, dim, yaw.unsqueeze(-1)], dim=-1)
