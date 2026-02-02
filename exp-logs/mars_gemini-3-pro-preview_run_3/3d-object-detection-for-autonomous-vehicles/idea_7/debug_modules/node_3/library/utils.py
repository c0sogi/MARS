import torch
import torch.nn.functional as F
import numpy as np
import math
from library.config import VOXEL_SIZE, POINT_CLOUD_RANGE


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculate the radius for the Gaussian kernel based on object dimensions and IoU overlap.
    Based on CornerNet/CenterNet derivation.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    """
    Generate a 2D Gaussian kernel.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Draw a 2D Gaussian on the heatmap at the specified center.
    Uses max operation to handle overlaps.
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
    Gather features from a specific index.
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
    Transpose feature map and gather features at specific spatial indices.
    Args:
        feat: (B, C, H, W)
        ind: (B, K) spatial indices
    Returns:
        (B, K, C)
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def decode_predictions(heatmap, rot, dim, height, reg, K=100, score_threshold=None):
    """
    Decode model outputs into 3D bounding boxes.

    Args:
        heatmap: (B, C, H, W) - Class heatmaps
        rot: (B, 2, H, W) - sin(yaw), cos(yaw)
        dim: (B, 3, H, W) - log(l), log(w), log(h)
        height: (B, 1, H, W) - z center
        reg: (B, 2, H, W) - x, y offsets relative to voxel center
        K: Top K objects to keep per sample
        score_threshold: Optional threshold to filter low confidence objects

    Returns:
        List of tensors, one per sample: (Num_Objects, 9)
        [x, y, z, width, length, height, yaw, score, class_id]
    """
    batch_size, num_classes, H, W = heatmap.size()

    # 1. Heatmap processing: Sigmoid and 3x3 Max Pooling (NMS)
    heatmap = torch.sigmoid(heatmap)
    hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    keep = (hmax == heatmap).float()
    heatmap = heatmap * keep

    # 2. Top K selection
    # Flatten to (B, C*H*W) to find top K peaks across all classes and locations
    topk_scores, topk_inds = torch.topk(heatmap.view(batch_size, -1), K)

    # Decouple class index and spatial index
    topk_clses = (topk_inds // (H * W)).int()
    topk_inds = topk_inds % (H * W)

    # Calculate grid coordinates
    topk_ys = (topk_inds // W).int().float()
    topk_xs = (topk_inds % W).int().float()

    # 3. Gather regression features at peak locations
    # All regression maps are (B, C_reg, H, W), we need (B, K, C_reg)
    reg = _transpose_and_gather_feat(reg, topk_inds)
    height = _transpose_and_gather_feat(height, topk_inds)
    dim = _transpose_and_gather_feat(dim, topk_inds)
    rot = _transpose_and_gather_feat(rot, topk_inds)

    # 4. Decode Geometry

    # Center X, Y: Grid Index + Offset -> World Coordinate
    # Note: reg is usually normalized or in voxel units.
    # Formula: (grid_idx + offset) * voxel_size + min_point_cloud_range
    xs = topk_xs + reg[..., 0]
    ys = topk_ys + reg[..., 1]

    x_world = xs * VOXEL_SIZE[0] + POINT_CLOUD_RANGE[0]
    y_world = ys * VOXEL_SIZE[1] + POINT_CLOUD_RANGE[1]

    # Center Z: Directly predicted
    z_world = height[..., 0]

    # Dimensions: Network predicts log(dim), so take exp()
    # dim input is (l, w, h) in log space
    dim = torch.exp(dim)

    # Rotation: Network predicts sin(r), cos(r)
    yaw = torch.atan2(rot[..., 0], rot[..., 1])

    # 5. Assemble final bounding boxes
    final_box_preds = []

    for i in range(batch_size):
        # Extract dimensions for this sample
        # Submission format requires: width, length, height
        # dim tensor is: length(0), width(1), height(2)
        l = dim[i, :, 0]
        w = dim[i, :, 1]
        h = dim[i, :, 2]

        # Stack: [x, y, z, width, length, height, yaw, score]
        box = torch.stack(
            [x_world[i], y_world[i], z_world[i], w, l, h, yaw[i], topk_scores[i]],
            dim=-1,
        )

        # Append class index
        clses = topk_clses[i].float().unsqueeze(-1)
        box = torch.cat([box, clses], dim=-1)

        # Optional: Filter by score
        if score_threshold is not None:
            mask = box[:, 7] > score_threshold
            box = box[mask]

        final_box_preds.append(box)

    return final_box_preds


def collater(batch_list):
    """
    Collate function for PyTorch DataLoader to handle point clouds and variable-length boxes.

    Args:
        batch_list: List of dictionaries returned by Dataset.__getitem__

    Returns:
        Dictionary with batched tensors:
            - points: (N_total, 5) [batch_idx, x, y, z, intensity]
            - gt_boxes: (B, Max_Objects, 8) [padded]
            - gt_labels: (B, Max_Objects) [padded]
            - metadata: List of metadata dicts
    """
    batched_points = []
    batched_gt_boxes = []
    batched_labels = []
    batched_metadata = []

    for i, sample in enumerate(batch_list):
        # 1. Process Points: Add batch index column
        points = sample["points"]
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)

        # Create batch index (N, 1)
        batch_idx = torch.full((points.shape[0], 1), i, dtype=points.dtype)
        # Concatenate [batch_idx, x, y, z, i]
        points_with_idx = torch.cat([batch_idx, points], dim=1)
        batched_points.append(points_with_idx)

        # 2. Collect GT Boxes and Labels
        if "gt_boxes" in sample:
            batched_gt_boxes.append(torch.from_numpy(sample["gt_boxes"]))
        if "gt_labels" in sample:
            batched_labels.append(torch.from_numpy(sample["gt_labels"]))

        # 3. Collect Metadata
        if "metadata" in sample:
            batched_metadata.append(sample["metadata"])

    # Concatenate all points into a single large tensor
    batched_points = torch.cat(batched_points, dim=0)

    # Pad GT boxes and labels to form a batch tensor
    padded_boxes = None
    padded_labels = None

    if len(batched_gt_boxes) > 0:
        max_boxes = max([b.shape[0] for b in batched_gt_boxes])
        if max_boxes > 0:
            batch_size = len(batch_list)
            # Initialize with zeros
            padded_boxes = torch.zeros(
                (batch_size, max_boxes, batched_gt_boxes[0].shape[1])
            )
            padded_labels = torch.zeros((batch_size, max_boxes)).long()

            for i, (boxes, labels) in enumerate(zip(batched_gt_boxes, batched_labels)):
                num_obj = boxes.shape[0]
                padded_boxes[i, :num_obj] = boxes
                padded_labels[i, :num_obj] = labels
        else:
            # Handle empty batch case
            padded_boxes = torch.zeros((len(batch_list), 0, 8))
            padded_labels = torch.zeros((len(batch_list), 0)).long()

    return {
        "points": batched_points,
        "gt_boxes": padded_boxes,
        "gt_labels": padded_labels,
        "metadata": batched_metadata,
        "batch_size": len(batch_list),
    }
