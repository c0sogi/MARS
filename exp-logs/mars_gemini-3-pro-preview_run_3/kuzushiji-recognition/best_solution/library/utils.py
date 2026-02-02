import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius for the Gaussian kernel based on the bounding box size.
    Ensures that the generated positive region does not overlap with negatives
    beyond the threshold.
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


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Draws a 2D Gaussian on the heatmap at the specified center.
    Modifies the heatmap in-place.
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


def _nms(heatmap, kernel=3):
    """
    Applies Max Pooling to find local maxima (peaks) in the heatmap.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def decode_detections(heatmap, size_map, offset_map, K=100, stride=None):
    """
    Decodes the output of the CenterNet-style detector.

    Args:
        heatmap: (Batch, 1, H, W) - Textness score
        size_map: (Batch, 2, H, W) - Width, Height (in feature map scale)
        offset_map: (Batch, 2, H, W) - X-offset, Y-offset
        K: Number of top detections to extract
        stride: Downsampling factor to map back to input image size.
                Defaults to Config.DETECTOR_OUTPUT_STRIDE.

    Returns:
        detections: (Batch, K, 6) tensor containing [x, y, w, h, score, class_idx]
                    Coordinates are in the scale of the model input (e.g., 1024x1024).
    """
    if stride is None:
        stride = Config.DETECTOR_OUTPUT_STRIDE

    batch, _, height, width = heatmap.shape

    # Apply NMS
    heatmap = _nms(heatmap)

    # Flatten variables
    scores = heatmap.view(batch, -1)
    inds = torch.topk(scores, K)
    topk_scores = inds.values
    topk_inds = inds.indices

    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()

    # Gather sizes and offsets
    # size_map: B, 2, H, W -> permute to B, H, W, 2 -> view B, H*W, 2
    size_map = size_map.permute(0, 2, 3, 1).contiguous().view(batch, -1, 2)
    offset_map = offset_map.permute(0, 2, 3, 1).contiguous().view(batch, -1, 2)

    # Expand indices to match dimensions for gather: (B, K) -> (B, K, 2)
    topk_inds_expanded = topk_inds.unsqueeze(-1).expand(-1, -1, 2)

    topk_sizes = torch.gather(size_map, 1, topk_inds_expanded)
    topk_offsets = torch.gather(offset_map, 1, topk_inds_expanded)

    # Apply offsets to center coordinates (feature map scale)
    topk_xs = topk_xs + topk_offsets[..., 0]
    topk_ys = topk_ys + topk_offsets[..., 1]

    # Scale everything back to input image size (e.g. 1024x1024)
    # Assuming sizes and offsets were regressed in feature map pixels
    topk_xs = topk_xs * stride
    topk_ys = topk_ys * stride
    topk_w = topk_sizes[..., 0] * stride
    topk_h = topk_sizes[..., 1] * stride

    # Stack results: x, y, w, h, score, class (0 for agnostic)
    # Shape: (B, K, 6)
    clses = torch.zeros_like(topk_scores).unsqueeze(-1)

    detections = torch.stack([topk_xs, topk_ys, topk_w, topk_h, topk_scores], dim=2)

    detections = torch.cat([detections, clses], dim=2)

    return detections


def transform_coordinates(detections, original_shapes, input_size):
    """
    Rescales detections from model input size to original image size.

    Args:
        detections: (Batch, K, 6) [x, y, w, h, score, class]
        original_shapes: List of (H, W) tuples for each image in batch
        input_size: Integer, the model input size (e.g., 1024)

    Returns:
        rescaled_list: List of numpy arrays (K, 6) with coordinates in original image space.
    """
    rescaled_list = []

    # Ensure detections are on CPU numpy
    if isinstance(detections, torch.Tensor):
        detections_np = detections.detach().cpu().numpy()
    else:
        detections_np = detections

    for i in range(len(original_shapes)):
        orig_h, orig_w = original_shapes[i]

        # Calculate scale factors
        # Note: This assumes simple resizing was used (no padding) or padding handled elsewhere.
        # If padding was used, this needs to account for it.
        # For this pipeline, we assume simple resize for simplicity unless specified.
        scale_x = orig_w / input_size
        scale_y = orig_h / input_size

        det = detections_np[i].copy()

        # Scale x (center), w
        det[:, 0] *= scale_x
        det[:, 2] *= scale_x

        # Scale y (center), h
        det[:, 1] *= scale_y
        det[:, 3] *= scale_y

        rescaled_list.append(det)

    return rescaled_list


def f1_score_calc(preds, targets):
    """
    Calculates the modified F1 score for the Kuzushiji recognition task.

    Args:
        preds: List of lists. Each inner list contains dictionaries:
               {'label': str, 'x': int, 'y': int, 'score': float}
        targets: List of lists. Each inner list contains dictionaries:
               {'label': str, 'x': int, 'y': int, 'w': int, 'h': int}

    Returns:
        dict: {'precision': float, 'recall': float, 'f1': float}
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_preds, img_targets in zip(preds, targets):

        # Sort predictions by score (descending) to prioritize high confidence matches
        img_preds_sorted = sorted(
            img_preds, key=lambda p: p.get("score", 0), reverse=True
        )

        matched_target_indices = set()

        # Count TPs and FPs
        for p in img_preds_sorted:
            p_label = p["label"]
            p_x = p["x"]
            p_y = p["y"]

            match_found = False

            for t_idx, t in enumerate(img_targets):
                if t_idx in matched_target_indices:
                    continue

                if p_label == t["label"]:
                    # Check if point is inside box
                    # GT format: x, y, w, h (top-left x, top-left y, width, height)
                    # Note: p_x, p_y are center coordinates from the model
                    if (t["x"] <= p_x < t["x"] + t["w"]) and (
                        t["y"] <= p_y < t["y"] + t["h"]
                    ):

                        matched_target_indices.add(t_idx)
                        match_found = True
                        break

            if match_found:
                total_tp += 1
            else:
                total_fp += 1

        # Count FNs (targets not matched)
        total_fn += len(img_targets) - len(matched_target_indices)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def collate_fn_detector(batch):
    """
    Custom collate function for the detector dataloader.
    Filters out None values and stacks tensors.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None

    imgs = torch.stack([item["img"] for item in batch])
    heatmaps = torch.stack([item["heatmap"] for item in batch])
    size_maps = torch.stack([item["size_map"] for item in batch])
    offset_maps = torch.stack([item["offset_map"] for item in batch])

    # Metadata can be a list
    meta = [item["meta"] for item in batch]

    return {
        "img": imgs,
        "heatmap": heatmaps,
        "size_map": size_maps,
        "offset_map": offset_maps,
        "meta": meta,
    }


def collate_fn_classifier(batch):
    """
    Custom collate function for the classifier dataloader.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None

    imgs = torch.stack([item["img"] for item in batch])
    labels = torch.tensor([item["label_idx"] for item in batch], dtype=torch.long)

    return {"img": imgs, "labels": labels}
