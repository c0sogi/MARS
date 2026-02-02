import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from library.config import Config

# ====================================================
# Gaussian Utilities for Heatmap Generation
# ====================================================


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius for the Gaussian kernel based on the bounding box size.
    Derived from CornerNet/CenterNet.
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


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Draws a 2D Gaussian on the heatmap in-place.
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


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


# ====================================================
# Label Encoder
# ====================================================


class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.char2idx = {}
        self.idx2char = {}

    def fit(self, load_cached_data=True):
        """
        Fits the label encoder.
        If load_cached_data is True and cache exists, loads from disk.
        Otherwise, processes metadata to find all unique characters.
        """
        cache_path = Config.LABEL_ENCODER_PATH

        # Ensure working directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        loaded = False
        if load_cached_data and os.path.exists(cache_path):
            try:
                # Load numpy array of strings
                self.classes_ = np.load(cache_path)
                loaded = True
            except Exception as e:
                print(f"Failed to load cached label encoder: {e}")
                loaded = False

        if not loaded:
            print("Building Label Encoder from training metadata...")
            df = pd.read_csv(Config.TRAIN_METADATA_PATH)

            # Extract all unicode characters
            all_chars = set()
            for label_str in df["labels"].dropna():
                parts = label_str.strip().split(" ")
                # Format: Unicode X Y W H ... (every 5th item)
                chars = parts[0::5]
                all_chars.update(chars)

            # Sort for determinism
            self.classes_ = np.array(sorted(list(all_chars)))

            # Save to cache
            np.save(cache_path, self.classes_)
            print(f"Label Encoder saved to {cache_path}")

        # Build lookup dicts
        self.idx2char = {i: c for i, c in enumerate(self.classes_)}
        self.char2idx = {c: i for i, c in enumerate(self.classes_)}

        return self

    def transform(self, char):
        return self.char2idx.get(char, -1)  # Return -1 or handle unknown

    def inverse_transform(self, idx):
        return self.idx2char.get(idx, "")

    def __len__(self):
        return len(self.classes_) if self.classes_ is not None else 0


# ====================================================
# Decoding Predictions
# ====================================================


def _nms(heatmap, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def decode_predictions(heatmap, offset, wh, K=100):
    """
    Decodes model output into bounding boxes/points.

    Args:
        heatmap: (batch, num_classes, H, W)
        offset: (batch, 2, H, W) - local offset
        wh: (batch, 2, H, W) - width and height
        K: Top K detections to keep

    Returns:
        dets: (batch, K, 6) [x, y, w, h, score, class_id]
    """
    batch, cat, height, width = heatmap.size()

    # Apply NMS via Max Pooling
    heatmap = _nms(heatmap)

    # Flatten vars
    scores = heatmap.view(batch, -1)

    # Get Top K
    topk_scores, topk_inds = torch.topk(scores, K)

    topk_clses = (topk_inds // (height * width)).float()
    topk_inds = topk_inds % (height * width)

    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()

    # Get Offset and WH at peak locations
    # offset: (b, 2, h, w) -> permute to (b, h, w, 2) -> view (b, h*w, 2)
    offset = offset.permute(0, 2, 3, 1).contiguous().view(batch, -1, 2)
    wh = wh.permute(0, 2, 3, 1).contiguous().view(batch, -1, 2)

    # Gather values
    # We need to expand inds to match last dim
    topk_inds_expand = topk_inds.unsqueeze(2).expand(batch, K, 2)

    topk_offset = torch.gather(offset, 1, topk_inds_expand)
    topk_wh = torch.gather(wh, 1, topk_inds_expand)

    # Refine locations
    topk_xs = topk_xs + topk_offset[:, :, 0]
    topk_ys = topk_ys + topk_offset[:, :, 1]

    # Assemble detections: x, y, w, h, score, class
    # Note: x, y are center coordinates in feature map scale
    dets = torch.stack(
        [topk_xs, topk_ys, topk_wh[:, :, 0], topk_wh[:, :, 1], topk_scores, topk_clses],
        dim=2,
    )

    return dets


# ====================================================
# Metrics
# ====================================================


def calc_f1_score(pred_strs, gt_strs):
    """
    Calculates the modified F1 score for the batch or dataset.

    Args:
        pred_strs: List of prediction strings for each image.
                   Format: "Label X Y Label X Y ..."
        gt_strs: List of ground truth strings for each image.
                 Format: "Label X Y W H Label X Y W H ..."

    Returns:
        dict with precision, recall, f1
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for p_str, g_str in zip(pred_strs, gt_strs):
        # Parse Ground Truth
        gt_boxes = []
        if isinstance(g_str, str) and len(g_str) > 0:
            parts = g_str.strip().split(" ")
            # Label X Y W H
            for i in range(0, len(parts), 5):
                try:
                    label = parts[i]
                    x = int(parts[i + 1])
                    y = int(parts[i + 2])
                    w = int(parts[i + 3])
                    h = int(parts[i + 4])
                    gt_boxes.append(
                        {
                            "label": label,
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                            "matched": False,
                        }
                    )
                except:
                    pass

        # Parse Predictions
        preds = []
        if isinstance(p_str, str) and len(p_str) > 0:
            parts = p_str.strip().split(" ")
            # Label X Y
            for i in range(0, len(parts), 3):
                try:
                    label = parts[i]
                    x = int(parts[i + 1])
                    y = int(parts[i + 2])
                    preds.append({"label": label, "x": x, "y": y})
                except:
                    pass

        # Match
        # A prediction is a TP if it falls within a GT box of the same label
        # One GT box can only match one prediction.

        current_tp = 0

        for pred in preds:
            matched = False
            for gt in gt_boxes:
                if gt["matched"]:
                    continue

                if pred["label"] == gt["label"]:
                    # Check geometry: point inside box
                    if (gt["x"] <= pred["x"] <= gt["x"] + gt["w"]) and (
                        gt["y"] <= pred["y"] <= gt["y"] + gt["h"]
                    ):
                        gt["matched"] = True
                        matched = True
                        break

            if matched:
                current_tp += 1

        current_fp = len(preds) - current_tp
        current_fn = len(gt_boxes) - current_tp

        total_tp += current_tp
        total_fp += current_fp
        total_fn += current_fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


# ====================================================
# Logging & Checkpointing
# ====================================================


class Tracker:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filepath):
    torch.save(state, filepath)
    if is_best:
        # Construct best path (usually handled by caller, but we can enforce logic here if needed)
        # The Config defines BEST_MODEL_PATH, so we assume filepath is just the current epoch path
        # or we just overwrite best.
        # Based on Config, we only have BEST_MODEL_PATH.
        pass
