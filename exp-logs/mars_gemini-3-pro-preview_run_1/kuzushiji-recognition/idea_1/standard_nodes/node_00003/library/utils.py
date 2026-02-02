import os
import math
import numpy as np
import cv2
import pandas as pd
import torch
from library.config import Config

# -----------------------------------------------------------------------------
# Math & Geometry Utilities
# -----------------------------------------------------------------------------


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Calculates the radius for the Gaussian kernel based on the bounding box size.
    Ensures that the IoU between the ground truth box and the generated box
    is at least min_overlap.
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
    Draws a Gaussian kernel onto the heatmap tensor at the specified center.
    """
    radius = int(radius)
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


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False
):
    """
    Generates the affine transformation matrix for resizing/cropping.
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(dst, src)
    else:
        trans = cv2.getAffineTransform(src, dst)

    return trans


def affine_transform(pt, t):
    """
    Applies the affine transform to a 2D point.
    """
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


# -----------------------------------------------------------------------------
# Data Processing Utilities
# -----------------------------------------------------------------------------


def load_unicode_map():
    """
    Loads the unicode translation file and returns mapping dictionaries.
    """
    df = pd.read_csv(Config.UNICODE_MAP_PATH)
    df = df.dropna(subset=["Unicode"])

    chars = df["Unicode"].values
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for i, c in enumerate(chars)}

    return char_to_id, id_to_char


def preprocess_metadata(
    metadata_path, char_to_id, load_cached_data=True, split_name="train"
):
    """
    Parses the metadata CSV, extracts labels, and caches the result.
    Returns a list of dictionaries containing image info and annotations.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_file_imgs = os.path.join(Config.CACHE_DIR, f"{split_name}_imgs.npy")
    cache_file_anns = os.path.join(Config.CACHE_DIR, f"{split_name}_anns.npy")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(cache_file_imgs)
        and os.path.exists(cache_file_anns)
    ):
        try:
            # Load from cache (No pickle)
            imgs_data = np.load(cache_file_imgs, allow_pickle=False)
            anns_data = np.load(cache_file_anns, allow_pickle=False)

            data = []
            # Efficiently group annotations back to images
            df_anns = pd.DataFrame(
                anns_data, columns=["img_idx", "class_id", "x", "y", "w", "h"]
            )
            grouped_anns = df_anns.groupby("img_idx")

            for idx, row in enumerate(imgs_data):
                img_id, file_path, group_id = row

                if idx in grouped_anns.groups:
                    group = grouped_anns.get_group(idx)
                    anns = group[["class_id", "x", "y", "w", "h"]].values
                else:
                    anns = np.zeros((0, 5), dtype=np.float32)

                data.append(
                    {
                        "image_id": img_id,
                        "file_path": file_path,
                        "group_id": group_id,
                        "annotations": anns,
                    }
                )

            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    df = pd.read_csv(metadata_path)
    df["labels"] = df["labels"].fillna("")

    imgs_list = []
    anns_list = []

    for idx, row in df.iterrows():
        img_id = row["image_id"]
        file_path = row["file_path"]
        group_id = row["group_id"] if "group_id" in row else ""

        imgs_list.append([img_id, file_path, str(group_id)])

        label_str = row["labels"]
        if not label_str:
            continue

        parts = label_str.strip().split(" ")
        # Format: Unicode X Y W H
        if len(parts) % 5 != 0:
            continue

        num_anns = len(parts) // 5
        for i in range(num_anns):
            base = i * 5
            unicode_char = parts[base]
            try:
                x = float(parts[base + 1])
                y = float(parts[base + 2])
                w = float(parts[base + 3])
                h = float(parts[base + 4])

                class_id = char_to_id.get(unicode_char, -1)
                if class_id != -1:
                    anns_list.append([idx, class_id, x, y, w, h])
            except ValueError:
                continue

    # Convert to numpy
    imgs_np = np.array(imgs_list)
    anns_np = np.array(anns_list, dtype=np.float32)

    if anns_np.size == 0:
        anns_np = anns_np.reshape(0, 6)

    # Save to cache
    np.save(cache_file_imgs, imgs_np)
    np.save(cache_file_anns, anns_np)

    # Construct return data
    data = []
    df_anns = pd.DataFrame(anns_np, columns=["img_idx", "class_id", "x", "y", "w", "h"])
    grouped_anns = df_anns.groupby("img_idx")

    for idx, row_data in enumerate(imgs_list):
        img_id, file_path, group_id = row_data

        if idx in grouped_anns.groups:
            group = grouped_anns.get_group(idx)
            anns = group[["class_id", "x", "y", "w", "h"]].values
        else:
            anns = np.zeros((0, 5), dtype=np.float32)

        data.append(
            {
                "image_id": img_id,
                "file_path": file_path,
                "group_id": group_id,
                "annotations": anns,
            }
        )

    return data


# -----------------------------------------------------------------------------
# Decoding Utilities
# -----------------------------------------------------------------------------


def decode_outputs(obj_hm, cls_map, reg_map, K=1200):
    """
    Decodes the model outputs into bounding boxes and class labels.

    Args:
        obj_hm: Objectness heatmap (B, 1, H, W)
        cls_map: Classification map (B, NumClasses, H, W)
        reg_map: Regression map (B, 4, H, W) [offset_x, offset_y, w, h]
        K: Number of top peaks to select

    Returns:
        xs, ys, scores, class_ids
    """
    batch_size, _, height, width = obj_hm.shape

    # 1. Objectness NMS (Max Pooling)
    pad = 1
    hmax = torch.nn.functional.max_pool2d(obj_hm, (3, 3), stride=1, padding=pad)
    keep = (hmax == obj_hm).float()
    obj_hm = obj_hm * keep

    # 2. Top K peaks
    scores = obj_hm.view(batch_size, -1)
    topk_scores, topk_inds = torch.topk(scores, K)

    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()

    # 3. Gather Classification and Regression
    # Transpose to (B, H*W, C)
    reg_map = reg_map.permute(0, 2, 3, 1).view(batch_size, -1, 4)
    cls_map = cls_map.permute(0, 2, 3, 1).view(batch_size, -1, cls_map.size(1))

    # Expand indices for gather
    # reg: (B, K, 4)
    reg_inds = topk_inds.unsqueeze(2).expand(-1, -1, 4)
    topk_reg = torch.gather(reg_map, 1, reg_inds)

    # cls: (B, K, NumClasses)
    cls_inds = topk_inds.unsqueeze(2).expand(-1, -1, cls_map.size(2))
    topk_cls_logits = torch.gather(cls_map, 1, cls_inds)

    # Get specific class predictions
    topk_cls_scores, topk_cls_ids = torch.max(topk_cls_logits, dim=2)

    # 4. Refine Coordinates
    ox = topk_reg[..., 0]
    oy = topk_reg[..., 1]

    topk_xs = topk_xs + ox
    topk_ys = topk_ys + oy

    return topk_xs, topk_ys, topk_scores, topk_cls_ids
