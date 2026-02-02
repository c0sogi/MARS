import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from library.config import CFG


def rle_encode(img):
    """
    Args:
        img: numpy array, 1 - mask, 0 - background
    Returns:
        rle: run-length encoded string
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Args:
        mask_rle: run-length encoded string
        shape: (height, width) of array to return
    Returns:
        mask: numpy array, 1 - mask, 0 - background
    """
    if pd.isna(mask_rle) or mask_rle == "" or mask_rle == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def compute_dice(y_true, y_pred):
    """
    Compute Dice coefficient between two binary masks/volumes.
    """
    # Flatten to ensure 1D
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union == 0:
        return 1.0

    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection) / union


def compute_hausdorff_3d(y_true_vol, y_pred_vol, spacing_norm=(1.0, 1.0, 1.0)):
    """
    Compute 3D Hausdorff distance with normalized coordinates.
    Args:
        y_true_vol: (D, H, W) binary mask
        y_pred_vol: (D, H, W) binary mask
        spacing_norm: tuple (z_norm, h_norm, w_norm) to normalize coordinates.
                      Coordinates will be multiplied by these factors.
                      Usually 1/D, 1/H, 1/W to map to [0,1].
    Returns:
        hd: Directed Hausdorff distance (max of both directions)
    """
    # Extract coordinates of non-zero pixels: (z, y, x)
    true_points = np.argwhere(y_true_vol)
    pred_points = np.argwhere(y_pred_vol)

    # Handle empty cases
    if len(true_points) == 0 and len(pred_points) == 0:
        return 0.0
    if len(true_points) == 0:
        return 1.0  # Max penalty
    if len(pred_points) == 0:
        return 1.0  # Max penalty

    # Normalize coordinates
    # argwhere returns (z, y, x)
    # spacing_norm is (scale_z, scale_y, scale_x)
    true_points = true_points * np.array(spacing_norm)
    pred_points = pred_points * np.array(spacing_norm)

    # Compute distances using KDTree
    # A to B
    tree_pred = cKDTree(pred_points)
    d_AB, _ = tree_pred.query(true_points, k=1)
    max_d_AB = np.max(d_AB)

    # B to A
    tree_true = cKDTree(true_points)
    d_BA, _ = tree_true.query(pred_points, k=1)
    max_d_BA = np.max(d_BA)

    return max(max_d_AB, max_d_BA)


def compute_metrics(df_pred, df_true):
    """
    Compute competition metrics (Dice + Hausdorff).

    Args:
        df_pred: DataFrame with ['id', 'class', 'predicted']
        df_true: DataFrame with ['id', 'case', 'day', 'slice', 'height', 'width', 'large_bowel', 'small_bowel', 'stomach']
                 (Metadata format, containing ground truth masks if available, or joined externally)
    Returns:
        dict with keys: 'dice', 'hausdorff', 'score'
    """

    # Prepare DataFrames
    # Melt df_true to long format for easier merging
    id_vars = ["id", "case", "day", "slice", "height", "width"]
    class_vars = ["large_bowel", "small_bowel", "stomach"]

    # Ensure class columns exist in df_true (fill with empty if missing)
    for c in class_vars:
        if c not in df_true.columns:
            df_true[c] = ""

    df_true_long = df_true.melt(
        id_vars=id_vars, value_vars=class_vars, var_name="class", value_name="gt_rle"
    )

    # Merge with predictions
    # df_pred should have ['id', 'class', 'predicted']
    df_merged = pd.merge(df_true_long, df_pred, on=["id", "class"], how="left")

    # Fill NaN predictions/GT with empty string
    df_merged["predicted"] = df_merged["predicted"].fillna("")
    df_merged["gt_rle"] = df_merged["gt_rle"].fillna("")

    # Group by Case+Day
    groups = df_merged.groupby(["case", "day"])

    dice_scores = []
    hd_scores = []

    # Iterate over each scan (Case + Day)
    for (case, day), group in groups:
        # Sort by slice index to ensure correct Z ordering
        # slice might be string '0001', convert to int for sorting
        group = group.copy()
        group["slice_idx"] = group["slice"].astype(int)
        group = group.sort_values("slice_idx")

        # Get dimensions
        # Assuming all slices in a scan have same h, w
        if len(group) == 0:
            continue

        h = int(group.iloc[0]["height"])
        w = int(group.iloc[0]["width"])
        d = group["slice_idx"].nunique()  # Number of slices

        # Normalization factors for HD: (1/d, 1/h, 1/w)
        # This maps the volume to a unit cube [0,1]^3
        norm_factors = (1.0 / d if d > 0 else 1.0, 1.0 / h, 1.0 / w)

        # Process each class separately
        for cls in class_vars:
            cls_group = group[group["class"] == cls]

            if len(cls_group) == 0:
                continue

            # Reconstruct volumes
            # Pre-allocate volumes (D, H, W)
            vol_true = np.zeros((d, h, w), dtype=np.uint8)
            vol_pred = np.zeros((d, h, w), dtype=np.uint8)

            # Map slices to volume indices 0..d-1
            # cls_group is already sorted by slice_idx
            # We assume slices are contiguous or we just stack them in order
            for i, (_, row) in enumerate(cls_group.iterrows()):
                if i >= d:
                    break
                vol_true[i] = rle_decode(row["gt_rle"], (h, w))
                vol_pred[i] = rle_decode(row["predicted"], (h, w))

            # Compute Dice
            dice = compute_dice(vol_true, vol_pred)
            dice_scores.append(dice)

            # Compute HD
            hd = compute_hausdorff_3d(vol_true, vol_pred, norm_factors)
            hd_scores.append(hd)

    # Aggregate
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    mean_hd = np.mean(hd_scores) if hd_scores else 0.0

    # Score: 0.4 * Dice + 0.6 * (1 - HD)
    # Ensure HD score is bounded [0, 1] (clip HD at 1.0)
    score_hd = 1.0 - min(1.0, mean_hd)

    score = 0.4 * mean_dice + 0.6 * score_hd

    return {"dice": mean_dice, "hausdorff": mean_hd, "score": score}
