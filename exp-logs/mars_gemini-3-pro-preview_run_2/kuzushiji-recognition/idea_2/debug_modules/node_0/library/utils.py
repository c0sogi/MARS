import os
import pandas as pd
import numpy as np


def get_label_map(
    unicode_map_path, load_cached_data=True, cache_dir="./working/idea_2/"
):
    """
    Creates a mapping from Unicode characters to integer IDs and vice versa.
    Reserves ID 0 for the background class.

    Args:
        unicode_map_path (str): Path to the unicode_translation.csv file.
        load_cached_data (bool): Whether to attempt loading the map from a cached parquet file.
        cache_dir (str): Directory to store and retrieve the cached map.

    Returns:
        tuple: (char_to_int, int_to_char) dictionaries.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "label_map.parquet")

    # Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            char_to_int = dict(zip(df["char"], df["id"]))
            int_to_char = dict(zip(df["id"], df["char"]))
            return char_to_int, int_to_char
        except Exception:
            # Fallback to recomputing if cache is corrupt
            pass

    # Compute from scratch
    if not os.path.exists(unicode_map_path):
        return {}, {}

    uni_df = pd.read_csv(unicode_map_path)
    chars = uni_df["Unicode"].unique()
    # Sort for deterministic ID assignment
    chars = sorted(chars)

    # Assign IDs starting from 1 (0 is reserved for background)
    char_to_int = {c: i + 1 for i, c in enumerate(chars)}
    int_to_char = {i + 1: c for i, c in enumerate(chars)}

    # Save to cache using parquet
    data = [{"char": c, "id": i} for c, i in char_to_int.items()]
    df_out = pd.DataFrame(data)
    df_out.to_parquet(cache_path)

    return char_to_int, int_to_char


def parse_ground_truth(label_str, char_to_int):
    """
    Parses a ground truth label string into bounding boxes and class IDs.

    Args:
        label_str (str): Space separated string in format "Unicode X Y W H ...".
        char_to_int (dict): Mapping from Unicode character to integer ID.

    Returns:
        tuple: (boxes, labels)
            boxes: np.array of shape (N, 4) with format [x1, y1, x2, y2].
            labels: np.array of shape (N,) with integer class IDs.
    """
    if pd.isna(label_str) or not label_str:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

    parts = label_str.split()
    # Basic validation: length must be multiple of 5 (Char, X, Y, W, H)
    if len(parts) % 5 != 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

    boxes = []
    labels = []

    for i in range(0, len(parts), 5):
        char = parts[i]
        try:
            x = int(parts[i + 1])
            y = int(parts[i + 2])
            w = int(parts[i + 3])
            h = int(parts[i + 4])

            if char in char_to_int:
                # Convert from (x, y, w, h) to (x1, y1, x2, y2)
                boxes.append([x, y, x + w, y + h])
                labels.append(char_to_int[char])
        except ValueError:
            continue

    if not boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def format_prediction_string(boxes, labels, int_to_char):
    """
    Formats model predictions into the submission string format.
    Calculates the center point of the bounding box.

    Args:
        boxes (iterable): List or array of bounding boxes [x1, y1, x2, y2].
        labels (iterable): List or array of integer class IDs.
        int_to_char (dict): Mapping from integer ID to Unicode character.

    Returns:
        str: Space separated string "Unicode X_center Y_center ...".
    """
    res = []

    if len(boxes) != len(labels):
        return ""

    for box, label_id in zip(boxes, labels):
        # Handle tensor or numpy scalar values
        if hasattr(label_id, "item"):
            label_id = label_id.item()

        if label_id not in int_to_char:
            continue

        if hasattr(box, "tolist"):
            box = box.tolist()

        x1, y1, x2, y2 = box
        # Calculate center point
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        uni = int_to_char[label_id]
        res.append(f"{uni} {cx} {cy}")

    return " ".join(res)


def calculate_kuzushiji_metrics(gt_df, pred_df):
    """
    Calculates the modified F1 score for the Kuzushiji recognition task.

    Args:
        gt_df (pd.DataFrame): DataFrame containing 'image_id' and 'labels' (GT format: U X Y W H).
        pred_df (pd.DataFrame): DataFrame containing 'image_id' and 'labels' (Pred format: U X Y).

    Returns:
        dict: Dictionary containing 'precision', 'recall', and 'f1' scores.
    """
    # 1. Parse Ground Truth
    gt_data = {}
    for _, row in gt_df.iterrows():
        img_id = row["image_id"]
        label_str = row["labels"]
        items = []
        if pd.notna(label_str) and label_str:
            parts = label_str.split()
            if len(parts) % 5 == 0:
                for i in range(0, len(parts), 5):
                    try:
                        items.append(
                            {
                                "char": parts[i],
                                "x": int(parts[i + 1]),
                                "y": int(parts[i + 2]),
                                "w": int(parts[i + 3]),
                                "h": int(parts[i + 4]),
                                "matched": False,
                            }
                        )
                    except ValueError:
                        pass
        gt_data[img_id] = items

    tp = 0
    fp = 0
    fn = 0

    # 2. Parse Predictions and Match
    pred_map = dict(zip(pred_df["image_id"], pred_df["labels"]))

    all_ids = set(gt_data.keys()).union(set(pred_map.keys()))

    for img_id in all_ids:
        gts = gt_data.get(img_id, [])
        preds_str = pred_map.get(img_id, "")

        preds = []
        if pd.notna(preds_str) and preds_str:
            p_parts = preds_str.split()
            if len(p_parts) % 3 == 0:
                for i in range(0, len(p_parts), 3):
                    try:
                        preds.append(
                            {
                                "char": p_parts[i],
                                "x": int(p_parts[i + 1]),
                                "y": int(p_parts[i + 2]),
                            }
                        )
                    except ValueError:
                        pass

        # Greedy matching strategy
        for p in preds:
            match_found = False
            for g in gts:
                if g["matched"]:
                    continue

                # Check label match
                if p["char"] == g["char"]:
                    # Check spatial containment (center point inside GT box)
                    if (g["x"] <= p["x"] <= g["x"] + g["w"]) and (
                        g["y"] <= p["y"] <= g["y"] + g["h"]
                    ):
                        g["matched"] = True
                        match_found = True
                        break

            if match_found:
                tp += 1
            else:
                fp += 1

        # False Negatives are any GTs that were not matched
        fn += sum(1 for g in gts if not g["matched"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}
