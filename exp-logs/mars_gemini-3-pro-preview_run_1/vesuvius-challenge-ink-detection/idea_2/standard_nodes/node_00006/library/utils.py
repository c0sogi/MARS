import os
import numpy as np
import pandas as pd
import torch
import cv2
from library.config import Config


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    Args:
        mask (np.ndarray): Binary mask (0 or 1).

    Returns:
        str: Space-delimited string of 'start length' pairs.
    """
    # Flatten the mask (row-major order as per requirements)
    pixels = mask.flatten()

    # Pad with zeros at both ends to detect changes at the boundaries
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array contains indices of starts and ends.
    # Even indices are starts, odd indices are ends.
    # Calculate lengths: end - start
    runs[1::2] -= runs[::2]

    # Convert to string
    return " ".join(str(x) for x in runs)


def calculate_fbeta(pred_mask, true_mask, beta=0.5):
    """
    Calculates the F-beta score.

    Args:
        pred_mask (np.ndarray): Predicted binary mask.
        true_mask (np.ndarray): Ground truth binary mask.
        beta (float): Beta value for F-score (default 0.5).

    Returns:
        float: F-beta score.
    """
    # Flatten arrays to ensure 1D calculation
    p = pred_mask.reshape(-1).astype(bool)
    t = true_mask.reshape(-1).astype(bool)

    tp = (p & t).sum()
    fp = (p & ~t).sum()
    fn = (~p & t).sum()

    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def optimize_threshold(val_preds, val_labels, start=None, end=None, step=None):
    """
    Finds the optimal threshold that maximizes the F0.5 score on validation data.

    Args:
        val_preds (list of np.ndarray): List of predicted probability maps.
        val_labels (list of np.ndarray): List of ground truth binary masks.
        start (float): Start of threshold range.
        end (float): End of threshold range.
        step (float): Step size.

    Returns:
        tuple: (best_threshold, best_score)
    """
    start = start or Config.THRESHOLD_START
    end = end or Config.THRESHOLD_END
    step = step or Config.THRESHOLD_STEP

    # Flatten all data once to speed up the loop
    # We can compute global TP/FP/FN stats over the entire validation set
    y_true = np.concatenate([l.flatten() for l in val_labels])
    y_pred_probs = np.concatenate([p.flatten() for p in val_preds])

    thresholds = np.arange(start, end + 1e-6, step)
    best_score = 0.0
    best_thresh = 0.5

    for th in thresholds:
        y_pred = y_pred_probs >= th
        score = calculate_fbeta(y_pred, y_true, beta=0.5)

        if score > best_score:
            best_score = score
            best_thresh = th

    return best_thresh, best_score


def load_patch_volume(surface_path, x, y, w, h):
    """
    Loads the 3D volume stack for a specific patch.

    Args:
        surface_path (str): Relative path to the surface volume directory.
        x, y, w, h (int): Patch coordinates and dimensions.

    Returns:
        np.ndarray: 3D volume of shape (Z_DIM, h, w).
    """
    volume = []
    base_path = os.path.join(Config.INPUT_DIR, surface_path)

    # Load all slices
    for i in range(Config.Z_DIM):
        filename = f"{i:02d}.tif"
        file_path = os.path.join(base_path, filename)

        if os.path.exists(file_path):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                crop = img[y : y + h, x : x + w]
            else:
                crop = np.zeros((h, w), dtype=np.uint8)
        else:
            crop = np.zeros((h, w), dtype=np.uint8)

        volume.append(crop)

    return np.stack(volume, axis=0)


def generate_submission(model, device, threshold=0.5):
    """
    Generates predictions for the test set and saves the submission CSV.

    Args:
        model (torch.nn.Module): Trained model.
        device (torch.device): Device to run inference on.
        threshold (float): Binarization threshold.
    """
    if not os.path.exists(Config.TEST_METADATA_PATH):
        print("Test metadata not found. Skipping submission generation.")
        return

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    fragment_ids = df_test["fragment_id"].unique()

    submission_rows = []
    model.eval()

    for fid in fragment_ids:
        frag_df = df_test[df_test["fragment_id"] == fid]

        # Determine canvas size for reconstruction
        max_x = (frag_df["x"] + frag_df["w"]).max()
        max_y = (frag_df["y"] + frag_df["h"]).max()

        full_prob_map = np.zeros((max_y, max_x), dtype=np.float32)
        count_map = np.zeros((max_y, max_x), dtype=np.float32)

        batch_data = []

        def process_batch(batch):
            # batch is list of (vol, x, y, w, h)
            vols = [b[0] for b in batch]
            coords = [b[1:] for b in batch]

            # Stack into tensor: (B, Z, H, W)
            input_tensor = torch.from_numpy(np.stack(vols)).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)

            for i, (bx, by, bw, bh) in enumerate(coords):
                p = probs[i, 0]  # (512, 512)
                # Crop back to original dimensions if padded
                p = p[:bh, :bw]
                full_prob_map[by : by + bh, bx : bx + bw] += p
                count_map[by : by + bh, bx : bx + bw] += 1.0

        # Iterate over patches
        for _, row in frag_df.iterrows():
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]

            # Load volume
            vol = load_patch_volume(row["surface_volume_path"], x, y, w, h)

            # Pad to PATCH_SIZE if necessary (for model consistency)
            pad_h = Config.PATCH_SIZE - h
            pad_w = Config.PATCH_SIZE - w
            if pad_h > 0 or pad_w > 0:
                vol = np.pad(
                    vol,
                    ((0, 0), (0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

            # Normalize
            vol = vol.astype(np.float32) / 255.0

            batch_data.append((vol, x, y, w, h))

            # Run batch inference
            if len(batch_data) >= Config.BATCH_SIZE:
                process_batch(batch_data)
                batch_data = []

        # Process remaining
        if batch_data:
            process_batch(batch_data)

        # Average overlapping predictions
        mask = count_map > 0
        full_prob_map[mask] /= count_map[mask]

        # Apply threshold
        binary_mask = (full_prob_map > threshold).astype(np.uint8)

        # Encode
        rle = rle_encode(binary_mask)
        submission_rows.append({"Id": fid, "Predicted": rle})

    # Save submission
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
