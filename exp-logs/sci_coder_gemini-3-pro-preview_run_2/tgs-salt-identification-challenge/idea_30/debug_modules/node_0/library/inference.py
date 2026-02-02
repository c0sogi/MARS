import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from library.utils import rle_encode
from library.dataset import get_depth_stats, ORIG_SIZE, TARGET_SIZE
from library.engine import validate_one_epoch


def calculate_map_vectorized(preds_bool, targets_bool):
    """
    Calculates the Mean Average Precision at IoU thresholds 0.5:0.95:0.05
    using vectorized operations.

    Args:
        preds_bool (np.ndarray): Binary predictions (N, H, W).
        targets_bool (np.ndarray): Binary targets (N, H, W).

    Returns:
        float: The mAP score.
    """
    N = preds_bool.shape[0]

    # Flatten spatial dimensions for IoU calculation
    p_flat = preds_bool.reshape(N, -1)
    t_flat = targets_bool.reshape(N, -1)

    # Check for empty masks
    pred_empty = p_flat.sum(axis=1) == 0
    gt_empty = t_flat.sum(axis=1) == 0

    # Calculate IoU for non-empty pairs
    # We only need to compute IoU where both are non-empty
    # But for vectorization, we compute all and mask later
    intersection = (p_flat & t_flat).sum(axis=1)
    union = (p_flat | t_flat).sum(axis=1)

    # Avoid division by zero
    iou = np.zeros(N, dtype=np.float32)
    valid_union = union > 0
    iou[valid_union] = intersection[valid_union] / union[valid_union]

    # Define thresholds
    thresholds = np.arange(0.5, 0.95 + 1e-6, 0.05)

    # Calculate score per image
    # Case 1: Both empty -> Score 1.0
    score_both_empty = (pred_empty & gt_empty).astype(np.float32)

    # Case 2: One empty, one not -> Score 0.0 (handled by default init of scores)

    # Case 3: Both non-empty -> Score based on IoU matches
    # (N, 1) > (1, n_thresh) -> (N, n_thresh)
    matches = iou[:, None] > thresholds[None, :]
    # Mean over thresholds -> (N,)
    score_iou = matches.mean(axis=1)

    # Combine scores
    # We use a mask for where both are non-empty
    both_non_empty = (~pred_empty) & (~gt_empty)

    final_scores = np.zeros(N, dtype=np.float32)
    final_scores[pred_empty & gt_empty] = 1.0
    final_scores[both_non_empty] = score_iou[both_non_empty]

    return final_scores.mean()


def optimize_threshold(model, val_loader, device):
    """
    Finds the optimal binarization threshold using the validation set.
    Uses standard inference (with true depths) for calibration.

    Args:
        model: PyTorch model.
        val_loader: Validation DataLoader.
        device: Device.

    Returns:
        float: Optimal threshold.
    """
    print("Optimizing threshold on validation set...")

    # Use engine's validation function to get raw logits
    # We pass a dummy criterion as we don't care about loss here
    dummy_criterion = torch.nn.BCEWithLogitsLoss()
    _, val_logits, val_targets, _ = validate_one_epoch(
        model, val_loader, dummy_criterion, device, epoch=0
    )

    # Squeeze channel dim if present: (N, 1, H, W) -> (N, H, W)
    if val_logits.ndim == 4:
        val_logits = val_logits.squeeze(1)
    if val_targets.ndim == 4:
        val_targets = val_targets.squeeze(1)

    # Crop to original size (101x101)
    # Padding logic from dataset.py: Center padding
    pad_t = (TARGET_SIZE - ORIG_SIZE) // 2
    pad_l = (TARGET_SIZE - ORIG_SIZE) // 2

    logits_cropped = val_logits[:, pad_t : pad_t + ORIG_SIZE, pad_l : pad_l + ORIG_SIZE]
    targets_cropped = val_targets[
        :, pad_t : pad_t + ORIG_SIZE, pad_l : pad_l + ORIG_SIZE
    ]

    # Convert to probabilities
    probs = 1.0 / (1.0 + np.exp(-logits_cropped))

    # Binarize targets (they should be 0 or 1 already, but safe to cast)
    targets_bool = targets_cropped > 0.5

    # Grid search for threshold
    best_threshold = 0.5
    best_score = -1.0

    # Search range: 0.1 to 0.9
    thresholds = np.linspace(0.1, 0.9, 17)

    for th in thresholds:
        preds_bool = probs > th
        score = calculate_map_vectorized(preds_bool, targets_bool)

        if score > best_score:
            best_score = score
            best_threshold = th

    print(f"Best Threshold: {best_threshold:.4f} | Validation mAP: {best_score:.4f}")
    return best_threshold


def predict_depth_scan(model, test_loader, device, num_depths=10):
    """
    Performs Marginalized Depth-Scan Inference on the test set.

    Args:
        model: PyTorch model.
        test_loader: Test DataLoader.
        device: Device.
        num_depths: Number of depth steps to marginalize over.

    Returns:
        list: List of dicts {'id': str, 'prob_map': np.ndarray}.
    """
    model.eval()

    # 1. Determine Depth Range from Training Metadata
    # We read metadata/train.csv to find global min/max z
    train_meta_path = "./metadata/train.csv"
    if os.path.exists(train_meta_path):
        df_train = pd.read_csv(train_meta_path)
        z_min = df_train["z"].min()
        z_max = df_train["z"].max()
    else:
        # Fallback to dataset description values
        z_min = 51
        z_max = 959

    # 2. Get Normalization Stats
    d_mean, d_std = get_depth_stats(load_cached_data=True)

    # 3. Create Scan Depths
    scan_depths_raw = np.linspace(z_min, z_max, num_depths)
    scan_depths_norm = (scan_depths_raw - d_mean) / d_std
    scan_depths_tensor = torch.tensor(scan_depths_norm, dtype=torch.float32).to(device)

    # Padding info for cropping
    pad_t = (TARGET_SIZE - ORIG_SIZE) // 2
    pad_l = (TARGET_SIZE - ORIG_SIZE) // 2

    results = []

    with torch.no_grad():
        for images, _, ids in test_loader:
            images = images.to(device, dtype=torch.float32)
            batch_size = images.size(0)

            # Accumulator: [B, 1, H, W]
            accum_probs = torch.zeros(
                (batch_size, 1, TARGET_SIZE, TARGET_SIZE), device=device
            )

            # Iterate over latent depth variable
            for z_val in scan_depths_tensor:
                # Create batch of depths
                z_batch = z_val.repeat(batch_size).view(batch_size, 1)

                # --- Original Forward ---
                logits = model(images, z_batch)
                probs = torch.sigmoid(logits)
                accum_probs += probs

                # --- TTA: Horizontal Flip ---
                # Flip width dimension (dim 3)
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped, z_batch)
                probs_flipped = torch.sigmoid(logits_flipped)
                # Flip back
                probs_flipped_back = torch.flip(probs_flipped, dims=[3])
                accum_probs += probs_flipped_back

            # Average over (num_depths * 2 TTA passes)
            avg_probs = accum_probs / (num_depths * 2)

            # Move to CPU
            avg_probs_np = avg_probs.cpu().numpy()

            # Process batch
            for i, img_id in enumerate(ids):
                # Extract single image prob map (1, H, W) -> (H, W)
                pm = avg_probs_np[i, 0]

                # Crop to 101x101
                pm_cropped = pm[pad_t : pad_t + ORIG_SIZE, pad_l : pad_l + ORIG_SIZE]

                results.append({"id": img_id, "prob_map": pm_cropped})

    return results


def generate_submission(
    model, test_loader, threshold, device, save_path="./submission/submission.csv"
):
    """
    Generates the final submission file.

    Args:
        model: PyTorch model.
        test_loader: Test DataLoader.
        threshold: Optimal binarization threshold.
        device: Device.
        save_path: Output CSV path.
    """
    print(f"Generating submission with threshold {threshold:.4f}...")

    # Run Marginalized Depth-Scan
    predictions = predict_depth_scan(model, test_loader, device)

    submission_data = []

    for item in predictions:
        img_id = item["id"]
        prob_map = item["prob_map"]

        # Apply Threshold
        mask = (prob_map > threshold).astype(np.uint8)

        # RLE Encode
        rle = rle_encode(mask)
        submission_data.append([img_id, rle])

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_sub = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path} with {len(df_sub)} rows.")
