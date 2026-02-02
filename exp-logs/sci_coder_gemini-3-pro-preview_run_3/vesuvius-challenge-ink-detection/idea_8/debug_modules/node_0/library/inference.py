import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.model import FRDUNet
from library.dataset import InkDataset
from library.utils import rle_encode, fbeta_score, load_inklabels


def apply_tta(model, inputs):
    """
    Applies Test-Time Augmentation (TTA) using the Dihedral Group D4.
    Averages predictions across 4 rotations and their horizontal flips (8 views).

    Args:
        model: The PyTorch model.
        inputs: Input tensor of shape (B, Z, H, W).

    Returns:
        torch.Tensor: Averaged probability map of shape (B, 1, H, W).
    """
    preds = []

    # 1. Standard Rotations (0, 90, 180, 270)
    for k in [0, 1, 2, 3]:
        # Rotate input
        x = torch.rot90(inputs, k=k, dims=[2, 3])

        # Predict
        logits = model(x)
        probs = torch.sigmoid(logits)

        # Rotate output back
        probs = torch.rot90(probs, k=-k, dims=[2, 3])
        preds.append(probs)

    # 2. Flips + Rotations (The other 4 symmetries)
    # Flip Horizontal first
    inputs_f = torch.flip(inputs, dims=[3])

    for k in [0, 1, 2, 3]:
        # Rotate flipped input
        x = torch.rot90(inputs_f, k=k, dims=[2, 3])

        # Predict
        logits = model(x)
        probs = torch.sigmoid(logits)

        # Rotate output back
        probs = torch.rot90(probs, k=-k, dims=[2, 3])

        # Flip back
        probs = torch.flip(probs, dims=[3])
        preds.append(probs)

    # Stack and Average
    avg_preds = torch.stack(preds).mean(dim=0)
    return avg_preds


def predict_tile(model, volume_patch):
    """
    Runs inference on a single tile or batch of tiles.

    Args:
        model: The PyTorch model.
        volume_patch: Tensor of shape (Z, H, W) or (B, Z, H, W).

    Returns:
        torch.Tensor: Probability map.
    """
    # Ensure batch dimension
    if volume_patch.dim() == 3:
        volume_patch = volume_patch.unsqueeze(0)

    device = next(model.parameters()).device
    volume_patch = volume_patch.to(device)

    if Config.TTA_ENABLED:
        return apply_tta(model, volume_patch)
    else:
        with torch.no_grad():
            logits = model(volume_patch)
            return torch.sigmoid(logits)


def predict_full_mask(model, split, load_cached_data=True):
    """
    Generates full-resolution probability maps for all fragments in the specified split.

    Args:
        model: The PyTorch model.
        split: 'val' or 'test'.
        load_cached_data: Whether to use cached dataset artifacts.

    Returns:
        tuple: (fragment_preds, fragment_masks)
            fragment_preds: Dict mapping fragment_id -> np.array (H, W) probabilities.
            fragment_masks: Dict mapping fragment_id -> np.array (H, W) binary valid masks.
    """
    device = torch.device(Config.DEVICE)

    # Initialize Dataset and Loader
    dataset = InkDataset(split=split, load_cached_data=load_cached_data)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Buffers
    fragment_preds = {}
    fragment_masks = {}

    for frag in dataset.fragments:
        fid = str(frag["id"])
        h, w = frag["mask"].shape
        fragment_preds[fid] = np.zeros((h, w), dtype=np.float32)
        fragment_masks[fid] = frag["mask"]

    model.eval()
    with torch.no_grad():
        for batch in loader:
            volumes = batch["volume"].to(device)
            f_ids = batch["fragment_id"]
            ys = batch["y"]
            xs = batch["x"]

            # Inference (with TTA if enabled)
            if Config.TTA_ENABLED:
                probs = apply_tta(model, volumes)
            else:
                logits = model(volumes)
                probs = torch.sigmoid(logits)

            probs = probs.squeeze(1).cpu().numpy()

            # Place patches into global maps
            for i in range(len(f_ids)):
                fid = f_ids[i]
                y = ys[i].item()
                x = xs[i].item()
                prob_patch = probs[i]

                h_p, w_p = prob_patch.shape

                # Assign prediction
                # Since stride equals patch size in Config, we can directly assign.
                # If overlapping, we would need a counter map and addition.
                fragment_preds[fid][y : y + h_p, x : x + w_p] = prob_patch

    return fragment_preds, fragment_masks


def optimize_threshold(model, load_cached_data=True):
    """
    Finds the optimal binarization threshold using the validation set.

    Args:
        model: The PyTorch model.
        load_cached_data: Whether to use cached data.

    Returns:
        float: The optimal threshold.
    """
    print("Optimizing threshold on validation set...")

    # Check if validation metadata exists
    if not Config.VAL_METADATA_PATH.exists():
        print("Validation metadata not found. Defaulting to 0.5.")
        return 0.5

    # Generate Predictions
    preds_map, masks_map = predict_full_mask(
        model, split="val", load_cached_data=load_cached_data
    )

    # Load Ground Truth
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    all_preds = []
    all_labels = []

    for _, row in df_val.iterrows():
        fid = str(row["fragment_id"])
        if fid not in preds_map:
            continue

        # Load Label
        label = load_inklabels(fid, "val", df_val, load_cached_data=load_cached_data)
        if label is None:
            continue

        mask = masks_map[fid]

        # Flatten and select only valid pixels
        valid_indices = mask > 0

        p_flat = preds_map[fid][valid_indices]
        l_flat = label[valid_indices]

        all_preds.append(p_flat)
        all_labels.append(l_flat)

    if not all_preds:
        print("No validation data available for thresholding. Defaulting to 0.5.")
        return 0.5

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Linear Search for Best Threshold
    best_score = -1.0
    best_th = 0.5

    thresholds = np.arange(
        Config.THRESHOLD_START, Config.THRESHOLD_END, Config.THRESHOLD_STEP
    )

    for th in thresholds:
        score = fbeta_score(all_preds, all_labels, beta=0.5, threshold=th)
        if score > best_score:
            best_score = score
            best_th = th

    print(f"Best Validation F0.5 Score: {best_score}")
    print(f"Optimal Threshold: {best_th}")

    return best_th


def generate_submission(load_cached_data=True):
    """
    Main inference pipeline:
    1. Loads best model.
    2. Optimizes threshold on validation set (if available).
    3. Predicts on test set.
    4. Generates RLE submission file.
    """
    device = torch.device(Config.DEVICE)

    # Load Model
    model = FRDUNet().to(device)
    if not Config.BEST_MODEL_PATH.exists():
        print(f"Error: Best model not found at {Config.BEST_MODEL_PATH}")
        return

    print(f"Loading model from {Config.BEST_MODEL_PATH}...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # 1. Determine Threshold
    if Config.VAL_METADATA_PATH.exists():
        threshold = optimize_threshold(model, load_cached_data=load_cached_data)

        # Cache threshold for reference
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        with open(Config.WORKING_DIR / "threshold.txt", "w") as f:
            f.write(str(threshold))
    else:
        print("Skipping threshold optimization (no val set). Using 0.5.")
        threshold = 0.5

    # 2. Predict Test Set
    print("Generating predictions for test set...")
    preds_map, masks_map = predict_full_mask(
        model, split="test", load_cached_data=load_cached_data
    )

    submission_data = []

    for fid in sorted(preds_map.keys()):
        pred = preds_map[fid]
        mask = masks_map[fid]

        # Binarize
        pred_bin = (pred > threshold).astype(np.uint8)

        # Apply Valid Mask (Zero out invalid areas)
        pred_bin = pred_bin * mask

        # RLE Encode
        rle = rle_encode(pred_bin)
        submission_data.append({"Id": fid, "Predicted": rle})

    # 3. Save Submission
    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
