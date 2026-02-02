import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.model import WSDN_ABS
from library.dataset import InferenceDataset, load_fragment_data
from library.utils import rle_encode, f05_score


def apply_tta(model, inputs):
    """
    Applies Test-Time Augmentation (8 views: 4 rotations x 2 flips).
    Args:
        model: The neural network.
        inputs: Batch of input volumes (B, Z, H, W).
    Returns:
        Averaged logits or probabilities (B, 1, H, W).
    """
    # We will accumulate probabilities
    accumulated_probs = None
    count = 0

    # D4 Group: combinations of flips and rotations
    # fl: flip dimension (None, 2=H, 3=W) - actually dims are (B, C, H, W) so 2 and 3
    # k: number of 90-degree rotations

    # We iterate through 2 flip states (no flip, flip vertical) and 4 rotations
    # Note: Flip V + Rotations covers all 8 symmetries of a square

    flips = [None, 2]  # 2 is H dimension (vertical flip)
    rotations = [0, 1, 2, 3]

    for flip_dim in flips:
        for k in rotations:
            # 1. Augment
            x = inputs.clone()
            if flip_dim is not None:
                x = torch.flip(x, dims=[flip_dim])
            if k > 0:
                x = torch.rot90(x, k, dims=[2, 3])

            # 2. Predict
            with torch.no_grad():
                outputs = model(x)
                # We use the mask head. Apply sigmoid to get probabilities.
                probs = torch.sigmoid(outputs["mask"])

            # 3. Inverse Augment
            if k > 0:
                # Inverse rotation is -k or 4-k
                probs = torch.rot90(probs, -k, dims=[2, 3])
            if flip_dim is not None:
                probs = torch.flip(probs, dims=[flip_dim])

            # 4. Accumulate
            if accumulated_probs is None:
                accumulated_probs = probs
            else:
                accumulated_probs += probs
            count += 1

    return accumulated_probs / count


def predict_fragment(model, fragment_id, split, device):
    """
    Performs sliding window inference on a single fragment.
    Reconstructs the full probability map.
    """
    dataset = InferenceDataset(
        split=split, fragment_id=fragment_id, load_cached_data=True
    )
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Get fragment dimensions from the first item in dataset metadata or by loading mask
    # We can get it from the dataset's tile information
    if len(dataset) == 0:
        return None

    # Dimensions are stored in the dataset tiles
    h_img = dataset.tiles[0]["h"]
    w_img = dataset.tiles[0]["w"]

    # Initialize accumulators on CPU to save GPU memory
    prob_map = torch.zeros((h_img, w_img), dtype=torch.float32)
    count_map = torch.zeros((h_img, w_img), dtype=torch.float32)

    model.eval()

    for batch_idx, (volumes, meta) in enumerate(loader):
        volumes = volumes.to(device)

        # Predict
        if Config.USE_TTA:
            probs = apply_tta(model, volumes)
        else:
            with torch.no_grad():
                outputs = model(volumes)
                probs = torch.sigmoid(outputs["mask"])

        probs = probs.cpu()

        # Place patches back into full image
        # meta is a dict of lists (collated)
        ys = meta["y"].numpy()
        xs = meta["x"].numpy()

        for i in range(len(ys)):
            y, x = ys[i], xs[i]
            p = probs[i, 0, :, :]  # (H_patch, W_patch)

            # Determine crop coords (handling padding logic from dataset)
            # The dataset returns (patch_size, patch_size).
            # The tile starts at (y, x) in the *original* image space.
            # However, the dataset pads the volume before cropping.
            # InferenceDataset logic:
            # vol_padded = pad(vol)
            # crop = vol_padded[y:y+size, x:x+size]
            # This corresponds to original image coords: y-pad to y+size-pad
            # Wait, let's check InferenceDataset implementation in provided code.
            # vol_padded = pad(vol, pad)
            # sliding window y from 0 to h, x from 0 to w.
            # crop = vol_padded[:, y:y+size, x:x+size]
            # Center of crop in padded image is y + size/2, x + size/2.
            # In original image, this corresponds to y, x (top-left of window).
            # Actually, looking at `InferenceDataset`:
            # y iterates 0..h, x iterates 0..w.
            # crop comes from `vol_padded`.
            # `vol_padded` has `pad` amount of padding.
            # So `vol_padded[pad]` corresponds to `vol[0]`.
            # If y=0, we crop `vol_padded[0:256]`. This covers `vol[-128:128]` roughly.
            # The valid area of the prediction we want to place into the map:
            # The patch corresponds to `y - pad` to `y + size - pad` in original coords.

            pad = Config.PATCH_SIZE // 2

            # Coordinates in probability map
            y_start_map = max(0, y - pad)
            y_end_map = min(h_img, y + Config.PATCH_SIZE - pad)
            x_start_map = max(0, x - pad)
            x_end_map = min(w_img, x + Config.PATCH_SIZE - pad)

            # Coordinates within the patch
            # If y - pad < 0, we need to crop the start of the patch
            y_start_patch = pad - y if (y - pad) < 0 else 0
            x_start_patch = pad - x if (x - pad) < 0 else 0

            # Calculate lengths
            h_len = y_end_map - y_start_map
            w_len = x_end_map - x_start_map

            # Extract valid region from patch
            patch_valid = p[
                y_start_patch : y_start_patch + h_len,
                x_start_patch : x_start_patch + w_len,
            ]

            # Accumulate
            prob_map[y_start_map:y_end_map, x_start_map:x_end_map] += patch_valid
            count_map[y_start_map:y_end_map, x_start_map:x_end_map] += 1.0

    # Average
    mask = count_map > 0
    prob_map[mask] /= count_map[mask]

    return prob_map.numpy()


def find_best_threshold(model, device):
    """
    Optimizes the threshold on the validation set.
    """
    print("Optimizing threshold on validation set...")

    val_meta_path = Config.METADATA_DIR / "val.csv"
    if not val_meta_path.exists():
        print("Validation metadata not found. Using default threshold 0.5.")
        return 0.5

    df_val = pd.read_csv(val_meta_path)
    if df_val.empty:
        print("Validation set empty. Using default threshold 0.5.")
        return 0.5

    # We will concatenate all pixels from all val fragments to find global best
    all_preds = []
    all_labels = []

    for _, row in df_val.iterrows():
        fid = str(row["fragment_id"])
        print(f"Predicting validation fragment {fid}...")

        # Get Prediction
        pred_map = predict_fragment(model, fid, "val", device)
        if pred_map is None:
            continue

        # Get Ground Truth
        # We need to load the inklabel.
        # load_fragment_data returns (vol, mask, label)
        _, mask, label = load_fragment_data(
            fid,
            "val",
            row["surface_volume_path"],
            row["mask_path"],
            row["inklabels_path"],
            load_cached_data=True,
        )

        # Apply mask to prediction to ignore outside areas
        pred_map = pred_map * mask

        # Flatten and store
        # We only care about pixels inside the valid mask for scoring usually,
        # but the metric is global. However, to save memory and be precise,
        # let's just use the valid mask area or the whole thing.
        # Given the metric description, we evaluate on the whole image.

        all_preds.append(pred_map.flatten())
        all_labels.append(label.flatten())

    if not all_preds:
        return 0.5

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    best_score = -1
    best_th = 0.5

    thresholds = np.arange(
        Config.THRESHOLD_START, Config.THRESHOLD_END, Config.THRESHOLD_STEP
    )

    print(f"Searching thresholds: {thresholds}")

    # Use a subset for speed if array is huge
    if len(all_preds) > 10_000_000:
        indices = np.random.choice(len(all_preds), 10_000_000, replace=False)
        sample_preds = all_preds[indices]
        sample_labels = all_labels[indices]
    else:
        sample_preds = all_preds
        sample_labels = all_labels

    for th in thresholds:
        score = f05_score(sample_preds, sample_labels, threshold=th)
        if score > best_score:
            best_score = score
            best_th = th

    print(f"Best Threshold: {best_th:.3f} with F0.5: {best_score:.4f}")
    return best_th


def inference():
    device = torch.device(Config.DEVICE)
    model_path = Config.WORKING_DIR / "best_model.pth"

    # 1. Load Model
    print(f"Loading model from {model_path}...")
    model = WSDN_ABS(
        in_channels=Config.Z_DIM,
        model_channels=Config.MODEL_CHANNELS,
        dilation_rates=Config.DILATION_RATES,
    )

    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Warning: Best model not found. Using random weights (for debugging).")

    model = model.to(device)
    model.eval()

    # 2. Find Threshold
    best_threshold = find_best_threshold(model, device)

    # Save threshold for reference
    with open(Config.WORKING_DIR / "threshold.txt", "w") as f:
        f.write(str(best_threshold))

    # 3. Predict Test Set
    test_meta_path = Config.METADATA_DIR / "test.csv"
    if not test_meta_path.exists():
        print("Test metadata not found.")
        return

    df_test = pd.read_csv(test_meta_path)
    submission_data = []

    for _, row in df_test.iterrows():
        fid = str(row["fragment_id"])
        print(f"Processing test fragment {fid}...")

        # Predict
        prob_map = predict_fragment(model, fid, "test", device)

        if prob_map is None:
            print(f"Failed to predict fragment {fid}")
            submission_data.append({"Id": fid, "Predicted": ""})
            continue

        # Load valid mask to clean up prediction
        _, mask, _ = load_fragment_data(
            fid,
            "test",
            row["surface_volume_path"],
            row["mask_path"],
            None,
            load_cached_data=True,
        )

        # Binarize
        binary_map = (prob_map > best_threshold).astype(np.uint8)

        # Mask out invalid areas (background)
        if mask is not None:
            if mask.shape != binary_map.shape:
                mask = cv2.resize(
                    mask,
                    (binary_map.shape[1], binary_map.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            binary_map = binary_map * mask

        # RLE Encode
        rle = rle_encode(binary_map)
        submission_data.append({"Id": fid, "Predicted": rle})

    # 4. Save Submission
    submission_df = pd.DataFrame(submission_data)
    submission_path = "./submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
