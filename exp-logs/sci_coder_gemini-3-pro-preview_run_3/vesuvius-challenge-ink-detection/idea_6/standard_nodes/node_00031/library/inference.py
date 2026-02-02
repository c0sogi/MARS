import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything, rle_encoding, load_volume
from library.model import SGDN
from library.data import InkDataset


def apply_tta(model, inputs):
    """
    Applies Test-Time Augmentation (TTA) using the Dihedral Group D4
    (8 combinations of rotations and flips).

    Args:
        model: The neural network model.
        inputs: Batch of input volumes (B, Z, H, W).

    Returns:
        torch.Tensor: Averaged logits or probabilities (B, 1, H, W).
    """
    # We will accumulate probabilities (sigmoid applied)
    accumulated_probs = None
    count = 0

    # D4 Group: 4 Rotations x 2 Flips (None, Horizontal)
    # k=0,1,2,3 corresponds to 0, 90, 180, 270 degrees

    # Iterate over flip states (False, True)
    for flip in [False, True]:
        # Iterate over rotations (0, 1, 2, 3)
        for k in range(4):
            # 1. Augment Input
            x = inputs.clone()

            if flip:
                # Flip width (dim 3)
                x = torch.flip(x, dims=[3])

            if k > 0:
                # Rotate by 90*k degrees in the H, W plane (dims 2, 3)
                x = torch.rot90(x, k=k, dims=[2, 3])

            # 2. Forward Pass
            logits = model(x)
            probs = torch.sigmoid(logits)

            # 3. De-augment Output
            # Inverse of Rotate(k) is Rotate(-k) or Rotate(4-k)
            if k > 0:
                probs = torch.rot90(probs, k=-k, dims=[2, 3])

            if flip:
                # Inverse of Flip is Flip
                probs = torch.flip(probs, dims=[3])

            # 4. Accumulate
            if accumulated_probs is None:
                accumulated_probs = probs
            else:
                accumulated_probs += probs

            count += 1

    # Average
    return accumulated_probs / count


def predict_fragment(model, fragment_id, device, threshold):
    """
    Performs sliding-window inference on a single fragment with TTA.

    Args:
        model: Loaded model.
        fragment_id (str): ID of the fragment to predict.
        device: Torch device.
        threshold (float): Binarization threshold.

    Returns:
        str: RLE encoded string of the prediction.
    """
    # 1. Setup Data Loading
    # We use the InkDataset with split='test' to get the deterministic grid
    ds = InkDataset(split="test", fragment_ids=[fragment_id], samples_per_epoch=None)
    loader = DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Mask for Shape Information
    # We need the original mask shape to initialize the reconstruction buffer
    _, mask, _ = load_volume(fragment_id, split="test", load_cached_data=True)
    H, W = mask.shape

    # 3. Initialize Reconstruction Buffers
    # prob_sum accumulates the predicted probabilities
    # count_map tracks how many times each pixel was predicted (for overlap normalization)
    prob_sum = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    half_size = Config.PATCH_SIZE // 2

    # 4. Inference Loop
    with torch.no_grad():
        for volumes, _, coords in loader:
            volumes = volumes.to(device)

            # Apply TTA (returns averaged probabilities)
            probs = apply_tta(model, volumes)
            probs = probs.cpu().numpy()
            coords = coords.numpy()

            batch_len = probs.shape[0]

            for i in range(batch_len):
                # coords[i] contains [frag_idx, center_y, center_x]
                # These centers are in the PADDED coordinate space used by InkDataset
                # We need to map them back to the original image space.
                # InkDataset.__getitem__ returns: orig_y = y - half_size
                # So the coords tensor actually contains the top-left corner in original space relative to center?
                # Let's check InkDataset.__getitem__:
                #   orig_y = y - self.half_size
                #   coord = torch.tensor([frag_idx, orig_y, orig_x], ...)
                # So coords contains the top-left coordinate of the patch in the ORIGINAL image space.

                y_start = int(coords[i, 1])
                x_start = int(coords[i, 2])

                # Calculate bounds
                y_end = y_start + Config.PATCH_SIZE
                x_end = x_start + Config.PATCH_SIZE

                # Handle Boundary Clipping
                # The patch might extend outside the original image (due to padding logic in dataset)
                # We only want to add the valid region to our reconstruction map.

                # Intersection with image bounds
                y_start_valid = max(0, y_start)
                x_start_valid = max(0, x_start)
                y_end_valid = min(H, y_end)
                x_end_valid = min(W, x_end)

                # If no overlap, skip
                if y_start_valid >= y_end_valid or x_start_valid >= x_end_valid:
                    continue

                # Offsets within the patch
                patch_y_start = y_start_valid - y_start
                patch_x_start = x_start_valid - x_start
                patch_y_end = patch_y_start + (y_end_valid - y_start_valid)
                patch_x_end = patch_x_start + (x_end_valid - x_start_valid)

                # Accumulate
                prob_sum[y_start_valid:y_end_valid, x_start_valid:x_end_valid] += probs[
                    i, 0, patch_y_start:patch_y_end, patch_x_start:patch_x_end
                ]

                count_map[y_start_valid:y_end_valid, x_start_valid:x_end_valid] += 1.0

    # 5. Normalize and Threshold
    # Avoid division by zero
    count_map[count_map == 0] = 1.0
    final_probs = prob_sum / count_map

    # Apply the valid pixel mask (set background to 0)
    final_probs = final_probs * (mask > 0)

    # Binarize
    binary_pred = (final_probs > threshold).astype(np.uint8)

    # 6. Encode
    return rle_encoding(binary_pred)


def inference():
    """
    Main inference driver.
    Loads model, iterates test fragments, generates predictions, and saves submission.csv.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Initializing Inference...")

    # 1. Load Model
    model = SGDN().to(device)
    model_path = Config.WORKING_DIR / "best_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load Threshold
    threshold_path = Config.WORKING_DIR / "threshold.txt"
    if threshold_path.exists():
        with open(threshold_path, "r") as f:
            best_threshold = float(f.read().strip())
        print(f"Loaded optimal threshold: {best_threshold}")
    else:
        best_threshold = 0.5
        print(f"Threshold file not found. Using default: {best_threshold}")

    # 3. Load Test Metadata
    if not Config.TEST_METADATA_PATH.exists():
        raise FileNotFoundError("Test metadata not found.")

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    test_ids = df_test["fragment_id"].astype(str).unique()

    print(f"Found {len(test_ids)} test fragments: {test_ids}")

    submission_rows = []

    # 4. Predict Each Fragment
    for fid in test_ids:
        print(f"Processing fragment {fid}...")
        rle = predict_fragment(model, fid, device, best_threshold)
        submission_rows.append({"Id": fid, "Predicted": rle})

    # 5. Save Submission
    sub_df = pd.DataFrame(submission_rows)

    # Ensure output directory exists if path contains directories
    if Config.SUBMISSION_PATH.parent != Path("."):
        Config.SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
