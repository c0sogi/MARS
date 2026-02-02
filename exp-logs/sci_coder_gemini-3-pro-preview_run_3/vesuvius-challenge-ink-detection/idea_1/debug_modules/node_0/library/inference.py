import os
import gc
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    CACHE_DIR,
    SUBMISSION_PATH,
    Z_DIM,
    PATCH_SIZE,
    INFERENCE_STRIDE,
    THRESHOLD,
    DEVICE,
    seed_everything,
)
from library.model import InkDetectorFCN
from library.utils import rle_encode


def load_normalization_stats():
    """
    Loads normalization statistics (mean, std) from the cache directory.
    These should have been generated during the training phase.
    """
    stats_path = CACHE_DIR / "normalization_stats.npy"
    if not stats_path.exists():
        print("Warning: Normalization stats not found in cache. Using default 0.0/1.0.")
        return 0.0, 1.0

    stats = np.load(stats_path, allow_pickle=True).item()
    return stats["mean"], stats["std"]


def load_volume(relative_path):
    """
    Loads the 3D surface volume from disk.

    Args:
        relative_path (str): Path relative to INPUT_DIR.

    Returns:
        np.ndarray: Volume of shape (Z, H, W).
    """
    vol_path = INPUT_DIR / relative_path
    slices = []
    for i in range(Z_DIM):
        slice_path = vol_path / f"{i:02d}.tif"
        if not slice_path.exists():
            # Fallback or error handling if a slice is missing
            raise FileNotFoundError(f"Slice missing: {slice_path}")

        img = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
        slices.append(img)

    return np.stack(slices, axis=0)


def tiled_inference(model, volume, device):
    """
    Performs inference on a large volume using a sliding window approach.

    Args:
        model (nn.Module): Trained model.
        volume (np.ndarray): Input volume of shape (Z, H, W).
        device (torch.device): Compute device.

    Returns:
        np.ndarray: Probability map of shape (H, W).
    """
    model.eval()

    z, h, w = volume.shape

    # Probability map accumulator
    probs = torch.zeros((h, w), device=device, dtype=torch.float32)
    # Count map to handle overlaps
    counts = torch.zeros((h, w), device=device, dtype=torch.float32)

    # Calculate grid
    # We ensure we cover the edges by forcing the last patch to align with the bottom/right edge
    y_steps = list(range(0, h - PATCH_SIZE, INFERENCE_STRIDE))
    if not y_steps or y_steps[-1] + PATCH_SIZE < h:
        y_steps.append(h - PATCH_SIZE)

    x_steps = list(range(0, w - PATCH_SIZE, INFERENCE_STRIDE))
    if not x_steps or x_steps[-1] + PATCH_SIZE < w:
        x_steps.append(w - PATCH_SIZE)

    # Handle case where image is smaller than patch size (unlikely but possible)
    if h < PATCH_SIZE or w < PATCH_SIZE:
        # Simple padding could be added here, but assuming standard dataset size > 256
        pass

    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                # Extract patch
                # Volume shape: (Z, H, W) -> Patch: (Z, PATCH_SIZE, PATCH_SIZE)
                patch = volume[:, y : y + PATCH_SIZE, x : x + PATCH_SIZE]

                # Add batch dimension: (1, Z, H, W)
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)

                # Forward pass
                # Output: (1, 1, PATCH_SIZE, PATCH_SIZE)
                output = model(patch_tensor)

                # Squeeze to (PATCH_SIZE, PATCH_SIZE)
                pred = output.squeeze(0).squeeze(0)

                # Accumulate
                probs[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += pred
                counts[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += 1.0

    # Average the predictions
    probs /= counts

    return probs.cpu().numpy()


def generate_submission(model_path):
    """
    Generates the submission.csv file for the test set.

    Args:
        model_path (str or Path): Path to the trained model weights.
    """
    seed_everything()

    # 1. Load Metadata
    test_meta_path = METADATA_DIR / "test.csv"
    if not test_meta_path.exists():
        print(
            f"Test metadata not found at {test_meta_path}. Cannot generate submission."
        )
        return

    df_test = pd.read_csv(test_meta_path)
    if df_test.empty:
        print("Test metadata is empty.")
        return

    # 2. Load Model
    print(f"Loading model from {model_path}...")
    model = InkDetectorFCN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 3. Load Normalization Stats
    mean, std = load_normalization_stats()
    print(f"Using normalization stats: Mean={mean:.4f}, Std={std:.4f}")

    # 4. Prepare Submission File
    # Using the path defined in config (usually ./submission.csv)
    submission_file = SUBMISSION_PATH

    # Write header
    with open(submission_file, "w") as f:
        f.write("Id,Predicted\n")

    # 5. Inference Loop
    print(f"Starting inference on {len(df_test)} fragments...")

    for _, row in df_test.iterrows():
        frag_id = str(row["fragment_id"])
        vol_path = row["surface_volume_path"]
        mask_path = row["mask_path"]  # Valid pixel mask

        print(f"Processing Fragment {frag_id}...")

        # Load Volume
        try:
            volume = load_volume(vol_path)
        except Exception as e:
            print(f"Error loading volume for fragment {frag_id}: {e}")
            continue

        # Normalize
        volume = (volume.astype(np.float32) - mean) / (std + 1e-6)

        # Run Inference
        prob_map = tiled_inference(model, volume, DEVICE)

        # Apply Valid Mask (if available)
        # The competition data includes a mask.png indicating valid areas of the papyrus.
        # Predictions outside this mask should be 0.
        if mask_path and pd.notna(mask_path):
            full_mask_path = INPUT_DIR / mask_path
            if full_mask_path.exists():
                valid_mask = cv2.imread(str(full_mask_path), cv2.IMREAD_GRAYSCALE)
                if valid_mask is not None:
                    # Resize if necessary (though dimensions should match)
                    if valid_mask.shape != prob_map.shape:
                        valid_mask = cv2.resize(
                            valid_mask,
                            (prob_map.shape[1], prob_map.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    prob_map = prob_map * (valid_mask > 0)

        # Thresholding
        binary_pred = (prob_map > THRESHOLD).astype(np.uint8)

        # Encode
        rle_str = rle_encode(binary_pred)

        # Append to submission file
        with open(submission_file, "a") as f:
            f.write(f"{frag_id},{rle_str}\n")

        # Memory Cleanup
        del volume, prob_map, binary_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Submission generated at {submission_file}")
