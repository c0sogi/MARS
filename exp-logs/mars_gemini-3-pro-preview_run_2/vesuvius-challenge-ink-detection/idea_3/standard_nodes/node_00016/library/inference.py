import os
import gc
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TEST_METADATA_PATH,
    SUBMISSION_PATH,
    DEVICE,
    IN_CHANNELS,
    PIXEL_MEAN,
    PIXEL_STD,
    TILE_SIZE,
)
from library.models import build_model
from library.data import get_fragment_projections
from library.utils import rle_encode
from library.train import tiled_inference


def predict_fragment(model, fragment_id, volume_path, device):
    """
    Generates a probability map for a specific fragment using 3-channel statistical projections
    and Test Time Augmentation (TTA).

    Args:
        model (nn.Module): The trained segmentation model.
        fragment_id (str): The ID of the fragment.
        volume_path (str): Relative path to the volume directory.
        device (str): Computation device ('cpu' or 'cuda').

    Returns:
        np.ndarray: Averaged probability map of shape (H, W).
    """
    # 1. Load 3-Channel Statistical Projection (Max, Mean, Std)
    # This function handles caching and normalization internally based on library.data
    image = get_fragment_projections(fragment_id, volume_path, load_cached_data=True)

    h, w = image.shape[:2]

    # 2. Define TTA Transformations
    # We use Original, Horizontal Flip, and Vertical Flip
    transforms = [
        {"name": "original", "func": lambda x: x, "inv_func": lambda x: x},
        {
            "name": "hflip",
            "func": lambda x: cv2.flip(x, 1),
            "inv_func": lambda x: cv2.flip(x, 1),
        },
        {
            "name": "vflip",
            "func": lambda x: cv2.flip(x, 0),
            "inv_func": lambda x: cv2.flip(x, 0),
        },
    ]

    accumulated_prob = np.zeros((h, w), dtype=np.float32)

    # 3. TTA Inference Loop
    for t in transforms:
        # Apply geometric transform
        aug_image = t["func"](image)

        # Convert to Tensor: (H, W, 3) -> (3, H, W) -> (1, 3, H, W)
        # We manually transpose because we aren't using the Albumentations pipeline here for simplicity in TTA control
        aug_tensor = torch.from_numpy(aug_image).permute(2, 0, 1).unsqueeze(0).float()

        # Defensive Inference: Check input shape
        if aug_tensor.shape[1] != IN_CHANNELS:
            raise ValueError(
                f"Input tensor has {aug_tensor.shape[1]} channels, expected {IN_CHANNELS}."
            )

        # Run Inference
        # tiled_inference handles moving tensor to device and tiling logic
        prob_map = tiled_inference(model, aug_tensor, device, tile_size=TILE_SIZE)

        # Apply inverse transform to align with original geometry
        prob_map = t["inv_func"](prob_map)

        accumulated_prob += prob_map

        # Cleanup
        del aug_tensor, prob_map
        gc.collect()

    # Average predictions
    avg_prob = accumulated_prob / len(transforms)

    return avg_prob


def run_inference(
    metadata_path=TEST_METADATA_PATH,
    model_path=None,
    threshold=0.5,
    submission_output=SUBMISSION_PATH,
):
    """
    Main inference pipeline. Generates predictions for all fragments in the metadata
    and saves the submission file.

    Args:
        metadata_path (str): Path to the test metadata CSV.
        model_path (str): Path to the trained model weights (.pth).
                          If None, defaults to 'best_model.pth' in WORKING_DIR.
        threshold (float): Binarization threshold.
        submission_output (str): Path to save the submission CSV.
    """
    print(f"Starting Inference with Threshold: {threshold}")

    # 1. Setup
    if model_path is None:
        model_path = os.path.join(WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    # Load Metadata
    df_test = pd.read_csv(metadata_path)

    # Build Model
    model = build_model()
    model.to(DEVICE)

    # Load Weights
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    submission_data = []

    # 2. Iterate Fragments
    for _, row in df_test.iterrows():
        fid = str(row["fragment_id"])
        vol_path = row["volume_path"]
        mask_path_rel = row["mask_path"]

        print(f"Processing Fragment {fid}...")

        # Predict
        prob_map = predict_fragment(model, fid, vol_path, DEVICE)

        # Load Valid Area Mask
        mask_full_path = os.path.join(INPUT_DIR, mask_path_rel)
        if os.path.exists(mask_full_path):
            valid_mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
            # Ensure mask is binary 0/1
            valid_mask = (valid_mask > 0).astype(np.uint8)
        else:
            print(
                f"Warning: Mask not found for fragment {fid}. Assuming full image valid."
            )
            valid_mask = np.ones_like(prob_map, dtype=np.uint8)

        # 3. Post-Processing
        # Binarize
        binary_pred = (prob_map > threshold).astype(np.uint8)

        # Apply Valid Mask
        # Resize mask if necessary (though shapes should match per metadata generation)
        if valid_mask.shape != binary_pred.shape:
            valid_mask = cv2.resize(
                valid_mask,
                (binary_pred.shape[1], binary_pred.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        binary_pred = binary_pred * valid_mask

        # Encode
        rle = rle_encode(binary_pred)
        submission_data.append({"Id": fid, "Predicted": rle})

        # Cleanup
        del prob_map, binary_pred, valid_mask
        gc.collect()

    # 4. Save Submission
    # Ensure output directory exists if path contains directories
    output_dir = os.path.dirname(submission_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv(submission_output, index=False)
    print(f"Submission saved to {submission_output}")
