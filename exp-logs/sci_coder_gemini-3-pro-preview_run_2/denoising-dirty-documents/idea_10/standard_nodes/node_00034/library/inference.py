import os
import numpy as np
import torch
import pandas as pd

from library.utils import (
    seed_everything,
    get_device,
    load_checkpoint,
    load_metadata,
    load_image_with_cache,
    save_submission,
)
from library.model import WaveCACResUNet, predict_tiled


def predict_image_tiled(model, image, device, patch_size=128, overlap=0.5):
    """
    Performs sliding-window inference with overlap to handle full-resolution images.
    Wraps the library function to match the module description.
    """
    return predict_tiled(model, image, device, patch_size, overlap)


def apply_tta(model, image, device):
    """
    Applies Test-Time Augmentation (TTA) by averaging predictions
    from the original image and its geometric transformations (D4 Group).
    """
    preds = []

    # 1. Original
    p1 = predict_image_tiled(model, image, device)
    preds.append(p1)

    # 2. Horizontal Flip
    img_h = np.flip(image, axis=1)
    p2 = predict_image_tiled(model, img_h, device)
    preds.append(np.flip(p2, axis=1))

    # 3. Vertical Flip
    img_v = np.flip(image, axis=0)
    p3 = predict_image_tiled(model, img_v, device)
    preds.append(np.flip(p3, axis=0))

    # 4. Rotate 90 (k=1)
    img_r1 = np.rot90(image, k=1)
    p4 = predict_image_tiled(model, img_r1, device)
    preds.append(np.rot90(p4, k=-1))

    # 5. Rotate 180 (k=2)
    img_r2 = np.rot90(image, k=2)
    p5 = predict_image_tiled(model, img_r2, device)
    preds.append(np.rot90(p5, k=-2))

    # 6. Rotate 270 (k=3)
    img_r3 = np.rot90(image, k=3)
    p6 = predict_image_tiled(model, img_r3, device)
    preds.append(np.rot90(p6, k=-3))

    # 7. Transpose (Swap axes) -> Equivalent to H-Flip + Rot90
    img_t = np.transpose(image)
    p7 = predict_image_tiled(model, img_t, device)
    preds.append(np.transpose(p7))

    # 8. Transpose + Flip (Anti-transpose)
    img_at = np.flip(np.transpose(image), axis=1)
    p8 = predict_image_tiled(model, img_at, device)
    preds.append(np.transpose(np.flip(p8, axis=1)))

    # Average all predictions
    return np.mean(preds, axis=0)


def generate_submission(
    checkpoint_path: str,
    data_dir: str = "./input",
    output_path: str = "./submission/submission.csv",
    cache_dir: str = "./working/idea_10/cache",
    device_name: str = None,
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        checkpoint_path (str): Path to the trained model weights.
        data_dir (str): Root directory of the input data.
        output_path (str): Path where the submission CSV will be saved.
        cache_dir (str): Directory for caching processed images.
        device_name (str, optional): Specific device to use (e.g., 'cuda:0').
    """
    # 1. Setup
    seed_everything(42)
    if device_name:
        device = torch.device(device_name)
    else:
        device = get_device()

    print(f"Inference running on: {device}")

    # 2. Load Metadata
    try:
        df_test = load_metadata("test")
    except FileNotFoundError:
        print("Test metadata not found. Cannot generate submission.")
        return

    # 3. Initialize Model
    model = WaveCACResUNet(in_channels=1, base_filters=64).to(device)

    # 4. Load Weights
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Aborting.")
        return

    print(f"Loading model weights from {checkpoint_path}...")
    try:
        load_checkpoint(checkpoint_path, model)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()

    # 5. Inference Loop
    predictions = {}
    test_cache_dir = os.path.join(cache_dir, "test")
    os.makedirs(test_cache_dir, exist_ok=True)

    print(f"Starting inference on {len(df_test)} images...")

    for idx, row in df_test.iterrows():
        img_id = str(row["id"])
        feature_rel_path = row["feature_path"]
        feature_full_path = os.path.join(data_dir, feature_rel_path)

        # Define cache path for this specific image
        img_cache_path = os.path.join(test_cache_dir, f"{img_id}_noisy.npy")

        try:
            # Load image (using cache if available, else process and save)
            img_np = load_image_with_cache(
                feature_full_path, img_cache_path, load_cached_data=True
            )

            # Predict using Test-Time Augmentation
            pred_clean = apply_tta(model, img_np, device)

            predictions[img_id] = pred_clean

        except Exception as e:
            print(f"Failed to process image {img_id}: {e}")
            continue

    # 6. Save Submission
    print(f"Saving submission to {output_path}...")
    save_submission(predictions, output_path)
    print("Submission generation complete.")
