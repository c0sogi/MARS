import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, rle_encoding
from library.architecture import SegFormerMiTB4
from library.dataset import InkDataset


def generate_test_patches(fragment_row):
    """
    Generates a DataFrame of patch coordinates for a single test fragment.
    This allows us to use the InkDataset logic which expects patch-level metadata.
    """
    patches = []
    fid = fragment_row["fragment_id"]
    mask_path = os.path.join(Config.INPUT_DIR, fragment_row["mask_path"])

    # Load mask to get dimensions
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    h, w = mask.shape

    # Generate grid (non-overlapping for efficiency, InkDataset handles padding)
    for y in range(0, h, Config.TILE_SIZE):
        for x in range(0, w, Config.TILE_SIZE):
            patches.append(
                {
                    "fragment_id": fid,
                    "x": x,
                    "y": y,
                    "width": Config.TILE_SIZE,
                    "height": Config.TILE_SIZE,
                    "mask_path": fragment_row["mask_path"],
                    "volume_path": fragment_row["volume_path"],
                    "orig_h": h,
                    "orig_w": w,
                }
            )

    return pd.DataFrame(patches), h, w


def predict_fragment(model, fragment_row, device):
    """
    Performs Z-Scanning inference for a single fragment.

    Protocol:
    1. Scan multiple Z-offsets.
    2. Apply TTA (Horizontal/Vertical flips) per offset.
    3. Fuse offsets using Maximum Probability Projection.
    """
    # Generate patches for the current fragment
    patch_df, H, W = generate_test_patches(fragment_row)

    # Container for the final max-projected probability map
    final_prob_map = np.zeros((H, W), dtype=np.float32)

    # Loop through Z-offsets (Scanning)
    for z_offset in Config.INFERENCE_Z_OFFSETS:
        # Container for the current Z-slice probability map
        z_prob_map = np.zeros((H, W), dtype=np.float32)

        # Initialize Dataset and Loader for this Z-offset
        # We use cached_data=True to leverage any pre-processed npy files from training
        # or cache new ones if this is a fresh run.
        dataset = InkDataset(
            patch_df, mode="test", z_offset=z_offset, load_cached_data=True
        )

        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)  # (B, 3, H, W)
                coords = batch["coords"].numpy()  # (B, 2) -> (x, y)

                # --- Test Time Augmentation (TTA) ---
                # 1. Original
                preds_orig = torch.sigmoid(model(images))

                # 2. Horizontal Flip
                images_h = torch.flip(images, [3])
                preds_h = torch.sigmoid(model(images_h))
                preds_h = torch.flip(preds_h, [3])

                # 3. Vertical Flip
                images_v = torch.flip(images, [2])
                preds_v = torch.sigmoid(model(images_v))
                preds_v = torch.flip(preds_v, [2])

                # Average predictions
                preds_avg = (preds_orig + preds_h + preds_v) / 3.0

                # Move to CPU
                preds_np = preds_avg.cpu().numpy()  # (B, 1, H, W)

                # Place patches into z_prob_map
                for i, (x, y) in enumerate(coords):
                    pred_patch = preds_np[i, 0]  # (512, 512)

                    # Calculate valid region (unpad if patch extended beyond image)
                    y_end = min(y + Config.TILE_SIZE, H)
                    x_end = min(x + Config.TILE_SIZE, W)

                    h_valid = y_end - y
                    w_valid = x_end - x

                    z_prob_map[y:y_end, x:x_end] = pred_patch[:h_valid, :w_valid]

        # Update final map with max projection across Z
        final_prob_map = np.maximum(final_prob_map, z_prob_map)

    return final_prob_map


def run_inference():
    """
    Main inference routine.
    Loads the best model, predicts on test fragments, and generates submission.csv.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Check for model existence
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Cannot generate submission.")
        return

    # Load Model
    print(f"Loading model from {model_path}...")
    model = SegFormerMiTB4()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        print("Test metadata not found.")
        return
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    submission_data = []

    print(f"Starting inference on {len(test_df)} fragments...")

    for _, row in test_df.iterrows():
        frag_id = row["fragment_id"]
        print(f"Processing Fragment {frag_id}...")

        # Predict with Z-Scanning and TTA
        prob_map = predict_fragment(model, row, device)

        # Apply Mask (Valid Pixels)
        mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
        valid_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        valid_mask = (valid_mask > 0).astype(np.float32)

        # Mask out invalid areas
        prob_map = prob_map * valid_mask

        # Threshold
        binary_pred = (prob_map > 0.5).astype(np.uint8)

        # RLE Encode
        rle = rle_encoding(binary_pred)
        submission_data.append([frag_id, rle])

    # Save Submission
    sub_df = pd.DataFrame(submission_data, columns=["Id", "Predicted"])
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
