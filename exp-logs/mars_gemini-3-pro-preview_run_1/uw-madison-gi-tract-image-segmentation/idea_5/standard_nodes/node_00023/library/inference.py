import os
import cv2
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.model import Unet
from library.dataset import prepare_test_loader
from library.utils import rle_encode, keep_largest_connected_component_3d


def predict_and_submit():
    """
    Runs inference on the test set, applies 3D post-processing, and generates
    the submission CSV file.
    """
    print("Initializing Inference Pipeline...")

    # 1. Setup Device and Model
    device = Config.DEVICE
    model = Unet().to(device)

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Predictions will be random."
        )

    model.eval()

    # 2. Load Data
    # We need the loader for images and the dataframe for metadata (original sizes, grouping)
    test_loader = prepare_test_loader(load_cached_data=True)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Dictionary to store raw predictions: id -> numpy array (3, 320, 320)
    raw_preds = {}

    print("Running 2D Inference...")
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device, dtype=torch.float32)

            # Forward pass
            outputs = model(images)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            # Move to CPU
            probs_np = probs.cpu().numpy()

            # Store in dictionary
            for i, img_id in enumerate(ids):
                raw_preds[img_id] = probs_np[i]

    # 3. 3D Reconstruction and Post-Processing
    print("Processing 3D volumes (CCA) and generating RLE...")

    submission_rows = []

    # Group metadata by Case and Day to form 3D volumes
    # We iterate over these groups to process one volume at a time
    groups = test_df.groupby(["case", "day"])

    for (case, day), group in groups:
        # Ensure slices are sorted by Z-index
        # Metadata 'slice' is usually a string '0001', convert to int for sorting
        group = group.copy()
        group["slice_idx"] = group["slice"].astype(int)
        group = group.sort_values("slice_idx")

        # Get IDs and Original Dimensions for this volume
        volume_ids = group["id"].values
        widths = group["width"].values
        heights = group["height"].values

        # Collect predictions for this volume
        # Shape: (Depth, 3, 320, 320)
        vol_slices = []
        valid_indices = []

        for idx, img_id in enumerate(volume_ids):
            if img_id in raw_preds:
                vol_slices.append(raw_preds[img_id])
                valid_indices.append(idx)
            else:
                # This should theoretically not happen if loader covers all test_df
                # But we handle it by skipping or inserting zeros if needed
                pass

        if not vol_slices:
            continue

        # Stack into 3D volume
        volume_prob = np.stack(vol_slices, axis=0)

        # Threshold to Binary
        volume_mask = (volume_prob > Config.MASK_THRESHOLD).astype(np.uint8)

        # Process each class channel independently
        # 0: large_bowel, 1: small_bowel, 2: stomach
        for class_idx, class_name in enumerate(Config.CLASSES):
            # Extract class volume: (Depth, 320, 320)
            class_vol = volume_mask[:, class_idx, :, :]

            # Apply 3D Connected Component Analysis
            # This removes small floating noise artifacts in 3D space
            class_vol_clean = keep_largest_connected_component_3d(
                class_vol, min_size=Config.MIN_COMPONENT_SIZE
            )

            # Iterate back through slices to resize and encode
            for i, vol_idx in enumerate(valid_indices):
                # Get the cleaned mask for this slice
                mask_slice = class_vol_clean[i]  # (320, 320)

                # Get original dimensions
                orig_w = widths[vol_idx]
                orig_h = heights[vol_idx]

                # Resize back to original resolution
                # cv2.resize expects (width, height)
                if (orig_h, orig_w) != mask_slice.shape:
                    mask_slice = cv2.resize(
                        mask_slice, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )

                # RLE Encode
                rle = rle_encode(mask_slice)

                # Append to submission list
                submission_rows.append(
                    {"id": volume_ids[vol_idx], "class": class_name, "predicted": rle}
                )

    # 4. Save Submission
    print("Formatting and saving submission...")
    submission_df = pd.DataFrame(submission_rows)

    # Ensure correct column order
    submission_df = submission_df[["id", "class", "predicted"]]

    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)

    print(f"Submission saved successfully to {save_path}")
    print(f"Total rows generated: {len(submission_df)}")
