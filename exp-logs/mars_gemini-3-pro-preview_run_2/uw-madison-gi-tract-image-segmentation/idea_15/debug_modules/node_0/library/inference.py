import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, rle_encode, keep_largest_component_3d
from library.dataset import get_processed_dataframe, UWMapDataset
from library.model import UnetPlusPlus


def inference_fn(debug=False):
    """
    Performs inference on the test set, applies 3D post-processing,
    and generates the submission file.

    Args:
        debug (bool): If True, runs on a small subset of the test data for debugging.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting inference (Debug={debug})...")

    # 2. Data Loading
    # Load test metadata (handles 2.5D neighbor logic internally)
    df_test = get_processed_dataframe(Config.TEST_METADATA_PATH, split_name="test")

    if debug:
        print("Debug mode: processing first 100 slices only.")
        df_test = df_test.iloc[:100].reset_index(drop=True)

    # Dataset & Loader
    # Test mode returns only images
    test_dataset = UWMapDataset(df_test, mode="test", img_size=Config.IMG_SIZE)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Loading
    model = UnetPlusPlus()
    weights_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(
            f"WARNING: Weights not found at {weights_path}. Using random initialization."
        )

    model.to(device)
    model.eval()

    # 4. Inference Loop
    # We need to group slices by (case, day) to perform 3D post-processing.
    # Structure: volume_storage[vol_id] = list of dicts with slice info and mask
    volume_storage = {}

    # Track index to map loader batches back to dataframe rows
    current_idx = 0

    print("Running prediction loop...")
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            # Threshold immediately to save memory (store as uint8)
            preds_binary = (preds > Config.THRESHOLD).cpu().numpy().astype(np.uint8)

            batch_size = images.size(0)

            for b in range(batch_size):
                # Retrieve metadata for this slice
                row = df_test.iloc[current_idx + b]

                # Unique volume identifier: case + day
                vol_id = f"{row['case']}_{row['day']}"

                if vol_id not in volume_storage:
                    volume_storage[vol_id] = []

                volume_storage[vol_id].append(
                    {
                        "slice_idx": row["slice"],
                        "orig_h": row["img_height"],
                        "orig_w": row["img_width"],
                        "id": row["id"],
                        "mask": preds_binary[b],  # Shape: (3, H, W)
                    }
                )

            current_idx += batch_size

    # 5. Post-processing & Formatting
    print("Applying 3D post-processing and generating RLEs...")

    results = []
    class_names = ["large_bowel", "small_bowel", "stomach"]

    # Iterate over each volume (Case + Day)
    for vol_id, slices_data in volume_storage.items():
        # Sort slices by Z-index to ensure correct 3D structure
        slices_data.sort(key=lambda x: x["slice_idx"])

        # Stack masks to create 4D array: (Num_Slices, Classes, H, W)
        # Then transpose to (Classes, Num_Slices, H, W) for easier processing per class
        vol_stack = np.stack([x["mask"] for x in slices_data], axis=0)  # (D, 3, H, W)
        vol_stack = vol_stack.transpose(1, 0, 2, 3)  # (3, D, H, W)

        # Process each class independently
        for c_idx in range(Config.NUM_CLASSES):
            class_vol = vol_stack[c_idx]  # (D, H, W)

            # Apply 3D Largest Connected Component
            processed_vol = keep_largest_component_3d(class_vol)

            # Resize back to original resolution and encode
            for i, slice_info in enumerate(slices_data):
                mask_slice = processed_vol[i]  # (H, W) - still 512x512

                orig_h = slice_info["orig_h"]
                orig_w = slice_info["orig_w"]

                # Resize if necessary
                if (orig_h != Config.IMG_SIZE) or (orig_w != Config.IMG_SIZE):
                    # cv2.resize expects (width, height)
                    mask_final = cv2.resize(
                        mask_slice, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )
                else:
                    mask_final = mask_slice

                # Encode
                rle = rle_encode(mask_final)

                # Add to results
                results.append(
                    {
                        "id": slice_info["id"],
                        "class": class_names[c_idx],
                        "predicted": rle,
                    }
                )

    # 6. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure correct column order
    submission_df = submission_df[["id", "class", "predicted"]]

    # Sort by ID and Class to look tidy (optional but good practice)
    submission_df.sort_values(by=["id", "class"], inplace=True)

    save_path = Config.SUBMISSION_PATH
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(f"Total rows generated: {len(submission_df)}")
