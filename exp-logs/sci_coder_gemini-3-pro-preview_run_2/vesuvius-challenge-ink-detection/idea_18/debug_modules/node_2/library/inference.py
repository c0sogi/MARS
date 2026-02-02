import os
import gc
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import TestInkDataset
from library.utils import rle_encoding


def run_inference(model, device):
    """
    Executes the inference pipeline using Multi-View Ensemble Scanning and TTA.
    Generates the submission file.

    Args:
        model (torch.nn.Module): The trained SegFormer model.
        device (torch.device): The computing device (CPU or CUDA).
    """
    print("Starting Inference with Multi-View Ensemble Scanning...")
    model.eval()

    # Ensure the output directory exists
    submission_path = Config.SUBMISSION_PATH
    submission_dir = os.path.dirname(submission_path)
    if submission_dir:
        os.makedirs(submission_dir, exist_ok=True)

    # Load test metadata
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    fragment_ids = test_df["fragment_id"].unique()

    submission_data = []

    for fid in fragment_ids:
        print(f"Processing fragment: {fid}")

        # Instantiate a dummy dataset to retrieve the original fragment dimensions
        # We use 'load_cached_data=True' to utilize the caching mechanism in dataset.py
        temp_ds = TestInkDataset(fid, view="B", load_cached_data=True)
        h_orig, w_orig = temp_ds.h, temp_ds.w

        # Accumulator for the full prediction maps from each view
        # We will store them as tensors on CPU to manage memory
        view_preds = []

        # Iterate through the 3 discrete views (A, B, C)
        # View A: High (Slices 16-40)
        # View B: Center (Slices 20-44)
        # View C: Low (Slices 24-48)
        views = ["A", "B", "C"]

        for view in views:
            # Initialize dataset for the specific view
            ds = TestInkDataset(fid, view=view, load_cached_data=True)
            loader = DataLoader(
                ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Allocate memory for the full view prediction map
            full_pred_view = torch.zeros((h_orig, w_orig), dtype=torch.float32)

            with torch.no_grad():
                for images, coords, sizes, _ in loader:
                    images = images.to(device)

                    # --- Test Time Augmentation (TTA) ---

                    # 1. Original
                    out = model(images)
                    preds = torch.sigmoid(out)

                    # 2. Horizontal Flip (dim 3)
                    images_flip = torch.flip(images, [3])
                    out_flip = model(images_flip)
                    preds_flip = torch.flip(torch.sigmoid(out_flip), [3])

                    # 3. Vertical Flip (dim 2)
                    images_vflip = torch.flip(images, [2])
                    out_vflip = model(images_vflip)
                    preds_vflip = torch.flip(torch.sigmoid(out_vflip), [2])

                    # Average predictions
                    batch_preds = (preds + preds_flip + preds_vflip) / 3.0

                    # Move to CPU for placement
                    batch_preds = batch_preds.cpu()

                    # Stitch patches into the full view map
                    for i in range(images.size(0)):
                        # coords is (B, 2) -> [x, y]
                        bx = coords[i, 0].item()
                        by = coords[i, 1].item()

                        # Prediction is (1, H, W), take first channel
                        patch = batch_preds[i, 0, :, :]

                        # Determine valid placement dimensions
                        # The dataset pads the image if it's smaller than tile_size.
                        # We must crop the prediction to the valid area relative to the original image.
                        valid_h = min(Config.TILE_SIZE, h_orig - by)
                        valid_w = min(Config.TILE_SIZE, w_orig - bx)

                        full_pred_view[by : by + valid_h, bx : bx + valid_w] = patch[
                            :valid_h, :valid_w
                        ]

            view_preds.append(full_pred_view)

            # Explicit cleanup for the view loop
            del ds, loader, full_pred_view
            gc.collect()

        # --- Max-Fusion ---
        # Stack views: (3, H, W)
        stack = torch.stack(view_preds, dim=0)

        # Take the maximum probability across views per pixel
        final_prob, _ = torch.max(stack, dim=0)

        # Thresholding
        binary_mask = (final_prob > Config.THRESHOLD).numpy().astype(np.uint8)

        # Run-Length Encoding
        rle = rle_encoding(binary_mask)
        submission_data.append({"Id": fid, "Predicted": rle})

        # Explicit garbage collection to free large tensors
        del view_preds, stack, final_prob, binary_mask
        gc.collect()

    # Save submission file
    sub_df = pd.DataFrame(submission_data)
    sub_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
