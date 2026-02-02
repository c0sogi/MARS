import os
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from library.config import Config
from library.dataset import UWDataset
from library.model import MobileNetV2UNet
from library.utils import rle_encode


def generate_submission(debug=False, load_cached_data=True):
    """
    Generates predictions for the test set using the best saved model
    and saves the submission CSV.

    Args:
        debug (bool): If True, runs on a subset of the test data for debugging.
        load_cached_data (bool): If True, attempts to load processed metadata from cache.
    """
    device = Config.DEVICE
    print(f"Initializing inference on {device}...")

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 1. Load Test Data
    # UWDataset handles metadata caching internally based on load_cached_data flag
    test_dataset = UWDataset(
        mode="test", debug=debug, load_cached_data=load_cached_data
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 2. Load Model
    # We initialize without pretrained weights as we will load the state_dict
    model = MobileNetV2UNet(pretrained=False)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {Config.MODEL_SAVE_PATH}. "
            "Please ensure the model is trained before running inference."
        )

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    results = []
    print(f"Starting prediction on {len(test_dataset)} samples...")

    # 3. Inference Loop
    with torch.no_grad():
        for images, ids, heights, widths in test_loader:
            images = images.to(device, dtype=torch.float32)

            # Use autocast for mixed precision inference
            with autocast(enabled=True):
                # Output shape: (Batch, NumClasses, Height, Width)
                # Model output is already Sigmoid activated
                outputs = model(images)

            # Apply Sigmoid to logits before thresholding
            probs = torch.sigmoid(outputs)
            preds = (probs > Config.PRED_THRESHOLD).cpu().numpy().astype(np.uint8)

            # Iterate through the batch
            for i, img_id in enumerate(ids):
                # preds[i] shape: (C, H, W)
                # Get original dimensions
                h_orig = heights[i].item()
                w_orig = widths[i].item()

                for class_idx, class_name in enumerate(Config.CLASS_LABELS):
                    mask = preds[i, class_idx, :, :]

                    # Resize mask back to original resolution
                    # cv2.resize expects (width, height)
                    mask_resized = cv2.resize(
                        mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST
                    )

                    # Optional: Post-processing to remove small noise
                    if Config.MIN_MASK_AREA > 0:
                        if np.sum(mask_resized) < Config.MIN_MASK_AREA:
                            mask_resized[:] = 0

                    # Encode to RLE
                    rle = rle_encode(mask_resized)

                    # Append result
                    results.append(
                        {"id": img_id, "class": class_name, "predicted": rle}
                    )

    # 4. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure columns are in the correct order: id, class, predicted
    submission_df = submission_df[["id", "class", "predicted"]]

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")

    return submission_df
