import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, rle_encode
from library.dataset import get_dataloader
from library.model import HybridResNetTransformerUNet


def predict_and_submit(max_samples=None):
    """
    Runs inference on the test dataset using the trained model and generates
    a submission CSV file in Run-Length Encoding (RLE) format.

    Implements Test-Time Augmentation (TTA) as defined in the configuration.

    Args:
        max_samples (int, optional): Limit the number of test samples for debugging.
                                     Defaults to None (process all).
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"Initializing inference on device: {Config.DEVICE}")

    # ---------------------------------------------------------
    # 1. Load Model
    # ---------------------------------------------------------
    model = HybridResNetTransformerUNet()
    model.to(Config.DEVICE)

    # Load weights
    if os.path.exists(Config.MODEL_PATH):
        print(f"Loading model weights from {Config.MODEL_PATH}...")
        state_dict = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(
            f"WARNING: Model weights not found at {Config.MODEL_PATH}. Using random initialization."
        )

    model.eval()

    # ---------------------------------------------------------
    # 2. Load Test Data
    # ---------------------------------------------------------
    test_loader = get_dataloader(
        split="test", batch_size=Config.BATCH_SIZE, max_samples=max_samples
    )
    print(f"Test loader initialized with {len(test_loader.dataset)} samples.")

    # ---------------------------------------------------------
    # 3. Inference Loop
    # ---------------------------------------------------------
    results = []

    print("Starting inference...")
    with torch.no_grad():
        for images, record_ids in test_loader:
            images = images.to(Config.DEVICE, dtype=torch.float)

            # -----------------------------------------------------
            # Test-Time Augmentation (TTA)
            # -----------------------------------------------------
            if Config.TTA_ENABLED:
                # 1. Original
                logits_orig = model(images)
                probs_orig = torch.sigmoid(logits_orig)

                # 2. Horizontal Flip (dim 3 is width)
                images_h = torch.flip(images, dims=[3])
                logits_h = model(images_h)
                probs_h = torch.flip(torch.sigmoid(logits_h), dims=[3])

                # 3. Vertical Flip (dim 2 is height)
                images_v = torch.flip(images, dims=[2])
                logits_v = model(images_v)
                probs_v = torch.flip(torch.sigmoid(logits_v), dims=[2])

                # 4. Rotate 180 (Equivalent to H-Flip + V-Flip)
                images_rot = torch.flip(images, dims=[2, 3])
                logits_rot = model(images_rot)
                probs_rot = torch.flip(torch.sigmoid(logits_rot), dims=[2, 3])

                # Average predictions
                avg_probs = (probs_orig + probs_h + probs_v + probs_rot) / 4.0
            else:
                # No TTA
                logits = model(images)
                avg_probs = torch.sigmoid(logits)

            # -----------------------------------------------------
            # Post-Processing
            # -----------------------------------------------------
            # Apply threshold to get binary mask
            preds = (avg_probs > Config.THRESHOLD).float().cpu().numpy()

            # Encode each mask in the batch
            for i, pred_mask in enumerate(preds):
                # pred_mask shape is (1, H, W), we need (H, W) for encoding
                mask_2d = pred_mask[0]

                # Run-Length Encoding
                rle = rle_encode(mask_2d)

                results.append({"record_id": record_ids[i], "encoded_pixels": rle})

    # ---------------------------------------------------------
    # 4. Save Submission
    # ---------------------------------------------------------
    submission_df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)

    print(f"Inference complete.")
    print(f"Submission saved to: {Config.SUBMISSION_FILE}")
    print(f"Total records processed: {len(submission_df)}")
