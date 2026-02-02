import os
import torch
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast

from library.config import Config
from library.utils import seed_everything, rle_encode
from library.dataset import get_test_loader
from library.model import ConvNeXtUNet


def make_predictions(
    model_path=Config.BEST_MODEL_PATH, output_path=Config.SUBMISSION_FILE_PATH
):
    """
    Generates predictions for the test set using the trained model.
    Applies Test-Time Augmentation (TTA) and saves the results to a CSV file.

    Args:
        model_path (str): Path to the trained model weights.
        output_path (str): Path to save the submission CSV.
    """
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    device = Config.DEVICE
    print(f"Inference using device: {device}")

    # Initialize Model
    model = ConvNeXtUNet()

    if not os.path.exists(model_path):
        print(
            f"Warning: Model weights not found at {model_path}. Predictions will be random."
        )
    else:
        print(f"Loading model weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Get Test Loader
    test_loader = get_test_loader()
    print(f"Test batches: {len(test_loader)}")

    submission_data = []

    print("Starting inference with TTA...")

    with torch.no_grad():
        for images, record_ids in test_loader:
            images = images.to(device, non_blocking=True)

            with autocast(enabled=True):
                # 1. Original
                logits = model(images)
                probs = torch.sigmoid(logits)

                if Config.USE_TTA:
                    # 2. Horizontal Flip
                    images_h = torch.flip(images, dims=[3])
                    logits_h = model(images_h)
                    probs_h = torch.sigmoid(logits_h)
                    probs_h = torch.flip(probs_h, dims=[3])
                    probs += probs_h

                    # 3. Vertical Flip
                    images_v = torch.flip(images, dims=[2])
                    logits_v = model(images_v)
                    probs_v = torch.sigmoid(logits_v)
                    probs_v = torch.flip(probs_v, dims=[2])
                    probs += probs_v

                    # 4. Rotate 180 (equivalent to H-Flip + V-Flip)
                    images_r = torch.rot90(images, k=2, dims=[2, 3])
                    logits_r = model(images_r)
                    probs_r = torch.sigmoid(logits_r)
                    probs_r = torch.rot90(probs_r, k=-2, dims=[2, 3])
                    probs += probs_r

                    # Average
                    probs /= 4.0

            # Apply Threshold
            preds = (probs > Config.THRESHOLD).float().cpu().numpy()

            # Encode
            for i in range(len(record_ids)):
                # Extract single mask: (1, H, W) -> (H, W)
                mask = preds[i].squeeze(0)
                rle = rle_encode(mask)
                submission_data.append(
                    {"record_id": record_ids[i], "encoded_pixels": rle}
                )

    # Save Submission
    df = pd.DataFrame(submission_data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df)} records.")
