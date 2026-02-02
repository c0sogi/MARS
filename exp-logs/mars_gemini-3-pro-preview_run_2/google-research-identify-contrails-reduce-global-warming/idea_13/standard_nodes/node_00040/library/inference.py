import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, rle_encode
from library.dataset import get_dataloader
from library.model import MacroContextUNet


def predict_and_submit(config: Config = None):
    """
    Performs inference on the test dataset using the trained model with Test-Time Augmentation (TTA).
    Generates a submission CSV file in the required format.

    Args:
        config (Config, optional): Configuration object. If None, a new Config is initialized.
    """
    if config is None:
        config = Config()

    # Ensure reproducibility
    seed_everything(config.seed)

    device = config.device
    print(f"Starting inference on device: {device}")

    # 1. Load Data
    # get_dataloader handles loading the test_metadata.csv and creating the dataset
    test_loader = get_dataloader(config, mode="test")
    print(f"Test data loaded. Batches: {len(test_loader)}")

    # 2. Load Model
    model = MacroContextUNet(config)
    model.to(device)

    weights_path = config.get_model_save_path("best_model.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. Please train the model first."
        )

    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded from {weights_path}")

    # 3. Inference Loop
    results = []

    print("Running inference with TTA (Original, H-Flip, V-Flip, Rot180)...")

    with torch.no_grad():
        for images, record_ids in test_loader:
            images = images.to(device)

            # --- Test-Time Augmentation (TTA) ---

            # Variant 1: Original
            # Model returns logits in eval mode
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # Variant 2: Horizontal Flip
            images_h = torch.flip(images, dims=[3])
            logits_h = model(images_h)
            probs_h = torch.sigmoid(logits_h)
            probs_h = torch.flip(probs_h, dims=[3])  # Flip back

            # Variant 3: Vertical Flip
            images_v = torch.flip(images, dims=[2])
            logits_v = model(images_v)
            probs_v = torch.sigmoid(logits_v)
            probs_v = torch.flip(probs_v, dims=[2])  # Flip back

            # Variant 4: Rotate 180 (Equivalent to H-Flip + V-Flip)
            images_r = torch.flip(images, dims=[2, 3])
            logits_r = model(images_r)
            probs_r = torch.sigmoid(logits_r)
            probs_r = torch.flip(probs_r, dims=[2, 3])  # Flip back

            # Average predictions
            avg_probs = (probs_orig + probs_h + probs_v + probs_r) / 4.0

            # --- Post-Processing ---

            # Apply threshold
            pred_masks = (avg_probs > config.threshold).float()

            # Convert to numpy for encoding
            pred_masks_np = pred_masks.cpu().numpy()

            # Iterate over batch to encode
            for i, record_id in enumerate(record_ids):
                # Extract single mask: (1, H, W) -> (H, W)
                mask = pred_masks_np[i, 0, :, :]

                # Encode
                rle_str = rle_encode(mask)

                results.append({"record_id": record_id, "encoded_pixels": rle_str})

    # 4. Save Submission
    df_submission = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(config.submission_dir, exist_ok=True)

    # Save to CSV
    df_submission.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")
    print(f"Total records processed: {len(df_submission)}")
