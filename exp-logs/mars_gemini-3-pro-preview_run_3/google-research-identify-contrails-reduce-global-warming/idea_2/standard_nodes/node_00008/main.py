import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch

# Ensure library modules are accessible
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, rle_encode
from library.data import get_loaders
from library.model import ContrailUNet
from library.training import train_model


def main():
    # --- 1. Setup & Configuration ---
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set reproducible seeds
    set_seed(Config.SEED)

    # --- 2. Training ---
    # train_model handles the full training loop, validation, and checkpoint averaging
    # It saves the best averaged model to Config.BEST_MODEL_PATH
    train_model(debug=False)

    # --- 3. Validation & Failure Analysis ---
    print("Starting final validation and failure analysis...")

    device = Config.DEVICE

    # Load the best model
    # We initialize with pretrained=False (encoder_weights=None) since we load full state dict
    model = ContrailUNet(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=None,
        in_channels=Config.N_CHANNELS,
        classes=1,
    )

    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at {Config.BEST_MODEL_PATH}")

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Get loaders (using full validation set)
    _, val_loader, test_loader = get_loaders(debug=False, batch_size=Config.BATCH_SIZE)

    # Metrics
    total_intersection = 0.0
    total_union = 0.0
    image_errors = []

    # Inference Loop
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Predict
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Flatten for Global Dice
            preds_flat = preds.view(-1)
            targets_flat = masks.view(-1)

            intersection = (preds_flat * targets_flat).sum().item()
            union = preds_flat.sum().item() + targets_flat.sum().item()

            total_intersection += intersection
            total_union += union

            # Per-image error for failure analysis
            # Error = 1 - Dice(image)
            B = images.size(0)
            for b in range(B):
                p = preds[b].view(-1)
                t = masks[b].view(-1)
                i_sum = (p * t).sum().item()
                u_sum = p.sum().item() + t.sum().item()
                # Smooth dice for stability in error calc
                dice = (2.0 * i_sum + 1.0) / (u_sum + 1.0)
                image_errors.append(1.0 - dice)

    # Compute Final Metric
    final_metric = (2.0 * total_intersection) / (total_union + 1e-7)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Correlate error with metadata
    val_df = val_loader.dataset.df.copy()

    # Ensure alignment (loader is sequential)
    if len(val_df) == len(image_errors):
        val_df["error"] = image_errors

        print("Failure Analysis (Correlation with Error):")
        features = ["timestamp", "row_min", "col_min"]
        for feat in features:
            if feat in val_df.columns:
                corr = val_df[feat].corr(val_df["error"])
                print(f"Correlation between error and {feat}: {corr}")
    else:
        print("Skipping failure analysis due to size mismatch.")

    # --- 4. Submission ---
    THRESHOLD = 0.5558921016716757

    if final_metric > THRESHOLD:
        print("Metric threshold passed. Generating submission...")

        submission_data = []

        with torch.no_grad():
            for images, record_ids in test_loader:
                images = images.to(device)

                logits = model(images)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float().cpu().numpy()  # B x 1 x H x W

                for b, rid in enumerate(record_ids):
                    mask = preds[b, 0]  # H x W

                    if np.sum(mask) == 0:
                        rle = "-"
                    else:
                        rle = rle_encode(mask)

                    submission_data.append({"record_id": rid, "encoded_pixels": rle})

        # Save
        sub_df = pd.DataFrame(submission_data)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric {final_metric} is not higher than {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
