import os
import sys
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, rle_encode, average_checkpoints
from library.train import Trainer
from library.model import ContrailUnet
from library.data import get_loaders


def predict_tta(model, images):
    """
    Predicts with Test-Time Augmentation (Horizontal and Vertical Flips).
    """
    # Original
    logits = model(images)

    if Config.USE_TTA:
        # Horizontal Flip (dim 3 is width)
        images_h = torch.flip(images, dims=[3])
        logits_h = model(images_h)
        logits_h = torch.flip(logits_h, dims=[3])
        logits += logits_h

        # Vertical Flip (dim 2 is height)
        images_v = torch.flip(images, dims=[2])
        logits_v = model(images_v)
        logits_v = torch.flip(logits_v, dims=[2])
        logits += logits_v

        # Average
        logits /= 3.0

    return logits


def main():
    # --- 1. Configuration & Setup ---
    # Cite solution_lesson_node_00009: Full training duration required for Cosine Annealing convergence
    set_seed(Config.SEED)

    print("Initializing training...")

    # --- 2. Training ---
    # Initialize trainer (uses full dataset)
    trainer = Trainer(debug=False)

    # Run training
    trainer.fit()

    # --- 3. Model Averaging ---
    print("Loading and averaging best checkpoints...")
    # trainer.top_k_checkpoints is a heap of (dice, epoch, path)
    best_checkpoints = [x[2] for x in trainer.top_k_checkpoints]

    model = ContrailUnet()
    model.to(Config.DEVICE)

    if best_checkpoints:
        model = average_checkpoints(model, best_checkpoints, device=Config.DEVICE)
    else:
        print("No checkpoints found. Using last model state.")
        model = trainer.model

    model.eval()

    # --- 4. Validation & Failure Analysis ---
    print("Starting Validation and Failure Analysis...")

    # We need the validation loader and the metadata dataframe for analysis
    val_loader = trainer.val_loader
    val_meta_path = os.path.join(Config.METADATA_DIR, "validation.csv")
    val_df = pd.read_csv(val_meta_path)

    intersection_sum = 0.0
    union_sum = 0.0
    image_dices = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)

            # Predict with TTA
            logits = predict_tta(model, images)
            preds_prob = torch.sigmoid(logits)
            preds_bin = (preds_prob > Config.THRESHOLD).float()

            # Update Global Dice Stats
            # Flatten batch for global calculation
            preds_flat = preds_bin.view(-1)
            masks_flat = masks.view(-1)

            intersection_sum += (preds_flat * masks_flat).sum().item()
            union_sum += preds_flat.sum().item() + masks_flat.sum().item()

            # Calculate Per-Image Dice for Failure Analysis
            batch_size = images.size(0)
            for i in range(batch_size):
                p = preds_bin[i].view(-1)
                t = masks[i].view(-1)

                inter = (p * t).sum().item()
                uni = p.sum().item() + t.sum().item()

                dice = (2.0 * inter + 1e-6) / (uni + 1e-6)
                image_dices.append(dice)

    # Compute Final Global Metric
    final_metric = (2.0 * intersection_sum + 1e-6) / (union_sum + 1e-6)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    if len(image_dices) == len(val_df):
        val_df["dice"] = image_dices
        val_df["error"] = 1.0 - val_df["dice"]

        # Calculate correlations
        # We are interested in timestamp, row_min (lat), col_min (lon)
        analysis_cols = ["timestamp", "row_min", "col_min", "error"]
        # Ensure columns exist
        available_cols = [c for c in analysis_cols if c in val_df.columns]

        if "error" in available_cols:
            correlations = val_df[available_cols].corr()["error"]
            print("Correlation between Error (1-Dice) and Metadata:")
            print(correlations)
        else:
            print("Error column missing.")
    else:
        print(
            f"Warning: Mismatch in predictions ({len(image_dices)}) and metadata ({len(val_df)}) counts."
        )

    # --- 5. Submission ---
    TARGET_METRIC = 0.6272749392944963

    if final_metric > TARGET_METRIC:
        print(
            f"\nMetric ({final_metric}) > Threshold ({TARGET_METRIC}). Generating submission..."
        )

        # Load Test Loader
        loaders = get_loaders(
            batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
        )
        test_loader = loaders["test"]

        submission_data = []

        with torch.no_grad():
            for images, record_ids in test_loader:
                images = images.to(Config.DEVICE)

                # Predict with TTA
                logits = predict_tta(model, images)
                preds_prob = torch.sigmoid(logits)
                preds_bin = (preds_prob > Config.THRESHOLD).cpu().numpy()

                # Encode
                for i, rid in enumerate(record_ids):
                    # Mask shape is (1, H, W), take index 0 -> (H, W)
                    mask = preds_bin[i, 0]
                    rle = rle_encode(mask)
                    submission_data.append({"record_id": rid, "encoded_pixels": rle})

        # Save
        sub_df = pd.DataFrame(submission_data)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({TARGET_METRIC}). Skipping submission."
        )


if __name__ == "__main__":
    main()
