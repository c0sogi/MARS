import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import provided library modules
from library.config import (
    WORKING_DIR,
    TEST_METADATA_PATH,
    VAL_METADATA_PATH,
    MODEL_CONFIG,
    BATCH_SIZE,
    NUM_WORKERS,
    get_device,
)
from library.utils import set_seed, rle_encode, get_transforms, dice_coeff
from library.dataset import ContrailsDataset
from library.model import DilatedResNetUNet
from library.train import train_model

# ==========================================
# Constants & Configuration
# ==========================================
# Fast baseline settings
FAST_TRAIN_SAMPLES = 12000
FAST_EPOCHS = 8
THRESHOLD_METRIC = 0.5676456935477064
SUBMISSION_PATH = "./submission/submission.csv"
SUBMISSION_DIR = "./submission"


def perform_failure_analysis(model, device):
    """
    Analyzes model performance on the validation set and correlates errors with metadata.
    """
    print("\nStarting Failure Analysis...")

    # Load validation metadata to get auxiliary features
    val_df = pd.read_csv(VAL_METADATA_PATH)
    val_df["record_id"] = val_df["record_id"].astype(str)

    # Create dataset and loader
    val_dataset = ContrailsDataset(
        metadata_path=VAL_METADATA_PATH,
        split="validation",
        transform=get_transforms(data="valid"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()

    # Store errors and record_ids
    errors = []
    record_ids = []

    # Accumulators for Global Dice
    total_intersection = 0.0
    total_union = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            r_ids = batch["record_id"]

            # Forward
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # 1. Compute Global Dice Statistics
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            total_intersection += (preds_flat * masks_flat).sum().item()
            total_union += preds_flat.sum().item() + masks_flat.sum().item()

            # 2. Compute per-sample error for correlation analysis
            # Error metric: 1 - Dice (Sample Level)
            # We compute dice per image in the batch
            B = images.size(0)
            for i in range(B):
                p = preds[i].view(-1)
                t = masks[i].view(-1)
                inter = (p * t).sum().item()
                union = p.sum().item() + t.sum().item()
                d = (2.0 * inter + 1e-6) / (union + 1e-6)

                errors.append(1.0 - d)
                record_ids.append(str(r_ids[i]))

    # Compute Final Global Dice
    epsilon = 1e-6
    global_dice = (2.0 * total_intersection + epsilon) / (total_union + epsilon)

    print(f"Final Validation Metric: {global_dice}")

    # Correlation Analysis
    error_df = pd.DataFrame({"record_id": record_ids, "error": errors})

    # Merge with metadata
    analysis_df = error_df.merge(val_df, on="record_id", how="left")

    # Features to correlate
    features = ["timestamp", "row_min", "col_min"]

    print("\nCorrelation between Error (1-Dice) and Metadata features:")
    for feat in features:
        if feat in analysis_df.columns:
            corr = analysis_df[feat].corr(analysis_df["error"])
            print(f"  {feat}: {corr:.4f}")

    return global_dice


def inference_and_submission(model, device):
    """
    Generates predictions for the test set using TTA and saves submission.
    """
    print("\nStarting Inference on Test Set...")

    # Create submission directory
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Load Test Dataset
    test_dataset = ContrailsDataset(
        metadata_path=TEST_METADATA_PATH,
        split="test",
        transform=get_transforms(data="test"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    submission_data = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            r_ids = batch["record_id"]

            # --- Test Time Augmentation (TTA) ---
            # 1. Original
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig)

            # 2. Horizontal Flip
            images_h = torch.flip(images, dims=[3])
            logits_h = model(images_h)
            probs_h = torch.sigmoid(logits_h)
            probs_h = torch.flip(probs_h, dims=[3])  # Flip back

            # 3. Vertical Flip
            images_v = torch.flip(images, dims=[2])
            logits_v = model(images_v)
            probs_v = torch.sigmoid(logits_v)
            probs_v = torch.flip(probs_v, dims=[2])  # Flip back

            # 4. Rotate 180
            images_r = torch.rot90(images, k=2, dims=[2, 3])
            logits_r = model(images_r)
            probs_r = torch.sigmoid(logits_r)
            probs_r = torch.rot90(
                probs_r, k=2, dims=[2, 3]
            )  # Rotate back (k=2 is same inverse)

            # Average probabilities
            avg_probs = (probs_orig + probs_h + probs_v + probs_r) / 4.0

            # Threshold
            preds = (avg_probs > 0.5).float().cpu().numpy()

            # Encode
            for i, r_id in enumerate(r_ids):
                mask = preds[i, 0, :, :]  # (H, W)
                rle = rle_encode(mask)
                submission_data.append({"record_id": r_id, "encoded_pixels": rle})

    # Save Submission
    sub_df = pd.DataFrame(submission_data)
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH} with {len(sub_df)} records.")


def main():
    set_seed()
    device = get_device()

    # 1. Train Model
    # We use the imported train_model function.
    # It saves the best model to working/idea_8/best_model.pth
    print("=== Phase 1: Training ===")
    train_model(
        epochs=FAST_EPOCHS, batch_size=BATCH_SIZE, debug_size=FAST_TRAIN_SAMPLES
    )

    # 2. Load Best Model
    print("\n=== Phase 2: Loading Best Model ===")
    model = DilatedResNetUNet(config=MODEL_CONFIG)
    model_path = os.path.join(WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print("Error: Model file not found. Training might have failed.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 3. Validation & Failure Analysis
    print("\n=== Phase 3: Validation & Failure Analysis ===")
    val_metric = perform_failure_analysis(model, device)

    # 4. Submission
    print("\n=== Phase 4: Submission Check ===")
    if val_metric > THRESHOLD_METRIC:
        print(
            f"Validation Metric {val_metric:.6f} > Threshold {THRESHOLD_METRIC:.6f}. Generating submission..."
        )
        inference_and_submission(model, device)
    else:
        print(
            f"Validation Metric {val_metric:.6f} <= Threshold {THRESHOLD_METRIC:.6f}. Skipping submission."
        )


if __name__ == "__main__":
    main()
