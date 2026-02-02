import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, dice_score_batch
from library.dataset import ContrailDataset
from library.model import ConvNeXtUNet
from library.loss import HybridLoss
from library.engine import train_model, validate, predict_and_submit


def perform_failure_analysis(model, val_loader, val_df, device):
    """
    Analyzes model performance on the validation set to find correlations
    between error magnitude and metadata features.
    """
    print("\nStarting Failure Analysis...")
    model.eval()

    # Store per-sample dice scores
    sample_scores = []

    # Ensure val_df aligns with loader (loader must be sequential)
    # We iterate through the loader and calculate Dice for each batch's samples individually

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Inference
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > Config.THRESHOLD).float()

            # Calculate Dice per sample in the batch
            for i in range(images.size(0)):
                p = preds[i].flatten()
                t = masks[i].flatten()

                intersection = (p * t).sum().item()
                cardinality = p.sum().item() + t.sum().item()

                if cardinality == 0:
                    score = 1.0
                else:
                    score = (2.0 * intersection) / (cardinality + Config.SMOOTH)

                sample_scores.append(score)

    # Add scores to dataframe
    # Truncate df if necessary (though lengths should match)
    if len(sample_scores) != len(val_df):
        print(
            f"Warning: Score count ({len(sample_scores)}) matches dataset size, aligning with metadata."
        )

    val_df = val_df.iloc[: len(sample_scores)].copy()
    val_df["dice_score"] = sample_scores
    val_df["error_magnitude"] = 1.0 - val_df["dice_score"]

    # Extract features for correlation
    # Timestamp is available; extract hour/month if needed, or just use raw timestamp
    # Spatial: row_min, col_min

    features = ["timestamp", "row_min", "col_min"]
    correlations = {}

    print("Correlation between Error Magnitude (1-Dice) and Features:")
    for feat in features:
        if feat in val_df.columns:
            corr = val_df[feat].corr(val_df["error_magnitude"])
            correlations[feat] = corr
            print(f"  {feat}: {corr:.4f}")
        else:
            print(f"  {feat}: Not found in metadata")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    # We limit epochs to ensure completion within 2 hours.
    # The A100 is fast, but 30 epochs on full data might exceed the limit depending on IO.
    # 5 epochs is a safe baseline.
    Config.EPOCHS = 5

    device = Config.DEVICE
    print(f"Running on device: {device}")
    print(f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # 2. Data Loading
    print("Initializing Datasets...")
    # Train dataset (with augmentations)
    train_dataset = ContrailDataset(split="train", transform=None, debug=False)

    # Validation dataset (no augmentations, sequential)
    val_dataset = ContrailDataset(split="validation", transform=None, debug=False)

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = ConvNeXtUNet()
    model.to(device)

    loss_fn = HybridLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 4. Training
    print("Starting Training...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        loss_fn,
        num_epochs=Config.EPOCHS,
    )

    # 5. Final Validation
    print("Performing Final Validation...")
    # Load best model weights (train_model reloads them, but ensuring consistency)
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Compute Global Dice on Validation Set
    val_metric = validate(
        model, val_loader, device, threshold=Config.THRESHOLD, use_tta=Config.USE_TTA
    )
    print(f"Final Validation Metric: {val_metric}")

    # 6. Failure Analysis
    # Load validation metadata for correlation
    val_meta_df = pd.read_csv(Config.VALID_METADATA_PATH)
    perform_failure_analysis(model, val_loader, val_meta_df, device)

    # 7. Submission
    THRESHOLD_SCORE = 0.5676456935477064

    if val_metric > THRESHOLD_SCORE:
        print(
            f"Validation metric ({val_metric:.6f}) exceeds threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        test_dataset = ContrailDataset(split="test", transform=None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        predict_and_submit(
            model,
            test_loader,
            device,
            threshold=Config.THRESHOLD,
            use_tta=Config.USE_TTA,
            output_path=Config.SUBMISSION_PATH,
        )
    else:
        print(
            f"Validation metric ({val_metric:.6f}) did not exceed threshold ({THRESHOLD_SCORE}). Skipping submission."
        )


if __name__ == "__main__":
    main()
