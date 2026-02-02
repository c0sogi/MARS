import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import from provided library
from library.config import Config
from library.utils import set_seed, dice_score
from library.data import get_dataloader
from library.model import ExtendedConvNeXtUNet
from library.engine import train_model, validate, inference

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def perform_failure_analysis(model, loader, device, metadata_path):
    """
    Calculates per-sample error (1 - Dice) on the validation set and
    correlates it with metadata features to identify failure modes.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    # Load metadata to correlate with errors
    # The loader is not shuffled for validation, so order matches CSV
    try:
        meta_df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"Metadata file not found at {metadata_path}. Skipping failure analysis.")
        return

    errors = []

    # Disable gradients for inference
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Calculate per-sample Dice score
            # Iterate through batch
            batch_size = images.size(0)
            for i in range(batch_size):
                p = preds[i].cpu().numpy()
                t = masks[i].cpu().numpy()

                # Compute Dice for this single sample
                d = dice_score(p, t)

                # Error is 1 - Dice
                errors.append(1.0 - d)

    # Ensure lengths match
    if len(errors) != len(meta_df):
        print(
            f"Warning: Mismatch between predictions ({len(errors)}) and metadata ({len(meta_df)}). Truncating to minimum."
        )
        min_len = min(len(errors), len(meta_df))
        errors = errors[:min_len]
        meta_df = meta_df.iloc[:min_len]

    # Add error to dataframe
    meta_df["error"] = errors

    # Calculate correlations with numeric metadata columns
    numeric_cols = meta_df.select_dtypes(include=[np.number]).columns
    correlations = {}

    print("Correlation between Error (1-Dice) and Input Features:")
    for col in numeric_cols:
        if col != "error":
            corr = meta_df["error"].corr(meta_df[col])
            correlations[col] = corr
            print(f"  {col}: {corr:.4f}")

    # Identify highest correlation
    if correlations:
        max_feat = max(correlations, key=lambda k: abs(correlations[k]))
        print(
            f"Strongest predictor of error: {max_feat} ({correlations[max_feat]:.4f})"
        )


def main():
    # 1. Setup System
    # Ensure reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Train on a subset (5000 samples) for a fast baseline
    # Load full validation set for accurate metric calculation
    print("Initializing DataLoaders...")
    train_loader = get_dataloader(
        split="train",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        sample_size=5000,
    )

    val_loader = get_dataloader(
        split="validation",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        sample_size=None,  # Full validation set
    )

    # 3. Model Initialization
    print(f"Initializing {Config.PROJECT_NAME} model...")
    model = ExtendedConvNeXtUNet(in_channels=Config.IN_CHANNELS, num_classes=1)
    model = model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing for 5 epochs (Fast Baseline)
    epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Training Loop
    # Using the engine's train_model function
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        patience=3,
    )

    # 6. Evaluation
    # Load the best model saved during training
    print(f"Loading best model from {Config.BEST_MODEL_PATH}...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Calculate Final Validation Metric on the full hold-out set
    print("Evaluating on full validation set...")
    val_dice, _ = validate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_dice}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device, Config.VALIDATION_METADATA_PATH)

    # 8. Submission
    # Threshold check
    THRESHOLD = 0.5910660985501295

    if val_dice > THRESHOLD:
        print(
            f"\nValidation metric ({val_dice:.6f}) exceeds threshold ({THRESHOLD:.6f}). Generating submission..."
        )

        # Load Test Data
        test_loader = get_dataloader(
            split="test", batch_size=Config.BATCH_SIZE, load_cached_data=True
        )

        # Run Inference (TTA is handled inside engine.inference)
        inference(model, test_loader, device)
    else:
        print(
            f"\nValidation metric ({val_dice:.6f}) did not exceed threshold ({THRESHOLD:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
