import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library import config, utils, model, dataset, train, predict


def main():
    # --- 1. Setup ---
    config.setup_directories()
    utils.set_seed(config.SEED)
    device = config.DEVICE
    print(f"Running on device: {device}")

    # Configuration for Fast Baseline
    EPOCHS = 10  # Reduced slightly for speed, though dataset is small
    BATCH_SIZE = config.BATCH_SIZE

    # --- 2. Data Loading ---
    print("Loading data...")
    train_loader, val_loader, _ = dataset.get_dataloaders(batch_size=BATCH_SIZE)

    # Load validation metadata for failure analysis later
    val_df = pd.read_csv(config.VAL_METADATA)

    # --- 3. Model Initialization ---
    print("Initializing model...")
    net = model.SFRPNet().to(device)

    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
    pos_weight = torch.tensor([config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- 4. Training Loop ---
    best_score = -1.0
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train.train_one_epoch(
            net, train_loader, criterion, optimizer, device
        )

        # Validate (Standard threshold 0.5 for monitoring)
        val_loss, val_preds, val_targets = train.validate(
            net, val_loader, criterion, device
        )
        val_score = utils.fbeta_score(val_preds, val_targets, beta=0.5, threshold=0.5)

        print(
            f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F0.5: {val_score:.4f}"
        )

        # Save Best
        if val_score > best_score:
            best_score = val_score
            utils.save_checkpoint(net, optimizer, epoch, val_score, best_model_path)

    # --- 5. Final Evaluation & Threshold Optimization ---
    print("Loading best model for optimization...")
    checkpoint = utils.load_checkpoint(net, best_model_path)

    # Get predictions from best model
    val_loss, val_preds, val_targets = train.validate(
        net, val_loader, criterion, device
    )

    # Optimize threshold
    best_threshold, best_opt_score = utils.optimize_threshold(
        val_preds, val_targets, beta=0.5
    )

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {best_opt_score}")

    # --- 6. Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Calculate Mean Absolute Error per patch
    # val_preds and val_targets are (N, 1, H, W)
    # We want a vector of size N
    mae_per_sample = torch.abs(val_preds - val_targets).mean(dim=(1, 2, 3)).numpy()

    # Ensure alignment with metadata
    # The val_loader is not shuffled, so order matches val_df
    if len(val_df) == len(mae_per_sample):
        val_df["error_mae"] = mae_per_sample

        # Calculate correlations
        # We focus on x, y, and potentially fragment_id (if encoded, but x/y is numeric)
        features = ["x", "y"]
        correlations = val_df[features].corrwith(val_df["error_mae"])

        print("Correlation between Error (MAE) and Input Features:")
        print(correlations)

        # Identify worst performers
        print("\nTop 5 Worst Performing Patches (Highest Error):")
        print(val_df.nlargest(5, "error_mae")[["sample_id", "x", "y", "error_mae"]])
    else:
        print(
            "Warning: Mismatch between validation set size and metadata size. Skipping detailed correlation."
        )

    # --- 7. Submission ---
    TARGET_SCORE = 0.41758

    if best_opt_score > TARGET_SCORE:
        print(
            f"\nValidation score ({best_opt_score}) > {TARGET_SCORE}. Generating submission..."
        )

        # Save the optimized threshold for the inference script
        threshold_path = os.path.join(config.WORKING_DIR, "best_threshold.txt")
        with open(threshold_path, "w") as f:
            f.write(str(best_threshold))

        # Run Inference
        predict.run_inference(
            checkpoint_path=best_model_path,
            threshold_path=threshold_path,
            output_file=config.SUBMISSION_FILE,
        )
        print(f"Submission generated at {config.SUBMISSION_FILE}")
    else:
        print(
            f"\nValidation score ({best_opt_score}) <= {TARGET_SCORE}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
