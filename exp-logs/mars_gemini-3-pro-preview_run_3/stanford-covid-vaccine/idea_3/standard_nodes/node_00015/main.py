import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Ensure library modules are importable
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, get_device, MCRMSELoss, format_submission
from library.data import get_dataloaders
from library.model import ConvTransformer
from library.train import train_one_epoch, validate, generate_predictions


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for fast baseline execution
    Config.EPOCHS = 15
    Config.EARLY_STOPPING_PATIENCE = 5

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # Get computation device
    device = get_device()
    print(f"Running on device: {device}")

    # Create necessary directories
    Config.create_dirs()

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading data...")
    # Using cached data if available for speed
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing model...")
    model = ConvTransformer().to(device)

    # ==========================================
    # 4. Training Setup
    # ==========================================
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = MCRMSELoss()

    # ==========================================
    # 5. Training Loop
    # ==========================================
    best_val_loss = float("inf")
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # ==========================================
    # 6. Final Evaluation
    # ==========================================
    # Load best model
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print("Loaded best model checkpoint.")

    # Compute Final Validation Metric
    final_val_loss = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_val_loss}")

    # ==========================================
    # 7. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")
    model.eval()

    # Collect predictions and targets for validation set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Slice to scored length (68) for error analysis
            outputs_scored = outputs[:, : Config.PRED_LEN, :]
            targets_scored = targets[:, : Config.PRED_LEN, :]

            all_preds.append(outputs_scored.cpu().numpy())
            all_targets.append(targets_scored.numpy())

    # Concatenate
    preds_arr = np.concatenate(all_preds, axis=0)  # (N, 68, 5)
    targets_arr = np.concatenate(all_targets, axis=0)  # (N, 68, 5)

    # Calculate MCRMSE per sample
    # Mean over sequence (axis 1), then Sqrt -> RMSE per target -> Mean over targets (axis 1)
    sq_diff = (targets_arr - preds_arr) ** 2
    mse_per_col = np.mean(sq_diff, axis=1)  # (N, 5)
    rmse_per_col = np.sqrt(mse_per_col)  # (N, 5)

    # Only average over scored targets
    sample_errors = np.mean(rmse_per_col[:, Config.SCORED_INDICES], axis=1)  # (N,)

    # Load validation metadata
    val_df = pd.read_parquet(Config.VAL_METADATA)

    # Ensure lengths match
    if len(val_df) == len(sample_errors):
        val_df["model_error"] = sample_errors

        # Calculate GC content
        val_df["gc_content"] = val_df["sequence"].apply(
            lambda s: (s.count("G") + s.count("C")) / len(s)
        )

        # Calculate correlations
        # We check correlation of error with Signal-to-Noise, SN_filter, and GC content
        correlations = val_df[
            ["model_error", "signal_to_noise", "SN_filter", "gc_content"]
        ].corr()["model_error"]
        print("Correlation between Model Error and Input Features:")
        print(correlations.drop("model_error"))
    else:
        print(
            f"Warning: Validation metadata length ({len(val_df)}) does not match prediction length ({len(sample_errors)}). Skipping correlation analysis."
        )

    # ==========================================
    # 8. Submission Logic
    # ==========================================
    TARGET_METRIC = 0.7421537041664124

    if final_val_loss < TARGET_METRIC:
        print(f"\nMetric {final_val_loss} < {TARGET_METRIC}. Generating submission...")

        # Generate predictions on test set
        predictions, ids = generate_predictions(model, test_loader, device)

        # Format submission
        submission_df = format_submission(predictions, ids)

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nMetric {final_val_loss} >= {TARGET_METRIC}. Submission skipped.")


if __name__ == "__main__":
    main()
