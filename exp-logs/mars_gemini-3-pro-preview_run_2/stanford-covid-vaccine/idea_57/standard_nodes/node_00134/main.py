import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, MCRMSELoss, format_submission
from library.data import get_loaders
from library.model import GCDARN
from library.train import train_epoch, validate, predict


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for Fast Baseline
    Config.EPOCHS = 15

    print(f"Running Fast Baseline on {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing model...")
    model = GCDARN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = MCRMSELoss().to(device)

    # 4. Training Loop
    best_score = float("inf")
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_score)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # 5. Final Evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    final_val_score = validate(model, val_loader, device)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_score}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    model.eval()
    val_preds = []
    val_targets = []
    val_ids = []

    with torch.no_grad():
        for features, partner_indices, targets, ids in val_loader:
            features = features.to(device)
            partner_indices = partner_indices.to(device)

            # Use the refined output (out2)
            _, out2 = model(features, partner_indices)

            val_preds.append(out2.cpu().numpy())
            val_targets.append(targets.numpy())
            val_ids.extend(ids)

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate RMSE per sample for scored columns
    # Scored indices in Config.TARGET_COLS: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]
    seq_scored = Config.SCORED_LENGTH

    # Slice data: (N, 68, 3)
    p_sliced = val_preds[:, :seq_scored, :][:, :, scored_indices]
    t_sliced = val_targets[:, :seq_scored, :][:, :, scored_indices]

    # MSE per sample (average over sequence and columns)
    mse_per_sample = np.mean((p_sliced - t_sliced) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load metadata for correlation analysis
    val_meta_df = pd.read_csv(Config.VAL_METADATA)

    # Create error dataframe
    error_df = pd.DataFrame({"id": val_ids, "error": rmse_per_sample})

    # Merge with metadata
    merged_df = pd.merge(error_df, val_meta_df, on="id")

    if "signal_to_noise" in merged_df.columns:
        corr = merged_df["error"].corr(merged_df["signal_to_noise"])
        print(f"Correlation between Error and Signal_to_Noise: {corr:.6f}")
    else:
        print("signal_to_noise column not found in metadata for correlation analysis.")

    # 7. Conditional Submission
    THRESHOLD = 0.47142532743789534

    if final_val_score < THRESHOLD:
        print(
            f"\nValidation metric {final_val_score} meets threshold {THRESHOLD}. Generating submission..."
        )
        test_preds, test_ids = predict(model, test_loader, device)
        format_submission(test_ids, test_preds, save_path=Config.SUBMISSION_PATH)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {final_val_score} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
