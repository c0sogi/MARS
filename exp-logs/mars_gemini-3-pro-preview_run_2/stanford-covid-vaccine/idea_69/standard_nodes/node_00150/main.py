import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import warnings

# Add the current directory to path to ensure library imports work if run from root
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, format_submission
from library.data import get_loaders
from library.model import RHS_GFN
from library.loss import MCRMSELoss
from library.engine import train_fn, eval_fn, predict_test

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_failure_analysis(model, val_loader, device):
    """
    Analyzes model performance on the validation set.
    Calculates correlation between error and signal_to_noise.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    # Load metadata to get signal_to_noise
    val_df = pd.read_csv(Config.VAL_CSV)
    id_to_sn = dict(zip(val_df["id"], val_df["signal_to_noise"]))

    sample_errors = []
    sample_sns = []

    scored_indices = Config.SCORED_INDICES

    with torch.no_grad():
        for inputs, partner_indices, targets, ids in val_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            # Forward pass (get refined prediction y_2)
            y_pred, _ = model(inputs, partner_indices)

            # Slice to scored region
            pred_scored = y_pred[:, : Config.SCORED_LEN, scored_indices]
            true_scored = targets[:, : Config.SCORED_LEN, scored_indices]

            # Calculate RMSE per sample
            # (B, L, C) -> MSE per sample (B,) -> RMSE per sample (B,)
            mse_per_sample = torch.mean((pred_scored - true_scored) ** 2, dim=(1, 2))
            rmse_per_sample = torch.sqrt(mse_per_sample).cpu().numpy()

            for i, sample_id in enumerate(ids):
                sample_errors.append(rmse_per_sample[i])
                sample_sns.append(id_to_sn.get(sample_id, 0.0))

    # Calculate Correlation
    if len(sample_errors) > 0:
        correlation = np.corrcoef(sample_errors, sample_sns)[0, 1]
        print(f"Correlation between Sample RMSE and Signal_to_Noise: {correlation:.4f}")
        print(f"Mean Sample RMSE: {np.mean(sample_errors):.4f}")

        # Identify worst performers
        worst_indices = np.argsort(sample_errors)[-5:][::-1]
        print("Top 5 Worst Samples (High Error):")
        for idx in worst_indices:
            print(
                f"  ID: {val_df.iloc[idx]['id']}, Error: {sample_errors[idx]:.4f}, SN: {sample_sns[idx]:.2f}"
            )
    else:
        print("No validation samples found for analysis.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    Config.setup()

    # Override Config for Fast Baseline
    # We reduce epochs to ensure it finishes quickly, but keep full data for performance.
    Config.EPOCHS = 25

    print(f"Running Fast Baseline with RHS-GFN on {device}")

    # 2. Data Loading
    # Using full data (debug=False) because the dataset is small (2k samples)
    # and we need good performance to pass the threshold.
    train_loader, val_loader, test_loader = get_loaders(debug=False)

    # 3. Model Initialization
    model = RHS_GFN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = MCRMSELoss()

    # 4. Training Loop
    best_score = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)
        val_score = eval_fn(model, val_loader, device)

        scheduler.step(val_score)

        # Simple logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 5. Final Validation & Metric
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    final_val_metric = eval_fn(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_metric}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 7. Submission
    THRESHOLD = 0.47142532743789534

    if final_val_metric < THRESHOLD:
        print(
            f"\nMetric ({final_val_metric:.5f}) meets threshold ({THRESHOLD:.5f}). Generating submission..."
        )

        test_ids, test_preds = predict_test(model, test_loader, device)

        # Save to ./submission/submission.csv as requested in prompt instructions
        # Note: Config defaults to ./working/..., so we override path here
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        format_submission(test_ids, test_preds, save_path=submission_path)
    else:
        print(
            f"\nMetric ({final_val_metric:.5f}) did not meet threshold ({THRESHOLD:.5f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
