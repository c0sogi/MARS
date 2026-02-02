import os
import torch
import numpy as np
import pandas as pd
import sys

# Import components from the provided library
from library.config import Config
from library.utils import set_seed, MCRMSEMetric
from library.loss import AnchoredMCRMSELoss
from library.data import load_data
from library.model import ADFRN
from library.train import train_one_epoch, validate, generate_submission


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for a fast baseline
    Config.EPOCHS = 10
    # We use the full provided dataset (approx 1700 train samples) as it is
    # small enough for fast execution.
    Config.SUBSET_SIZE = None

    # Set random seeds for reproducibility
    set_seed()

    # Setup device
    device = torch.device(Config.DEVICE)

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    # Load cached data if available to speed up execution
    train_loader = load_data(mode="train", load_cached_data=True)
    val_loader = load_data(mode="val", load_cached_data=True)

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    model = ADFRN().to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Loss and Metric
    criterion = AnchoredMCRMSELoss()
    metric = MCRMSEMetric()

    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, metric, device)

        # Update Scheduler
        scheduler.step(val_mcrmse)

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    # =========================================================================
    # 5. Final Validation Metric
    # =========================================================================
    print(f"Final Validation Metric: {best_mcrmse}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    print("Running Failure Analysis...")

    # Load best model for analysis
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Collect predictions and targets for the validation set
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for inputs, partner_indices, targets in val_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Forward Pass (Pass 2 is the refined prediction used for inference)
            _, preds_pass2 = model(inputs, partner_indices)

            val_preds.append(preds_pass2.cpu().numpy())
            val_targets.append(targets.numpy())

    val_preds = np.concatenate(val_preds, axis=0)  # (N, 107, 5)
    val_targets = np.concatenate(val_targets, axis=0)  # (N, 107, 5)

    # Calculate RMSE per sample on scored columns and positions
    # Scored indices based on Config: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]
    scored_len = Config.SCORED_LEN

    # Slice to valid region
    preds_scored = val_preds[:, :scored_len, scored_indices]
    targets_scored = val_targets[:, :scored_len, scored_indices]

    # Compute MSE per sample: mean over length and columns
    mse_per_sample = np.mean((preds_scored - targets_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Load Metadata for correlation
    val_csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
    val_df = pd.read_csv(val_csv_path)

    # Get IDs from loader to ensure alignment
    loader_ids = val_loader.dataset.ids

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame({"id": loader_ids, "error": rmse_per_sample})

    # Merge with metadata
    analysis_df = analysis_df.merge(val_df, on="id", how="left")

    # Derive GC content
    analysis_df["GC_content"] = analysis_df["sequence"].apply(
        lambda x: (x.count("G") + x.count("C")) / len(x)
    )

    # Compute Correlations
    features_to_check = ["signal_to_noise", "GC_content", "mean_reactivity"]

    print("Correlation between Error and Features:")
    for feat in features_to_check:
        if feat in analysis_df.columns:
            # Drop NaNs for valid correlation
            valid_data = analysis_df[[feat, "error"]].dropna()
            if len(valid_data) > 1:
                x = valid_data[feat].values
                y = valid_data["error"].values
                # Use numpy for correlation
                corr = np.corrcoef(x, y)[0, 1]
                print(f"{feat}: {corr:.4f}")

    # =========================================================================
    # 7. Submission
    # =========================================================================
    threshold = 0.47142532743789534

    if best_mcrmse < threshold:
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        generate_submission(model, device, submission_path)
    else:
        print(
            f"Validation metric {best_mcrmse} is not lower than {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
