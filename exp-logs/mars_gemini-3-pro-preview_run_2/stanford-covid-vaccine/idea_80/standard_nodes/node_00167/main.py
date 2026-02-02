import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_scored_indices
from library.data import get_dataloaders
from library.model import AS_DRN
from library.train import train_epoch, validate, generate_submission, MCRMSELoss


def perform_failure_analysis(model, loader, device):
    """
    Analyzes model performance on the validation set.
    Computes per-sample error and correlates it with metadata features.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    all_preds = []
    all_targets = []
    all_ids = []

    # 1. Collect Predictions and Targets
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            # Inference (taking y_2 as the final prediction)
            y_2, _ = model(inputs, partner_indices)

            all_preds.append(y_2.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 2. Calculate Per-Sample Error (MCRMSE per sample)
    # Filter to scored length and columns
    scored_len = Config.SCORED_LENGTH
    scored_indices = get_scored_indices()

    y_pred_scored = all_preds[:, :scored_len, :][:, :, scored_indices]
    y_true_scored = all_targets[:, :scored_len, :][:, :, scored_indices]

    # MSE per sample: Mean over sequence length and channels
    mse_per_sample = np.mean((y_true_scored - y_pred_scored) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # Create a DataFrame for analysis
    error_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # 3. Load Metadata to correlate
    if os.path.exists(Config.VAL_CSV):
        meta_df = pd.read_csv(Config.VAL_CSV)

        # Merge error data with metadata
        analysis_df = pd.merge(error_df, meta_df, on="id", how="inner")

        # Calculate Correlations
        features_to_check = ["signal_to_noise", "seq_length", "mean_reactivity"]
        # Add GC content if available or calculate it, but let's stick to available columns
        # Calculate GC content from sequence if not present
        if "sequence" in analysis_df.columns:
            analysis_df["gc_content"] = analysis_df["sequence"].apply(
                lambda x: (x.count("G") + x.count("C")) / len(x) if len(x) > 0 else 0
            )
            features_to_check.append("gc_content")

        print("-" * 40)
        print(f"{'Feature':<20} | {'Correlation with Error':<20}")
        print("-" * 40)

        for feat in features_to_check:
            if feat in analysis_df.columns:
                corr = analysis_df["error"].corr(analysis_df[feat])
                print(f"{feat:<20} | {corr:.4f}")
            else:
                print(f"{feat:<20} | Not found")
        print("-" * 40)
    else:
        print("Validation metadata not found. Skipping correlation analysis.")


def main():
    # 1. Configuration Override for Fast Baseline
    # We reduce epochs to ensure execution finishes well within the time limit.
    Config.EPOCHS = 20

    # 2. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 3. Data Loading
    # debug=False ensures we use the full dataset for a valid baseline
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # 4. Model Initialization
    model = AS_DRN().to(device)

    # 5. Optimization Setup
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = MCRMSELoss()

    # 6. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_score)

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

        # Optional: Print progress (kept minimal)
        # print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Score {val_score:.4f}")

    # 7. Final Evaluation
    print("Training complete. Loading best model...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_metric = validate(model, val_loader, device)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # 8. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 9. Submission Generation
    # Threshold check as per requirements
    THRESHOLD = 0.47142532743789534

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Metric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
