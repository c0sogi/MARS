import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.data import get_dataloaders
from library.model import RHIGFN
from library.train import train_one_epoch, validate, generate_submission, set_seed


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model errors on the validation set and correlates with metadata features.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    # 1. Compute per-sample errors
    sample_rmses = []

    # Scored columns indices from Config
    scored_indices = Config.SCORED_COLS_INDICES
    seq_scored = Config.SEQ_SCORED

    with torch.no_grad():
        for x, p_idx, y in val_loader:
            x = x.to(device)
            p_idx = p_idx.to(device)
            y = y.to(device)

            # Inference (Two passes as per RHI-GFN)
            # Pass 1
            pred_1 = model(x, p_idx, feedback=None)
            # Pass 2
            pred_2 = model(x, p_idx, feedback=pred_1)

            # Slice to scored region and columns
            # pred_2: (B, L, 5)
            p = pred_2[:, :seq_scored, scored_indices]
            t = y[:, :seq_scored, scored_indices]

            # Compute RMSE per sample
            # MSE per sample: mean over (Length, Channels)
            mse = torch.mean((p - t) ** 2, dim=(1, 2))
            rmse = torch.sqrt(mse)

            sample_rmses.extend(rmse.cpu().numpy())

    # 2. Load Metadata
    # The val_loader is sequential (shuffle=False), so it matches val.csv order
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Handle DEBUG subsetting if applicable
    if len(sample_rmses) != len(val_df):
        val_df = val_df.iloc[: len(sample_rmses)].copy()

    val_df["model_rmse"] = sample_rmses

    # 3. Feature Engineering for Correlation
    # Existing: signal_to_noise, SN_filter
    # Derived: Base counts
    val_df["count_A"] = val_df["sequence"].apply(lambda s: s.count("A"))
    val_df["count_G"] = val_df["sequence"].apply(lambda s: s.count("G"))
    val_df["count_C"] = val_df["sequence"].apply(lambda s: s.count("C"))
    val_df["count_U"] = val_df["sequence"].apply(lambda s: s.count("U"))

    # 4. Compute Correlations
    features = [
        "signal_to_noise",
        "SN_filter",
        "count_A",
        "count_G",
        "count_C",
        "count_U",
    ]

    print(f"{'Feature':<20} | {'Correlation (Pearson)':<20}")
    print("-" * 45)

    for feat in features:
        if feat in val_df.columns:
            # Drop NaNs if any
            valid_df = val_df[[feat, "model_rmse"]].dropna()
            if len(valid_df) > 1:
                corr, _ = stats.pearsonr(valid_df[feat], valid_df["model_rmse"])
                print(f"{feat:<20} | {corr:.4f}")
            else:
                print(f"{feat:<20} | N/A (Not enough data)")
        else:
            print(f"{feat:<20} | Not Found")
    print("-" * 45)


def main():
    # =========================================================================
    # 1. Setup & Configuration
    # =========================================================================
    # Modify Config for Fast Baseline
    Config.EPOCHS = 15
    Config.DEBUG = False

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        debug=Config.DEBUG
    )

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    model = RHIGFN().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=False,
    )

    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    best_score = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler
        scheduler.step(val_score)

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # =========================================================================
    # 5. Final Evaluation
    # =========================================================================
    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Compute final metric
    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    perform_failure_analysis(model, val_loader, device)

    # =========================================================================
    # 7. Submission
    # =========================================================================
    THRESHOLD = 0.47142532743789534

    if final_metric < THRESHOLD:
        # Prepare output directory
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        generate_submission(model, test_loader, test_ids, device, submission_path)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
