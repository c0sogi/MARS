import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, MCRMSEMetric
from library.loss import MaskedMCRMSELoss
from library.data import get_loaders
from library.model import RNANet
from library.train import train_one_epoch, validate, predict, generate_submission

# =========================================================================
# Configuration Overrides for Fast Baseline
# =========================================================================
# Limit epochs to ensure execution finishes well within 2 hours
Config.NUM_EPOCHS = 15
Config.WORK_DIR = "./working/idea_27"
Config.MODEL_PATH = os.path.join(Config.WORK_DIR, "best_model.pth")

# Ensure unique cache files for this run
Config.TRAIN_CACHE = os.path.join(Config.WORK_DIR, "train_data_projected_dense_v1.npz")
Config.VAL_CACHE = os.path.join(Config.WORK_DIR, "val_data_projected_dense_v1.npz")
Config.TEST_CACHE = os.path.join(Config.WORK_DIR, "test_data_projected_dense_v1.npz")

# Ensure working directory exists
os.makedirs(Config.WORK_DIR, exist_ok=True)


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates per-sample error and correlates it with metadata features.
    """
    print("Performing Failure Analysis...")

    # 1. Get Predictions on Validation Set
    # Shape: (N_Samples, Seq_Len, Num_Targets)
    preds = predict(model, val_loader, device)

    # 2. Get Ground Truth Targets
    # We iterate through the loader to get targets in the correct order
    targets_list = []
    for _, _, _, t in val_loader:
        targets_list.append(t.numpy())
    targets = np.concatenate(targets_list, axis=0)

    # 3. Slice to Scored Region and Columns
    # Config.PRED_LEN = 68
    score_len = Config.PRED_LEN

    # Identify indices of scored columns
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # Slice predictions and targets
    p_sliced = preds[:, :score_len, scored_indices]
    t_sliced = targets[:, :score_len, scored_indices]

    # 4. Calculate Error Per Sample (RMSE averaged over columns)
    # Squared Error: (N, 68, 3)
    squared_error = (p_sliced - t_sliced) ** 2
    # Mean over sequence and columns -> (N,)
    mse_per_sample = np.mean(squared_error, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 5. Load Metadata
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment (loader order should match CSV order if shuffle=False)
    if len(val_df) != len(rmse_per_sample):
        print(
            f"Warning: Metadata length ({len(val_df)}) matches predictions ({len(rmse_per_sample)}) mismatch."
        )
        return

    val_df["model_error"] = rmse_per_sample

    # 6. Compute Correlations
    features_to_check = ["signal_to_noise", "mean_reactivity", "SN_filter"]

    print("\nCorrelation between Model Error and Input Features:")
    for feat in features_to_check:
        if feat in val_df.columns:
            corr = val_df[feat].corr(val_df["model_error"])
            print(f"  {feat}: {corr:.4f}")
        else:
            print(f"  {feat}: Not found in metadata")
    print("-" * 40)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Load Data
    # load_cached_data=True allows using pre-processed .npz files if they exist
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Initialize Model
    model = RNANet().to(device)

    # 4. Setup Training Components
    criterion = MaskedMCRMSELoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 5. Training Loop
    best_mcrmse = float("inf")

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        # Scheduler Update
        scheduler.step(val_mcrmse)

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_PATH)

        # Optional: Print progress (commented out to reduce clutter, but useful for debugging)
        # print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val MCRMSE={val_mcrmse:.4f}")

    # 6. Final Evaluation
    print(f"Final Validation Metric: {best_mcrmse}")

    # Load best model for analysis and inference
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # 7. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 8. Submission
    # Threshold defined in task
    SUBMISSION_THRESHOLD = 0.5417620723771521

    if best_mcrmse < SUBMISSION_THRESHOLD:
        print(
            f"Validation metric {best_mcrmse} meets threshold {SUBMISSION_THRESHOLD}. Generating submission..."
        )
        generate_submission(model, test_loader, device)
    else:
        print(
            f"Validation metric {best_mcrmse} does not meet threshold {SUBMISSION_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
