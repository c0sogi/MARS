import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Ensure the library modules can be imported
sys.path.append(os.path.abspath("."))

from library.config import Config
from library.network import GroupedEfficientNetV2
from library.data_loader import get_dataloaders
from library.trainer import train_epoch, validate_epoch, generate_submission, set_seed


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and metadata features (Slice Count).
    """
    print("\n--- Failure Analysis ---")
    model.eval()
    all_targets = []
    all_preds = []
    all_ids = []

    # 1. Collect Predictions
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # We need to retrieve IDs. The val_loader in library.data_loader returns (vol, label).
            # We can't easily get IDs from the loader iteration directly without modifying the dataset __getitem__
            # or relying on order.
            # However, the dataset is not shuffled for validation (shuffle=False).
            # We can retrieve IDs from the dataset object directly.

            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_targets.extend(targets.numpy().flatten())

    # Retrieve IDs from dataset (assuming order is preserved as shuffle=False)
    # val_loader.dataset is a BraTSDataset
    val_df = val_loader.dataset.df.copy()

    # Ensure lengths match
    if len(val_df) != len(all_preds):
        print(
            "Warning: Mismatch in validation set size and predictions during failure analysis."
        )
        return

    val_df["target"] = all_targets
    val_df["prediction"] = all_preds
    val_df["error"] = np.abs(val_df["target"] - val_df["prediction"])

    # 2. Extract Features for Correlation
    # We will use 'FLAIR_slices' as a proxy for scan complexity/volume.
    # We need to count files in the path_FLAIR directory.
    slice_counts = []
    for _, row in val_df.iterrows():
        flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
        try:
            # Fast count of files
            count = len(
                [
                    name
                    for name in os.listdir(flair_path)
                    if os.path.isfile(os.path.join(flair_path, name))
                ]
            )
        except Exception:
            count = 0
        slice_counts.append(count)

    val_df["flair_slice_count"] = slice_counts

    # 3. Compute Correlation
    if val_df["error"].std() > 0 and val_df["flair_slice_count"].std() > 0:
        corr, _ = pearsonr(val_df["error"], val_df["flair_slice_count"])
        print(f"Correlation between Error Magnitude and FLAIR Slice Count: {corr:.4f}")

        # Insight
        if abs(corr) > 0.1:
            print(
                "  -> Significant relationship detected. Model struggles with "
                + ("larger" if corr > 0 else "smaller")
                + " volumes."
            )
        else:
            print("  -> No significant relationship detected with scan volume.")
    else:
        print("Correlation could not be computed (constant values).")


def main():
    # 1. Setup
    # Override Config for Fast Baseline
    Config.EPOCHS = 10

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model Initialization
    print(f"Initializing Model: {Config.BACKBONE}...")
    model = GroupedEfficientNetV2().to(device)

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_auc = 0.0
    patience_counter = 0

    print(f"Starting Training Loop for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_auc = validate_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
        )

        # Checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Final Validation & Metric Printing
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Re-run validation to be absolutely sure of the metric on the best weights
    final_loss, final_auc = validate_epoch(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, device)

    # 8. Submission
    threshold = 0.6321818181818182
    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) > threshold ({threshold}). Generating submission..."
        )
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric ({final_auc}) <= threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
