import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import glob

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_device, print_metrics
from library.data_loader import process_dataset, BraTSDataset
from library.model import AsymmetricEfficientNet
from library.train import train_epoch, validate
from library.evaluate import run_inference


def perform_failure_analysis(model, val_loader, val_df, device):
    """
    Analyzes model errors on the validation set.
    Calculates correlation between error magnitude and metadata features.
    """
    print("\n--- Failure Analysis ---")
    model.eval()
    all_probs = []
    all_targets = []

    # Get predictions
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(targets.numpy())

    all_probs = np.array(all_probs).flatten()
    all_targets = np.array(all_targets).flatten()

    # Calculate Error Magnitude
    errors = np.abs(all_targets - all_probs)

    # Add to DataFrame
    val_df = val_df.copy()
    val_df["error"] = errors
    val_df["prediction"] = all_probs

    # Extract simple structural features for correlation
    # We'll use FLAIR slice count as a proxy for volume/scan complexity
    # based on the EDA findings.
    flair_counts = []
    for idx, row in val_df.iterrows():
        try:
            flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
            # Fast count of files
            count = len(
                [
                    name
                    for name in os.listdir(flair_path)
                    if os.path.isfile(os.path.join(flair_path, name))
                ]
            )
            flair_counts.append(count)
        except:
            flair_counts.append(0)

    val_df["flair_slices"] = flair_counts

    # Calculate correlations
    # 1. Error vs Target (Class Bias)
    corr_target = val_df["error"].corr(val_df["MGMT_value"])
    print(f"Correlation (Error vs MGMT_value): {corr_target:.4f}")

    # 2. Error vs Volume Depth (FLAIR slices)
    corr_slices = val_df["error"].corr(val_df["flair_slices"])
    print(f"Correlation (Error vs FLAIR_slices): {corr_slices:.4f}")

    # 3. Error vs ID (Temporal/Batch drift)
    corr_id = val_df["error"].corr(val_df["BraTS21ID"])
    print(f"Correlation (Error vs BraTS21ID): {corr_id:.4f}")


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = get_device()

    # 2. Data Loading
    # Using load_cached_data=True to utilize any existing preprocessed arrays
    train_data, train_labels = process_dataset(
        os.path.join(Config.METADATA_DIR, "train.csv"),
        "train_cache",
        load_cached_data=True,
    )

    val_data, val_labels = process_dataset(
        os.path.join(Config.METADATA_DIR, "val.csv"), "val_cache", load_cached_data=True
    )

    # Create Datasets
    # Transform=True for training (Geometric Augmentations)
    train_dataset = BraTSDataset(train_data, train_labels, transform=True)
    val_dataset = BraTSDataset(val_data, val_labels, transform=False)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = AsymmetricEfficientNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Limit epochs to Config.EPOCHS (10) for fast baseline
    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print_metrics(
            epoch + 1, Config.EPOCHS, train_loss, train_auc, val_loss, val_auc
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                break

    # 5. Final Validation Metric
    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Re-run validation on full set to ensure accurate metric reporting
    final_loss, final_auc = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    perform_failure_analysis(model, val_loader, val_df, device)

    # 7. Submission
    # Threshold check
    THRESHOLD = 0.6321818181818182

    if final_auc > THRESHOLD:
        print(f"Validation metric {final_auc} > {THRESHOLD}. Generating submission...")
        # Use the provided evaluate module which handles TTA and submission file generation
        # We need to ensure the best model is saved where run_inference expects it (Config.WORKING_DIR/best_model.pth)
        # which we did in the training loop.
        run_inference()
    else:
        print(
            f"Validation metric {final_auc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
