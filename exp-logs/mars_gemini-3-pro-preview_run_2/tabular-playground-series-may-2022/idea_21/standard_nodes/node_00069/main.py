import pandas as pd
import numpy as np
import torch
import sys
import os
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------------------
# 1. Configuration and Patching
# ------------------------------------------------------------------------------
from library.config import Config

# Configure for Fast Baseline
Config.EPOCHS = 10
Config.BATCH_SIZE = 2048

# Patch to handle potential dataset bug where f_27 (string) is treated as continuous
# We rename the sequence feature in the dataframe so the continuous column selector
# (which looks for 'f_27') skips it, while the sequence processor (which looks for
# Config.SEQUENCE_FEATURE) finds the renamed column.
original_seq_feature = Config.SEQUENCE_FEATURE  # "f_27"
Config.SEQUENCE_FEATURE = "seq_f_27"

original_read_csv = pd.read_csv


def patched_read_csv(*args, **kwargs):
    df = original_read_csv(*args, **kwargs)
    # Check if this is one of the data files containing the sequence feature
    if original_seq_feature in df.columns:
        df.rename(columns={original_seq_feature: Config.SEQUENCE_FEATURE}, inplace=True)
    return df


pd.read_csv = patched_read_csv

# ------------------------------------------------------------------------------
# 2. Library Imports
# ------------------------------------------------------------------------------
from library.dataset import get_dataloaders
from library.network import HybridModel
from library.engine import run_training, generate_submission, validate, set_seed


# ------------------------------------------------------------------------------
# 3. Main Execution Flow
# ------------------------------------------------------------------------------
def main():
    # Ensure reproducibility
    set_seed(Config.SEED)

    print(
        f"Starting execution with EPOCHS={Config.EPOCHS}, BATCH_SIZE={Config.BATCH_SIZE}"
    )

    # --- Training ---
    print("Starting Training...")
    # run_training trains the model and saves the best checkpoint to Config.MODEL_PATH
    run_training()

    # --- Validation Assessment ---
    print("Performing Final Validation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model checkpoint
    model = HybridModel().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded best model from {Config.MODEL_PATH}")
    else:
        print("Error: Model checkpoint not found.")
        return

    # Get validation data
    # We use load_cached_data=True to leverage the data processed during training
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    model.eval()
    all_targets = []
    all_preds = []
    all_continuous = []

    # Inference loop (no grad for speed)
    with torch.no_grad():
        for batch in val_loader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["target"].to(device)

            outputs = model(continuous, sequence)
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())
            all_continuous.append(continuous.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_continuous = np.concatenate(all_continuous, axis=0)

    # Compute Metric
    val_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {val_auc}")

    # --- Failure Analysis ---
    print("Performing Failure Analysis...")
    errors = np.abs(all_targets - all_preds)

    # Reconstruct feature names. The continuous data has 30 columns.
    # Original indices were 0-30. f_27 was excluded by our patch.
    feat_names = [f"f_{i:02d}" for i in range(31) if i != 27]

    # Create DataFrame for analysis
    df_analysis = pd.DataFrame(all_continuous, columns=feat_names)
    df_analysis["error"] = errors

    # Compute correlations
    correlations = df_analysis.corr()["error"].drop("error")
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    print("Top 5 features correlated with error magnitude:")
    print(top_correlations)

    # --- Submission ---
    threshold = 0.9970005855169476
    if val_auc > threshold:
        print(f"Validation metric {val_auc} > {threshold}. Generating submission...")
        generate_submission()
    else:
        print(f"Validation metric {val_auc} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
