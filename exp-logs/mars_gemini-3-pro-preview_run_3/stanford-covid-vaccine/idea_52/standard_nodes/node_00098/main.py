import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
import library.utils
import library.train
from library.utils import set_seed, calculate_metric
from library.data import get_dataloaders
from library.model import DeepStabilizedBiGRU
from library.train import Trainer


# =========================================================================
# Monkey-Patching for Validation Stability
# =========================================================================
# The provided Trainer passes full-length targets (107) to calculate_metric.
# However, calculate_metric expects targets to match the scored length (68)
# because it slices predictions internally. We patch this to ensure shape alignment.
def patched_calculate_metric(preds, targets):
    # If targets are full sequence length (107), slice them to scored length (68)
    if targets.shape[1] == Config.SEQ_LEN:
        targets = targets[:, : Config.SEQ_SCORED, :]
    return library.utils.calculate_metric(preds, targets)


# Apply the patch to the library.train module
library.train.calculate_metric = patched_calculate_metric


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for Fast Baseline execution
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 32

    print(f"Running with Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # Set reproducibility
    set_seed(Config.SEED)

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # =========================================================================
    # 3. Model Initialization & Training
    # =========================================================================
    print("Initializing Deep Stabilized BiGRU Model...")
    model = DeepStabilizedBiGRU()
    model.to(Config.DEVICE)

    print("Starting Training...")
    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit()

    # =========================================================================
    # 4. Final Validation & Metric Calculation
    # =========================================================================
    print(f"Loading best model from {Config.MODEL_SAVE_PATH}...")
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    print("Performing Final Validation Inference...")
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for inputs, pair_indices, targets in val_loader:
            inputs = inputs.to(Config.DEVICE)
            pair_indices = pair_indices.to(Config.DEVICE)

            outputs = model(inputs, pair_indices)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Metric (Ensure targets are sliced for the metric calculation)
    val_targets_scored = val_targets[:, : Config.SEQ_SCORED, :]
    final_metric = calculate_metric(val_preds, val_targets_scored)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 5. Failure Analysis
    # =========================================================================
    print("Running Failure Analysis...")

    # Load metadata for analysis
    df_val = pd.read_parquet(Config.VAL_METADATA_PATH)

    # Calculate error per sample (RMSE on scored columns/positions)
    # Scored columns indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    # Slice predictions to match scored targets
    preds_sliced = val_preds[:, : Config.SEQ_SCORED, :]

    # Compute squared difference
    diff = preds_sliced[:, :, scored_indices] - val_targets_scored[:, :, scored_indices]

    # Mean squared error per sample (averaged over sequence and columns)
    mse_per_sample = np.mean(diff**2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    df_val["error"] = rmse_per_sample

    # Feature Engineering for Correlation
    df_val["A_pct"] = df_val["sequence"].apply(lambda x: x.count("A") / len(x))
    df_val["G_pct"] = df_val["sequence"].apply(lambda x: x.count("G") / len(x))
    df_val["C_pct"] = df_val["sequence"].apply(lambda x: x.count("C") / len(x))
    df_val["U_pct"] = df_val["sequence"].apply(lambda x: x.count("U") / len(x))

    # Calculate correlations
    corr_cols = [
        "error",
        "signal_to_noise",
        "SN_filter",
        "A_pct",
        "G_pct",
        "C_pct",
        "U_pct",
    ]
    # Filter columns that exist in dataframe
    available_cols = [c for c in corr_cols if c in df_val.columns]

    correlations = df_val[available_cols].corr()["error"].sort_values(ascending=False)
    print("Correlation between Error and Features:")
    print(correlations.drop("error"))

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    THRESHOLD = 0.5884495377540588

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")

        test_preds = []

        # Inference on Test Set
        with torch.no_grad():
            for inputs, pair_indices, _ in test_loader:
                inputs = inputs.to(Config.DEVICE)
                pair_indices = pair_indices.to(Config.DEVICE)

                outputs = model(inputs, pair_indices)
                test_preds.append(outputs.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)  # Shape: (240, 107, 5)

        # Load Test IDs
        df_test = pd.read_parquet(Config.TEST_METADATA_PATH)
        ids = df_test["id"].values

        # Prepare Submission DataFrame
        # We need to flatten the predictions: (Samples * Seq_Len, Targets)
        n_samples, seq_len, n_targets = test_preds.shape
        preds_flat = test_preds.reshape(-1, n_targets)

        # Generate 'id_seqpos' column
        # Repeat IDs for each sequence position
        ids_repeated = np.repeat(ids, seq_len)
        # Tile sequence positions (0..106) for each sample
        seqpos_tiled = np.tile(np.arange(seq_len), n_samples)

        id_seqpos_list = [f"{i}_{s}" for i, s in zip(ids_repeated, seqpos_tiled)]

        # Create DataFrame
        submission_df = pd.DataFrame(
            preds_flat,
            columns=["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"],
        )
        submission_df.insert(0, "id_seqpos", id_seqpos_list)

        # Save
        submission_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")

    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
