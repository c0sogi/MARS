import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.train import run_training
from library.predict import generate_submission
from library.model import RMSHDNet
from library.utils import get_device
from library.data_loader import get_dataset_arrays, BraTSDataset


def main():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    # Override defaults to ensure quick execution while maintaining convergence
    Config.EPOCHS = 15

    # ==========================================
    # 2. Training
    # ==========================================
    # Trains the model and saves the best version to Config.MODEL_SAVE_PATH
    run_training(epochs=Config.EPOCHS)

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    device = get_device()

    # Load the best model saved during training
    model = RMSHDNet().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Error: Model file not found at {Config.MODEL_SAVE_PATH}")
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Load Validation Data
    # We use get_dataset_arrays to ensure we have aligned IDs for failure analysis
    X_val, y_val, ids_val = get_dataset_arrays(
        Config.VAL_METADATA_PATH, "val", load_cached_data=True
    )

    # Create DataLoader for efficient batch processing
    val_dataset = BraTSDataset(X_val, y_val, ids_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Essential to preserve order for ID mapping
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Run Inference
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            # Forward pass
            logits = model(data)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(target.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Final Metric
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    print(f"Final Validation Metric: {val_auc}")

    # Failure Analysis
    # Calculate error magnitude per sample
    errors = np.abs(all_targets - all_preds)

    # Load metadata to access input features (e.g., slice counts)
    val_meta_df = pd.read_parquet(Config.VAL_METADATA_PATH)

    # Create analysis dataframe
    # ids_val matches all_preds order because shuffle=False
    analysis_df = pd.DataFrame({"BraTS21ID": ids_val, "error": errors})

    # Merge with metadata
    analysis_df = analysis_df.merge(val_meta_df, on="BraTS21ID", how="left")

    # Feature: Total Slices (Sum of slices across all 4 modalities)
    def calculate_total_slices(row):
        count = 0
        for mod in ["flair", "t1w", "t1wce", "t2w"]:
            col = f"{mod}_paths"
            if col in row and isinstance(row[col], (list, np.ndarray)):
                count += len(row[col])
        return count

    analysis_df["total_slices"] = analysis_df.apply(calculate_total_slices, axis=1)

    # Calculate Correlation
    if len(analysis_df) > 1:
        corr_slices, _ = pearsonr(analysis_df["total_slices"], analysis_df["error"])
        print(f"Correlation between Error and Total Slices: {corr_slices}")

    # ==========================================
    # 4. Submission
    # ==========================================
    THRESHOLD = 0.6978181818181817

    if val_auc > THRESHOLD:
        generate_submission()
    else:
        print(
            f"Validation metric {val_auc} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
