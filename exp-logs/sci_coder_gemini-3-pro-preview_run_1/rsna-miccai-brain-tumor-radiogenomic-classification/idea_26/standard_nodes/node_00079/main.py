import os
import re
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import library modules
from library.utils import set_seed, get_device, load_checkpoint
from library.dataset import SIADSDataset, get_processed_metadata
from library.model import SIA_DS_EfficientNet
from library.training import run_fold
from library.inference import predict_test_set

# Constants
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_PATH = "./submission/submission.csv"
NUM_FOLDS = 5
NUM_EPOCHS = 6
BATCH_SIZE = 32
PATIENCE = 3
SEED = 42
THRESHOLD = 0.6705454545454544


def extract_file_index(path):
    """Extracts the integer index from a DICOM filename (e.g., Image-123.dcm -> 123)."""
    if pd.isna(path):
        return 0
    match = re.search(r"Image-(\d+)\.dcm", str(path))
    return int(match.group(1)) if match else 0


def main():
    # Ensure reproducibility
    set_seed(SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 1. Load Metadata
    # We load both train and val metadata.
    # We will use 'train_metadata.csv' for the Cross-Validation training loop.
    # We will use 'val_metadata.csv' as the fixed Hold-Out set for the final metric reporting.
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")

    # Load and process metadata (caching handles the heavy lifting)
    print("Loading metadata...")
    df_train_cv = get_processed_metadata(
        train_meta_path, "train", load_cached_data=True
    )
    df_holdout = get_processed_metadata(val_meta_path, "val", load_cached_data=True)

    # 2. 5-Fold Cross Validation Training
    # We split the training data into 5 folds
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    y_cv = df_train_cv["MGMT_value"].values

    fold_aucs = []

    print(f"\nStarting {NUM_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train_cv, y_cv)):
        print(f"\n=== Fold {fold} ===")

        # Split Data
        df_fold_train = df_train_cv.iloc[train_idx].reset_index(drop=True)
        df_fold_val = df_train_cv.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        ds_train = SIADSDataset(df_fold_train, phase="train")
        ds_val = SIADSDataset(df_fold_val, phase="val")

        # Create Loaders
        # num_workers=2 is safe for most environments
        dl_train = DataLoader(
            ds_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        dl_val = DataLoader(
            ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )

        # Init Model
        model = SIA_DS_EfficientNet(num_classes=1, drop_rate=0.3)

        # Train
        save_path = os.path.join(WORKING_DIR, f"best_model_fold{fold}.pth")
        best_auc = run_fold(
            model,
            dl_train,
            dl_val,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            save_path=save_path,
        )
        fold_aucs.append(best_auc)

    print(f"\nAverage CV AUC: {np.mean(fold_aucs):.6f}")

    # 3. Final Validation on Hold-Out Set
    print("\n=== Final Validation on Hold-Out Set ===")

    # Create Hold-out Loader
    ds_holdout = SIADSDataset(df_holdout, phase="val")
    dl_holdout = DataLoader(
        ds_holdout, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    # Ensemble Inference
    model_paths = [
        os.path.join(WORKING_DIR, f"best_model_fold{i}.pth") for i in range(NUM_FOLDS)
    ]

    holdout_preds_accum = np.zeros(len(ds_holdout))
    holdout_targets = []

    # Get targets once
    for _, targets in dl_holdout:
        holdout_targets.extend(targets.numpy())
    holdout_targets = np.array(holdout_targets)

    valid_models_count = 0

    for model_path in model_paths:
        if not os.path.exists(model_path):
            continue

        # Re-init model
        model = SIA_DS_EfficientNet(num_classes=1, drop_rate=0.3)
        load_checkpoint(model_path, model, device=device)

        model.to(device)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for inputs, _ in dl_holdout:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_preds.extend(probs)

        holdout_preds_accum += np.array(fold_preds)
        valid_models_count += 1

    if valid_models_count > 0:
        avg_preds = holdout_preds_accum / valid_models_count
    else:
        # Fallback if no models trained (should not happen)
        avg_preds = np.zeros(len(ds_holdout))

    # Calculate Metric
    final_metric = roc_auc_score(holdout_targets, avg_preds)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(holdout_targets - avg_preds)

    # Construct Analysis DataFrame
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "target": holdout_targets,
            "pred": avg_preds,
            "BraTS21ID": df_holdout["BraTS21ID"].values,
        }
    )

    # Feature Engineering for Correlation: Estimated ROI Depth
    # We use the difference in indices between 55% and 45% depth to estimate the "thickness" or depth of the ROI
    # We use FLAIR channel as proxy
    roi_depths = []
    for idx, row in df_holdout.iterrows():
        try:
            idx_45 = extract_file_index(row.get("flair_45", ""))
            idx_55 = extract_file_index(row.get("flair_55", ""))
            # The difference represents 10% of the ROI. Multiply by 10 to get approx total slices.
            depth = (idx_55 - idx_45) * 10
            roi_depths.append(depth)
        except:
            roi_depths.append(0)

    df_analysis["est_roi_depth"] = roi_depths

    # Calculate correlations
    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation of Error with features:")
    print(correlations)

    # 5. Submission
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_test_set(
            model_dir=WORKING_DIR,
            output_path=SUBMISSION_PATH,
            metadata_dir=METADATA_DIR,
            batch_size=BATCH_SIZE,
            num_folds=NUM_FOLDS,
            load_cached_data=True,
        )
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
