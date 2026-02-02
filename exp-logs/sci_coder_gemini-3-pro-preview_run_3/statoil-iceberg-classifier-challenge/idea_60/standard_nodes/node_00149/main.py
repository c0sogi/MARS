import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.model import (
    load_and_process_data,
    ISCI_CNN,
    predict_test,
    ShipIcebergDataset,
)
from library.train_eval import train_model, make_submission


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Training
    # Execute the 5-Fold CV workflow. This saves the best checkpoints for each fold.
    print("Starting 5-Fold Cross-Validation Training...")
    train_model(debug=False)

    # 3. Generate OOF Predictions for Validation
    # We need to manually generate OOF predictions to evaluate specifically on the
    # metadata-defined validation set without leakage.
    print("Generating Out-Of-Fold (OOF) predictions...")

    # Load raw data
    X_train, y_train, angles_train, ids_train, _, _, _ = load_and_process_data(
        load_cached_data=True
    )

    # Dictionary to store OOF predictions: ID -> Probability
    oof_preds_dict = {}

    # Re-create the Stratified K-Fold splits (deterministic due to seed)
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    device = Config.DEVICE

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        # Load the best model for this fold
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold}.pth"
        )
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint for fold {fold} not found.")
            continue

        model = ISCI_CNN().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        # Prepare validation data for this fold
        X_val = X_train[val_idx]
        ang_val = angles_train[val_idx]
        ids_val = ids_train[val_idx]

        # Leak-Free Imputation: Calculate median from TRAINING split of this fold
        ang_tr = angles_train[train_idx]
        valid_angles_tr = ang_tr[~np.isnan(ang_tr)]
        median_angle = np.median(valid_angles_tr) if len(valid_angles_tr) > 0 else 0.0

        # Apply imputation to validation data
        ang_val_imputed = np.where(np.isnan(ang_val), median_angle, ang_val)

        # Create DataLoader
        val_ds = ShipIcebergDataset(X_val, None, ang_val_imputed, transform=None)
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Predict
        preds = predict_test(model, val_loader, device)

        # Store predictions mapped by ID
        for id_, pred in zip(ids_val, preds):
            oof_preds_dict[id_] = pred

    # 4. Evaluation on Hold-out Set
    # Load the validation metadata
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Metadata file not found: {val_meta_path}")

    df_val_meta = pd.read_csv(val_meta_path)
    val_ids = df_val_meta["id"].values
    val_targets = df_val_meta["is_iceberg"].values

    # Retrieve predictions for the hold-out set
    val_probs = []
    valid_targets_filtered = []

    for id_, target in zip(val_ids, val_targets):
        if id_ in oof_preds_dict:
            val_probs.append(oof_preds_dict[id_])
            valid_targets_filtered.append(target)
        else:
            print(f"Warning: ID {id_} from metadata not found in OOF predictions.")

    # Compute Metric
    final_metric = log_loss(valid_targets_filtered, val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    val_probs = np.array(val_probs)
    val_targets = np.array(valid_targets_filtered)
    errors = np.abs(val_targets - val_probs)

    # Re-map IDs to indices to retrieve features for the validation subset
    id_to_idx = {id_: i for i, id_ in enumerate(ids_train)}
    val_indices = [id_to_idx[id_] for id_ in val_ids if id_ in id_to_idx]

    # Extract features
    X_val_subset = X_train[val_indices]

    # Compute Image Statistics
    band_1_mean = np.mean(X_val_subset[:, 0, :, :], axis=(1, 2))
    band_2_mean = np.mean(X_val_subset[:, 1, :, :], axis=(1, 2))
    band_1_std = np.std(X_val_subset[:, 0, :, :], axis=(1, 2))
    band_2_std = np.std(X_val_subset[:, 1, :, :], axis=(1, 2))

    # Incidence Angle (from metadata)
    inc_angles = df_val_meta["inc_angle"].values

    # Correlations
    correlations = {}

    # Handle NaNs in inc_angle
    valid_angle_mask = ~np.isnan(inc_angles)
    if np.sum(valid_angle_mask) > 1:
        corr_angle, _ = pearsonr(errors[valid_angle_mask], inc_angles[valid_angle_mask])
        correlations["inc_angle"] = corr_angle
    else:
        correlations["inc_angle"] = np.nan

    correlations["band_1_mean"] = pearsonr(errors, band_1_mean)[0]
    correlations["band_2_mean"] = pearsonr(errors, band_2_mean)[0]
    correlations["band_1_std"] = pearsonr(errors, band_1_std)[0]
    correlations["band_2_std"] = pearsonr(errors, band_2_std)[0]

    print("Correlation between Error Magnitude and Features:")
    for feature, corr in correlations.items():
        print(f"  {feature}: {corr:.4f}")

    # 6. Submission
    threshold = 0.17174082291273365
    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")
        make_submission()
    else:
        print(f"\nMetric {final_metric} >= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
