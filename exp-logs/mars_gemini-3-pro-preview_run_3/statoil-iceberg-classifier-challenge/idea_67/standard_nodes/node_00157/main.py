import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.train as train_lib


def main():
    # 1. Initialization
    utils.set_seed(config.SEED)
    utils.disable_warnings()

    print("Initializing Fast Baseline Run...")

    # 2. Training (5-Fold CV)
    # This trains models for all 5 folds and saves checkpoints to ./working/idea_67/checkpoints/
    # Using cached data for speed as requested.
    print("Starting 5-Fold Cross-Validation Training...")
    train_lib.train_all_folds(load_cached_data=True)

    # 3. Validation on Hold-out Set
    # Requirement: Load hold-out validation dataset using metadata
    print("Loading validation metadata and data...")
    val_meta_path = config.VAL_META
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Validation metadata not found at {val_meta_path}")

    val_df = pd.read_csv(val_meta_path)
    val_ids = set(val_df["id"].values)

    # Load all preprocessed data to extract the specific validation samples
    X_all, angles_all, y_all, ids_all, _, _, _ = data_loader.load_data(
        load_cached_data=True
    )

    # Filter for validation samples based on IDs in metadata/val.csv
    val_mask = np.array([uid in val_ids for uid in ids_all])

    X_val = X_all[val_mask]
    y_val = y_all[val_mask]
    ids_val = ids_all[val_mask]
    angles_val_raw = angles_all[val_mask]

    # Handle missing angles for validation
    # Impute with global median of all training data (standard practice for inference)
    global_median = np.nanmedian(angles_all)
    angles_val = angles_val_raw.copy()
    angles_val[np.isnan(angles_val)] = global_median

    # Create Validation DataLoader
    val_dataset = data_loader.IcebergDataset(
        X_val, angles_val, y_val, ids_val, transform=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 4. Inference (Ensemble of 5 folds)
    print("Running inference on hold-out validation set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_lib.LeakyAttentiveIsomorphicCNN().to(device)

    # Array to store accumulated probabilities
    ensemble_preds = np.zeros(len(val_dataset))

    folds_found = 0
    for fold_idx in range(config.NUM_FOLDS):
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint for fold {fold_idx} not found.")
            continue

        # Load model state
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                angles = batch["angle"].to(device)

                # Forward pass
                logits = model(images, angles)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                fold_preds.extend(probs)

        ensemble_preds += np.array(fold_preds)
        folds_found += 1

    if folds_found > 0:
        ensemble_preds /= folds_found
    else:
        raise RuntimeError("No models were trained/found. Cannot perform validation.")

    # 5. Compute Metric
    # Log Loss (BCE)
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    preds_clipped = np.clip(ensemble_preds, epsilon, 1 - epsilon)
    final_metric = log_loss(y_val, preds_clipped)

    # Print the required metric
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("Performing failure analysis...")
    # Calculate error magnitude
    errors = np.abs(y_val - ensemble_preds)

    # Calculate image statistics for correlation
    # X_val is (N, 3, 75, 75). Channel 0 is HH, Channel 1 is HV.
    b1_mean = np.mean(X_val[:, 0, :, :], axis=(1, 2))
    b1_std = np.std(X_val[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X_val[:, 1, :, :], axis=(1, 2))
    b2_std = np.std(X_val[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angles_val,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
        }
    )

    # Compute correlation
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 7. Submission Generation
    # Threshold check
    THRESHOLD = 0.17174082291273365

    if final_metric < THRESHOLD:
        print(f"Validation metric {final_metric} is better than threshold {THRESHOLD}.")
        print("Generating submission file...")
        train_lib.generate_submission(load_cached_data=True)
    else:
        print(f"Validation metric {final_metric} did not meet threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
