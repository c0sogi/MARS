import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import library modules
from library import config, utils, model, data_loader, train


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Train 5 Folds
    # The training function handles data loading, model init, training loop, and checkpoint saving.
    print("Starting 5-Fold Cross-Validation Training...")
    for fold in range(config.NUM_FOLDS):
        train.train_fold(fold)

    # 3. Load Hold-out Validation Set (Metadata-based)
    print("\nLoading Hold-out Validation Set based on metadata...")
    val_meta_path = config.VAL_META_PATH
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Validation metadata not found at {val_meta_path}")

    val_df = pd.read_csv(val_meta_path)
    val_ids = set(val_df["id"].values)

    # Load full raw data to retrieve features for the validation IDs
    # We use process_json_data directly to get IDs and raw features
    X_full, angles_full, y_full, ids_full = data_loader.process_json_data(
        config.TRAIN_JSON_PATH, is_train=True
    )

    # Filter for validation samples
    val_mask = np.array([uid in val_ids for uid in ids_full])
    X_val = X_full[val_mask]
    angles_val = angles_full[val_mask]
    y_val = y_full[val_mask]

    # Impute missing angles in validation set
    # We calculate median from the non-validation (training) part to avoid leakage
    train_mask = ~val_mask
    train_angles_valid = angles_full[train_mask]
    # Filter NaNs
    train_angles_valid = train_angles_valid[~np.isnan(train_angles_valid)]
    angle_median = np.median(train_angles_valid)

    # Fill NaNs in validation angles
    angles_val[np.isnan(angles_val)] = angle_median

    # Create Validation DataLoader
    val_dataset = data_loader.IcebergDataset(X_val, angles_val, y_val, transform=None)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # 4. Validation Inference (Ensemble)
    print("Running inference on validation set...")
    ensemble_val_preds = np.zeros((len(y_val),))

    for fold in range(config.NUM_FOLDS):
        # Initialize model and load weights
        net = model.EAP_CNN().to(device)
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, f"model_best_fold_{fold}.pth"
        )
        utils.load_checkpoint(checkpoint_path, net)
        net.eval()

        fold_preds = []
        with torch.no_grad():
            for images, angles, _ in val_loader:
                images = images.to(device)
                angles = angles.to(device)
                outputs = net(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_preds.extend(probs)

        ensemble_val_preds += np.array(fold_preds)

    # Average predictions
    ensemble_val_preds /= config.NUM_FOLDS

    # 5. Calculate Metric
    # Clip predictions to prevent log(0)
    ensemble_val_preds_clipped = np.clip(ensemble_val_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y_val, ensemble_val_preds_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(y_val - ensemble_val_preds)

    # Extract features for correlation analysis
    # X_val shape: (N, 3, 75, 75). Channel 0: HH, Channel 1: HV
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
    corr = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(corr)

    # 7. Submission
    THRESHOLD = 0.1806015565870406

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is below threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data (using cached data via data_loader)
        # load_and_process_data returns: X_train, angles_train, y_train, X_test, angles_test, ids_test
        _, _, _, X_test, angles_test, ids_test = data_loader.load_and_process_data(
            load_cached_data=True
        )

        test_dataset = data_loader.IcebergDataset(
            X_test, angles_test, y=None, transform=None
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )

        ensemble_test_preds = np.zeros((len(ids_test),))

        for fold in range(config.NUM_FOLDS):
            net = model.EAP_CNN().to(device)
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR, f"model_best_fold_{fold}.pth"
            )
            utils.load_checkpoint(checkpoint_path, net)
            net.eval()

            fold_preds = []
            with torch.no_grad():
                for images, angles in test_loader:
                    images = images.to(device)
                    angles = angles.to(device)
                    outputs = net(images, angles)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    fold_preds.extend(probs)

            ensemble_test_preds += np.array(fold_preds)

        # Average predictions
        ensemble_test_preds /= config.NUM_FOLDS

        # Create submission file
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        sub_df = pd.DataFrame({"id": ids_test, "is_iceberg": ensemble_test_preds})

        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric ({final_metric}) >= Threshold ({THRESHOLD}). Submission generation skipped."
        )


if __name__ == "__main__":
    main()
