import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import from library
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data_loader import process_data, IcebergDataset
from library.model import HybridSECNN
from library.train import run_fold


def main():
    # Enforce reproducibility
    set_seed(Config.SEED)

    # Override Config for fast baseline execution as requested
    # Reducing epochs from 75 to 50 to ensure completion within the 2-hour soft limit
    # while allowing sufficient convergence for the SE-CNN.
    Config.EPOCHS = 50

    print("Processing data...")
    # Load all data (cached if available)
    # X_all corresponds to the full train.json data
    X_all, angles_all, y_all, X_test, angles_test, ids_test = process_data(
        load_cached_data=True
    )

    # --- Training Phase ---
    print("Starting training (5 Folds)...")
    for fold in range(Config.N_FOLDS):
        run_fold(fold)

    # --- Validation Assessment ---
    print("Performing validation assessment...")

    # Load hold-out validation metadata
    # We must evaluate on the specific subset defined in metadata/val.csv
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    val_indices = val_meta["original_index"].values

    # Extract hold-out set from the full loaded data
    X_val_holdout = X_all[val_indices]
    angles_val_holdout = angles_all[val_indices]
    y_val_holdout = y_all[val_indices]

    # Prepare DataLoader for Validation
    val_dataset = IcebergDataset(
        X_val_holdout, angles_val_holdout, y_val_holdout, transform=None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Ensemble Models
    device = torch.device(Config.DEVICE)
    models = []
    print(f"Loading {Config.N_FOLDS} models for ensemble inference...")
    for fold in range(Config.N_FOLDS):
        model = HybridSECNN()
        checkpoint_path = f"model_fold_{fold}.pth"
        checkpoint = load_checkpoint(checkpoint_path, device=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        models.append(model)

    # Ensemble Inference
    preds_accum = []
    targets_accum = []

    with torch.no_grad():
        for images, angles, labels in val_loader:
            images = images.to(device)
            angles = angles.to(device)

            # Collect predictions from all models
            batch_preds = []
            for model in models:
                logits = model(images, angles)
                probs = torch.sigmoid(logits)
                batch_preds.append(probs.cpu().numpy())

            # Average probabilities (Ensemble)
            avg_preds = np.mean(batch_preds, axis=0)
            preds_accum.extend(avg_preds)
            targets_accum.extend(labels.numpy())

    y_pred = np.array(preds_accum).flatten()
    y_true = np.array(targets_accum).flatten()

    # Compute Final Metric
    final_metric = log_loss(y_true, y_pred)
    # Print full precision metric as required
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("Performing failure analysis...")
    errors = np.abs(y_true - y_pred)

    # Calculate features for correlation analysis
    # X_val_holdout is (N, 3, 75, 75). Band 0 is HH, Band 1 is HV.
    b1_mean = np.mean(X_val_holdout[:, 0, :, :], axis=(1, 2))
    b1_std = np.std(X_val_holdout[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X_val_holdout[:, 1, :, :], axis=(1, 2))
    b2_std = np.std(X_val_holdout[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angles_val_holdout,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
        }
    )

    # Calculate and print correlations
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False)
    )
    print("Correlation between Error and Features:")
    print(correlations)

    # --- Submission Generation ---
    threshold = 0.18120490171618245
    if final_metric < threshold:
        print("Metric passed threshold. Generating submission...")

        test_dataset = IcebergDataset(X_test, angles_test, ids=ids_test, transform=None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for images, angles, ids in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                batch_preds = []
                for model in models:
                    logits = model(images, angles)
                    probs = torch.sigmoid(logits)
                    batch_preds.append(probs.cpu().numpy())

                avg_preds = np.mean(batch_preds, axis=0)
                test_preds.extend(avg_preds)
                test_ids.extend(ids)

        submission = pd.DataFrame(
            {"id": test_ids, "is_iceberg": np.array(test_preds).flatten()}
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"Metric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
