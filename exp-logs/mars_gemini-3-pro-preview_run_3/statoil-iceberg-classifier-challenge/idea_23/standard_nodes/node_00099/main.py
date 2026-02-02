import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import from provided library files
from library.config import Config
from library.utils import set_seed, log_message
from library.model import MSMANet
from library.data_loader import process_and_cache_data, IcebergDataset
from library.train_eval import train_fold


def predict(model, loader, device):
    """
    Generate predictions using the model.
    Returns:
        probs (np.array): Probability of iceberg (class 1).
        targets (np.array): True labels (if available, else None).
    """
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            # Unpack batch depending on whether it has labels
            if len(batch) == 3:
                images, angles, labels = batch
                all_targets.append(labels.numpy())
            else:
                images, angles, ids = batch

            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs).flatten()
    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets).flatten()
        return all_probs, all_targets
    else:
        return all_probs, None


def main():
    # 1. Configuration Overrides for Fast Baseline
    Config.EPOCHS = 50
    Config.PATIENCE = 10
    Config.print_config()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)

    # 2. Load Data
    log_message("Loading data...")
    train_data_dict, test_data_dict = process_and_cache_data(load_cached_data=True)

    # Prepare Test Loader
    test_dataset = IcebergDataset(
        test_data_dict["X"],
        test_data_dict["angles"],
        ids=test_data_dict["ids"],
        transform=None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Prepare Fixed Validation Loader (based on metadata/val.csv)
    # We need to map the IDs in val.csv to the indices in the full train_data
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    val_ids_set = set(val_meta["id"].values)

    # Find indices in the loaded training data that match the validation set IDs
    full_ids = train_data_dict["ids"]
    val_indices = [i for i, img_id in enumerate(full_ids) if img_id in val_ids_set]

    X_val_fixed = train_data_dict["X"][val_indices]
    y_val_fixed = train_data_dict["y"][val_indices]
    angles_val_fixed = train_data_dict["angles"][val_indices]
    ids_val_fixed = train_data_dict["ids"][val_indices]

    val_fixed_dataset = IcebergDataset(
        X_val_fixed,
        angles_val_fixed,
        labels=y_val_fixed,
        ids=ids_val_fixed,
        transform=None,
    )
    val_fixed_loader = DataLoader(
        val_fixed_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    log_message(f"Fixed Validation Set Size: {len(val_fixed_dataset)}")

    # 3. Training Loop (5 Folds)
    fold_val_preds = []
    fold_test_preds = []

    for fold in range(Config.NUM_FOLDS):
        # Train the fold
        train_fold(fold)

        # Load Best Model for this fold
        model = MSMANet().to(device)
        checkpoint_path = os.path.join(
            Config.WORKING_DIR, f"model_best_fold_{fold}.pth"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

        # Predict on Fixed Validation Set
        val_probs, _ = predict(model, val_fixed_loader, device)
        fold_val_preds.append(val_probs)

        # Predict on Test Set
        test_probs, _ = predict(model, test_loader, device)
        fold_test_preds.append(test_probs)

        # Clear memory
        del model
        torch.cuda.empty_cache()

    # 4. Ensembling
    avg_val_preds = np.mean(fold_val_preds, axis=0)
    avg_test_preds = np.mean(fold_test_preds, axis=0)

    # 5. Evaluation
    final_metric = log_loss(y_val_fixed, avg_val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    log_message("\n--- Failure Analysis ---")
    errors = np.abs(y_val_fixed - avg_val_preds)

    # Calculate features for correlation
    # Band 1 Mean
    b1_means = np.mean(X_val_fixed[:, 0, :, :], axis=(1, 2))
    # Band 2 Mean
    b2_means = np.mean(X_val_fixed[:, 1, :, :], axis=(1, 2))
    # Incidence Angle
    angles = angles_val_fixed

    # Correlations
    corr_angle = np.corrcoef(errors, angles)[0, 1]
    corr_b1 = np.corrcoef(errors, b1_means)[0, 1]
    corr_b2 = np.corrcoef(errors, b2_means)[0, 1]

    print(f"Correlation (Error vs Inc Angle): {corr_angle:.4f}")
    print(f"Correlation (Error vs Band 1 Mean): {corr_b1:.4f}")
    print(f"Correlation (Error vs Band 2 Mean): {corr_b2:.4f}")

    # 7. Submission
    threshold = 0.18120490171618245
    if final_metric < threshold:
        log_message(f"\nMetric {final_metric} < {threshold}. Generating submission...")

        submission = pd.DataFrame(
            {"id": test_data_dict["ids"], "is_iceberg": avg_test_preds}
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        log_message(f"Submission saved to {Config.SUBMISSION_PATH}")

        # Validate submission format
        sample = pd.read_csv(Config.SAMPLE_SUBMISSION)
        if submission.shape == sample.shape and set(submission.columns) == set(
            sample.columns
        ):
            log_message("Submission format validated.")
        else:
            log_message("WARNING: Submission format mismatch.")
    else:
        log_message(f"\nMetric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
