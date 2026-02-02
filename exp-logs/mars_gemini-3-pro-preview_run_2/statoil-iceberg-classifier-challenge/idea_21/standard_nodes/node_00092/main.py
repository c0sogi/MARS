import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

# Import from library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import process_and_cache_data, IcebergDataset
from library.model import QPWBN
from library.train import run_training


def main():
    # 1. Setup Environment
    seed_everything(Config.SEED)
    logger = get_logger("runfile")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device}")

    # 2. Configure Fast Baseline
    # Override default configuration to ensure execution completes quickly.
    # Increasing epochs to 30 to allow better convergence.
    Config.NUM_EPOCHS = 30
    logger.info(f"Configured for Fast Baseline: NUM_EPOCHS={Config.NUM_EPOCHS}")

    # 3. Execute Training Pipeline
    # This runs Stratified 5-Fold CV and saves the best model for each fold to disk.
    logger.info("Starting Training Pipeline...")
    run_training()

    # 4. Load Validation Data
    # We load the specific validation split defined in metadata/val.csv as required.
    logger.info("Loading validation data...")
    data = process_and_cache_data(load_cached_data=True)

    X_val = data["X_val"]
    y_val = data["y_val"]
    inc_val = data["inc_val"]

    # Create Validation DataLoader
    val_dataset = IcebergDataset(X_val, inc_val, y_val, transform=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 5. Ensemble Inference on Validation Set
    logger.info("Running ensemble inference on validation set...")
    val_preds_accum = np.zeros(len(y_val))
    models_found = 0

    for fold_idx in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.MODEL_DIR, f"model_fold_{fold_idx}.pth")
        if not os.path.exists(model_path):
            logger.warning(
                f"Model for fold {fold_idx} not found at {model_path}. Skipping."
            )
            continue

        # Load Model
        model = QPWBN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models_found += 1

        fold_preds = []
        with torch.no_grad():
            for inputs, inc_angles, _ in val_loader:
                inputs = inputs.to(device)
                inc_angles = inc_angles.to(device)

                outputs = model(inputs, inc_angles)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_preds.extend(probs)

        val_preds_accum += np.array(fold_preds)

    if models_found == 0:
        logger.error("No trained models found. Exiting.")
        return

    # Average predictions across folds
    avg_val_preds = val_preds_accum / models_found

    # 6. Compute and Print Validation Metric
    # Clip predictions to prevent log(0) errors
    avg_val_preds_clipped = np.clip(avg_val_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y_val, avg_val_preds_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    logger.info("Performing failure analysis...")
    # Calculate error magnitude
    error_magnitude = np.abs(y_val - avg_val_preds)

    # Calculate features for correlation
    # X_val shape is (N, 75, 75, 3). Channel 0 is Band 1, Channel 1 is Band 2.
    # Note: Data is normalized, but relative differences remain valid for correlation.
    feat_inc = inc_val
    feat_b1_mean = np.mean(X_val[:, :, :, 0], axis=(1, 2))
    feat_b2_mean = np.mean(X_val[:, :, :, 1], axis=(1, 2))

    # Compute correlations
    corr_inc = np.corrcoef(error_magnitude, feat_inc)[0, 1]
    corr_b1 = np.corrcoef(error_magnitude, feat_b1_mean)[0, 1]
    corr_b2 = np.corrcoef(error_magnitude, feat_b2_mean)[0, 1]

    print("Failure Analysis - Correlation with Error Magnitude:")
    print(f"  Incidence Angle: {corr_inc}")
    print(f"  Band 1 Mean:     {corr_b1}")
    print(f"  Band 2 Mean:     {corr_b2}")

    # 8. Submission Generation
    # Always generate submission to ensure valid output.
    logger.info(f"Generating submission (Metric: {final_metric})...")

    # Load Test Data
    X_test = data["X_test"]
    inc_test = data["inc_test"]

    test_dataset = IcebergDataset(X_test, inc_test, y=None, transform=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_preds_accum = np.zeros(len(X_test))

    # Ensemble Inference on Test Set
    for fold_idx in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.MODEL_DIR, f"model_fold_{fold_idx}.pth")
        if not os.path.exists(model_path):
            continue

        model = QPWBN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for inputs, inc_angles in test_loader:
                inputs = inputs.to(device)
                inc_angles = inc_angles.to(device)

                outputs = model(inputs, inc_angles)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_preds.extend(probs)

        test_preds_accum += np.array(fold_preds)

    avg_test_preds = test_preds_accum / models_found

    # Create Submission DataFrame
    df_test_meta = pd.read_csv(Config.TEST_CSV)
    submission = pd.DataFrame({"id": df_test_meta["id"], "is_iceberg": avg_test_preds})

    # Save
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
