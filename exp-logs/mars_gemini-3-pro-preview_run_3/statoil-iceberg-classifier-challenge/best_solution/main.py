import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

# Import provided library modules
import library.config
from library.config import load_and_process_data, DEVICE, SEED, BATCH_SIZE
from library.utils import set_seed, get_logger
from library.train_eval import run_fold
from library.model import IDPH_CNN
from library.data_loader import IcebergDataset, get_test_loader

# Override configuration for fast baseline execution
# 25 epochs is sufficient for this small dataset and ensures execution within time limits
library.config.NUM_EPOCHS = 25


def main():
    # 1. Setup
    set_seed(SEED)
    logger = get_logger("Runfile")

    logger.info("Starting Runfile execution...")

    # 2. Load Data
    # We load the full training data to perform CV and OOF prediction generation
    X_train, angles_train, y_train, X_test, angles_test, test_ids = (
        load_and_process_data(load_cached_data=True)
    )

    # Initialize container for Out-Of-Fold predictions
    # This allows us to have "unseen" predictions for every sample in the dataset
    oof_preds = np.zeros(len(y_train))

    # 3. Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=library.config.NUM_FOLDS, shuffle=True, random_state=SEED
    )
    fold_model_paths = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        logger.info(f"\n--- Processing Fold {fold_idx} ---")

        # Train the model for this fold
        # run_fold saves the best model to ./checkpoints/model_fold_{fold_idx}.pth
        run_fold(fold_idx, load_cached_data=True)

        # Path to the saved checkpoint
        model_path = f"./checkpoints/model_fold_{fold_idx}.pth"
        fold_model_paths.append(model_path)

        # Load the model for inference
        model = IDPH_CNN().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        # Create a DataLoader for the validation set of this fold
        val_dataset = IcebergDataset(
            X_train[val_idx], angles_train[val_idx], y_train[val_idx], transform=None
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        # Generate predictions
        fold_preds = []
        with torch.no_grad():
            for images, angs, _ in val_loader:
                images = images.to(DEVICE)
                angs = angs.to(DEVICE)

                # Forward pass
                outputs = model(images, angs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_preds.extend(probs)

        # Store predictions in the OOF array
        oof_preds[val_idx] = np.array(fold_preds)

    # 4. Validation on Metadata Hold-out Set
    logger.info("\nLoading metadata validation set for final evaluation...")
    val_meta_path = "./metadata/val.csv"
    if not os.path.exists(val_meta_path):
        logger.error(f"Metadata file {val_meta_path} not found.")
        return

    val_meta_df = pd.read_csv(val_meta_path)

    # Map metadata entries to original indices
    # The 'original_index' column allows us to retrieve the correct OOF prediction
    val_indices = val_meta_df["original_index"].values

    holdout_preds = oof_preds[val_indices]
    holdout_targets = y_train[val_indices]
    holdout_angles = angles_train[val_indices]

    # Calculate Metric
    final_metric = log_loss(holdout_targets, holdout_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("Performing failure analysis...")

    # Calculate absolute error
    errors = np.abs(holdout_targets - holdout_preds)

    # Calculate correlation between error and incidence angle
    # We use the angles from the training data corresponding to the holdout set
    corr = np.corrcoef(errors, holdout_angles)[0, 1]
    print(f"Correlation between Error and Incidence Angle: {corr}")

    # 6. Submission Generation
    THRESHOLD = 0.1806015565870406

    if final_metric < THRESHOLD:
        logger.info(
            f"\nMetric {final_metric} meets threshold {THRESHOLD}. Generating submission..."
        )

        # Get Test Loader
        test_loader, test_ids_loader = get_test_loader(
            batch_size=BATCH_SIZE, load_cached=True
        )
        test_preds_accum = np.zeros(len(test_ids_loader))

        # Ensemble Inference
        for i, model_path in enumerate(fold_model_paths):
            logger.info(f"Inferencing with model from Fold {i}...")

            model = IDPH_CNN().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()

            fold_test_preds = []
            with torch.no_grad():
                for images, angs in test_loader:
                    images = images.to(DEVICE)
                    angs = angs.to(DEVICE)

                    outputs = model(images, angs)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    fold_test_preds.extend(probs)

            test_preds_accum += np.array(fold_test_preds)

        # Average predictions
        avg_test_preds = test_preds_accum / library.config.NUM_FOLDS

        # Save Submission
        os.makedirs("./submission", exist_ok=True)
        sub_df = pd.DataFrame({"id": test_ids_loader, "is_iceberg": avg_test_preds})
        sub_df.to_csv("./submission/submission.csv", index=False)
        logger.info("Submission saved to ./submission/submission.csv")

    else:
        logger.info(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
