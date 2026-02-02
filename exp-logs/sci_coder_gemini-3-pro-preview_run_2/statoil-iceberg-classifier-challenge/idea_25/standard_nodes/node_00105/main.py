import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

from library.config import Config, set_seed
from library.utils import get_logger
from library.data_loader import load_data, get_dataloaders, get_test_loader
from library.train import run_fold


def main():
    # 1. Setup
    set_seed(Config.SEED)
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    logger = get_logger(os.path.join(Config.WORKING_DIR, "run.log"))
    logger.info("Starting pipeline execution...")

    # 2. Load Data
    # load_cached_data=True as requested
    train_data, test_data = load_data(debug=Config.DEBUG, load_cached_data=True)

    # 3. Cross-Validation Loop
    models = []

    # Lists to store Out-Of-Fold (OOF) predictions and metadata for global evaluation
    global_preds = []
    global_targets = []
    global_inc_angles = []

    device = torch.device(Config.DEVICE)

    for fold in range(Config.NUM_FOLDS):
        logger.info(f"--- Processing Fold {fold} ---")

        # Train the model for this fold
        model, best_loss = run_fold(fold, train_data, logger)
        models.append(model)

        # Generate predictions on the validation set for this fold
        # We need to reconstruct the loader to get the validation data
        _, val_loader = get_dataloaders(fold, train_data)

        model.eval()
        fold_preds = []
        fold_targets = []
        fold_inc = []

        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                # Forward pass
                outputs = model(images, angles)

                # Collect results
                preds = outputs.cpu().numpy().flatten()
                fold_preds.extend(preds)
                fold_targets.extend(labels.numpy().flatten())
                fold_inc.extend(angles.cpu().numpy().flatten())

        global_preds.extend(fold_preds)
        global_targets.extend(fold_targets)
        global_inc_angles.extend(fold_inc)

    # 4. Global Validation Metric
    y_true = np.array(global_targets)
    y_pred = np.array(global_preds)

    # Clip predictions to prevent log(0) errors, standard practice for log loss
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

    final_metric = log_loss(y_true, y_pred_clipped)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("Running failure analysis...")
    errors = np.abs(y_true - y_pred)
    inc_angles = np.array(global_inc_angles)

    # Correlation between error magnitude and incidence angle
    # Note: inc_angles might have imputed values, which is what the model saw.
    corr_inc = np.corrcoef(errors, inc_angles)[0, 1]

    print(f"Correlation between Error and Incidence Angle: {corr_inc}")

    # 6. Submission Generation
    THRESHOLD = 0.16676861786296204

    if final_metric < THRESHOLD:
        logger.info(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")

        # Get Test Loader
        test_loader, test_ids = get_test_loader(train_data, test_data)

        # Ensemble Prediction
        # Initialize accumulator
        avg_preds = np.zeros(len(test_ids))

        for i, model in enumerate(models):
            logger.info(f"Predicting with model from fold {i}...")
            model.eval()
            fold_test_preds = []

            with torch.no_grad():
                for images, angles in test_loader:
                    images = images.to(device)
                    angles = angles.to(device)

                    outputs = model(images, angles)
                    fold_test_preds.extend(outputs.cpu().numpy().flatten())

            avg_preds += np.array(fold_test_preds)

        # Average
        avg_preds /= len(models)

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.warning(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
