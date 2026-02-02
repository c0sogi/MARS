import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import process_and_cache_data, get_kfold_loaders
from library.train_eval import train_fold, generate_submission
from library.model import MLCWNet


def main():
    # 1. Setup Environment
    set_seed(Config.SEED)
    logger = setup_logger("Runfile", os.path.join(Config.WORKING_DIR, "run.log"))
    device = torch.device(Config.DEVICE)

    logger.info("Initializing Fast Baseline Execution for MLCW-Net")

    # 2. Data Preparation
    # Load processed data arrays to handle ID mapping for OOF
    # load_cached_data=True ensures we use pre-computed arrays if available
    X_train, y_train, angles_train, ids_train, _, _, _ = process_and_cache_data(
        load_cached_data=True
    )

    # 3. Stratified K-Fold Training & OOF Collection
    # We reconstruct the split indices to map validation loaders back to image IDs
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get the pre-configured loaders (which use the same seed/split logic)
    fold_loaders = get_kfold_loaders(load_cached_data=True)

    # Dictionaries to store Out-Of-Fold predictions and targets
    oof_preds = {}  # id -> probability
    oof_targets = {}  # id -> true label

    # Training Loop
    # We limit epochs to 100 to allow for "Low and Slow" convergence (Cite solution_lesson_node_00023)
    MAX_EPOCHS = 100

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        logger.info(
            f"\n{'='*10} Processing Fold {fold_idx + 1}/{Config.NUM_FOLDS} {'='*10}"
        )

        train_loader, val_loader = fold_loaders[fold_idx]

        # Train the model for this fold
        train_fold(fold_idx, train_loader, val_loader, logger, num_epochs=MAX_EPOCHS)

        # --- Inference on Validation Fold (for OOF) ---
        # Load the best model saved during training
        model_path = os.path.join(Config.WORKING_DIR, f"mlcw_net_fold_{fold_idx}.pth")
        if not os.path.exists(model_path):
            logger.error(
                f"Model file {model_path} not found. Skipping OOF inference for this fold."
            )
            continue

        model = MLCWNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_probs = []

        # Disable gradient calculation for inference speed
        with torch.no_grad():
            for images, angles, targets in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                outputs = model(images, angles)
                probs = torch.sigmoid(outputs)
                fold_probs.extend(probs.cpu().numpy().flatten())

        # Map predictions back to IDs
        # The val_loader iterates sequentially over the validation subset
        val_ids = ids_train[val_idx]
        val_y = y_train[val_idx]

        if len(fold_probs) != len(val_ids):
            logger.error(
                f"Mismatch in prediction count for Fold {fold_idx+1}: {len(fold_probs)} vs {len(val_ids)}"
            )
            continue

        for i, uid in enumerate(val_ids):
            oof_preds[uid] = fold_probs[i]
            oof_targets[uid] = val_y[i]

    # 4. Evaluation on Metadata Hold-out Set
    # Load the specific validation set defined in metadata
    val_meta_path = Config.VAL_META_CSV
    if not os.path.exists(val_meta_path):
        logger.error(f"Metadata file {val_meta_path} not found.")
        return

    val_meta_df = pd.read_csv(val_meta_path)
    target_ids = val_meta_df["id"].values

    y_true_eval = []
    y_pred_eval = []
    angles_eval = []  # For failure analysis

    # Lookup for angles for failure analysis
    id_to_angle = dict(zip(ids_train, angles_train))

    missing_ids = 0
    for uid in target_ids:
        if uid in oof_preds:
            y_true_eval.append(oof_targets[uid])
            y_pred_eval.append(oof_preds[uid])
            angles_eval.append(id_to_angle.get(uid, 0))
        else:
            missing_ids += 1

    if missing_ids > 0:
        logger.warning(
            f"{missing_ids} validation IDs were not found in OOF predictions."
        )

    # Compute Metric
    final_metric = log_loss(y_true_eval, y_pred_eval)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("\nFailure Analysis:")
    y_true_arr = np.array(y_true_eval)
    y_pred_arr = np.array(y_pred_eval)
    angles_arr = np.array(angles_eval)

    errors = np.abs(y_true_arr - y_pred_arr)

    # Correlation: Error vs Incidence Angle
    if len(errors) > 1:
        corr_angle = np.corrcoef(errors, angles_arr)[0, 1]
        print(f"Correlation (Error vs Inc_Angle): {corr_angle:.16f}")

        # Correlation: Error vs Target (Class Bias)
        corr_target = np.corrcoef(errors, y_true_arr)[0, 1]
        print(f"Correlation (Error vs Target): {corr_target:.16f}")

    # 6. Submission Generation
    THRESHOLD = 0.17493283735739185

    if final_metric < THRESHOLD:
        logger.info(
            f"Validation metric {final_metric:.6f} passed threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission()
    else:
        logger.warning(
            f"Validation metric {final_metric:.6f} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
