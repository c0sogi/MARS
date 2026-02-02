import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_loader import make_dataloaders
from library.model import RDPWBN
from library.engine import train_fold, validate, predict


def run():
    # 1. Setup Environment
    seed_everything(Config.SEED)
    logger = get_logger()

    # Ensure we use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.DEVICE = str(device)  # Update config just in case

    # 2. Configure for Fast Baseline
    Config.NUM_EPOCHS = 35

    logger.info(f"Running on device: {device}")
    logger.info(f"Max Epochs set to: {Config.NUM_EPOCHS}")

    # 3. Data Preparation for 5-Fold CV
    # Load metadata
    df_train_orig = pd.read_csv(Config.TRAIN_META)
    df_val_orig = pd.read_csv(Config.VAL_META)

    # Combine to use full dataset (Cite Lesson 00165)
    full_df = pd.concat([df_train_orig, df_val_orig], ignore_index=True)

    # 4. Cross-Validation Loop
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)

    oof_preds = np.zeros(len(full_df))
    test_preds_accum = []

    # Store OOF data for failure analysis
    oof_data = {"error": [], "inc_angle": [], "img_mean": [], "img_std": []}

    logger.info("Starting 5-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_df, full_df["is_iceberg"])
    ):
        logger.info(f"--- Fold {fold} ---")

        # Split Data
        train_df_fold = full_df.iloc[train_idx]
        val_df_fold = full_df.iloc[val_idx]

        # Create DataLoaders
        train_loader, val_loader, test_loader = make_dataloaders(
            load_cached_data=True, train_df=train_df_fold, val_df=val_df_fold
        )

        # Initialize Model (RDPWBN)
        model = RDPWBN()
        model.to(device)

        # Train
        model = train_fold(
            fold_idx=fold,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        # Validation (Generate OOF Preds)
        model.eval()
        fold_probs = []
        fold_ids = []

        # For Failure Analysis
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                angles = batch["inc_angle"].to(device)
                labels = batch["label"].to(device)
                ids = batch["id"]  # Not used for indexing here, relying on order

                outputs = model(images, angles)
                probs = torch.sigmoid(outputs)

                # Store preds
                fold_probs.extend(probs.cpu().numpy().flatten())

                # Analysis Data
                batch_errors = torch.abs(probs - labels).cpu().numpy().flatten()
                oof_data["error"].extend(batch_errors)
                oof_data["inc_angle"].extend(angles.cpu().numpy().flatten())

                b_mean = torch.mean(images, dim=(1, 2, 3)).cpu().numpy().flatten()
                b_std = torch.std(images, dim=(1, 2, 3)).cpu().numpy().flatten()
                oof_data["img_mean"].extend(b_mean)
                oof_data["img_std"].extend(b_std)

        # Store OOF predictions in the global array
        # Note: val_loader preserves order of val_df_fold
        oof_preds[val_idx] = fold_probs

        # Test Predictions (Accumulate)
        _, t_probs = predict(model, test_loader, device)
        test_preds_accum.append(t_probs)

    # 5. Calculate Final Metrics
    final_log_loss = log_loss(full_df["is_iceberg"], oof_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_log_loss}")

    # 6. Failure Analysis (Aggregated OOF)
    logger.info("Performing failure analysis on OOF data...")
    df_analysis = pd.DataFrame(oof_data)

    corr_angle = df_analysis["error"].corr(df_analysis["inc_angle"])
    corr_mean = df_analysis["error"].corr(df_analysis["img_mean"])
    corr_std = df_analysis["error"].corr(df_analysis["img_std"])

    print(f"Correlation (Error vs Inc Angle): {corr_angle}")
    print(f"Correlation (Error vs Image Mean): {corr_mean}")
    print(f"Correlation (Error vs Image Std): {corr_std}")

    # 7. Submission Generation
    THRESHOLD = 0.14772333549413377

    if final_log_loss < THRESHOLD:
        logger.info(
            f"Validation metric ({final_log_loss}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Average predictions across folds
        avg_test_probs = np.mean(test_preds_accum, axis=0)

        # Get IDs from test loader (same for all folds)
        _, _, test_loader_ref = make_dataloaders(load_cached_data=True)
        # We need to extract IDs manually or run predict one last time just for IDs
        # predict() returns (ids, probs)
        test_ids, _ = predict(model, test_loader_ref, device)

        # Create submission DataFrame
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_probs})

        # Save to file
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Validation metric ({final_log_loss}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
