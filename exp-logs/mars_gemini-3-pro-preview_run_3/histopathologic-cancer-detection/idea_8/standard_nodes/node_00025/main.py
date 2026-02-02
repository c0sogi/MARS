import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import cv2

# Import from library
from library.config import Config
from library.utils import seed_everything, setup_logger, calculate_auc
from library.dataset import PathologyDataset, get_transforms, load_metadata
from library.models import get_model
from library.engine import train_one_epoch, evaluate, predict_tta

# --- Configuration Overrides for Fast Baseline ---
# Overriding configuration to ensure execution completes within the time limit.
# Increased batch size to 512 (Cite solution_lesson_node_00010)
# Increased epochs to 5 for better convergence (Cite solution_lesson_node_00014)
Config.EPOCHS = 5
Config.BATCH_SIZE = 512
Config.NUM_FOLDS = 5
Config.TTA_STEPS = 4


def main():
    logger = setup_logger()
    seed_everything(Config.SEED)

    logger.info("Starting End-to-End Pipeline (Idea 8)")
    logger.info(
        f"Configuration: Epochs={Config.EPOCHS}, Batch={Config.BATCH_SIZE}, Folds={Config.NUM_FOLDS}"
    )

    # 1. Load Data
    logger.info("Loading Metadata...")
    # Load all metadata using the library function (handles caching)
    df_train_full = load_metadata("train")
    df_val_holdout = load_metadata("val")
    df_test = load_metadata("test")

    # 2. Prepare Folds
    # Create stratified folds for the stacking ensemble
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    df_train_full["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train_full, df_train_full["label"])
    ):
        df_train_full.loc[val_idx, "fold"] = fold

    device = torch.device(Config.DEVICE)

    # 3. Training Loop (Models x Folds)
    # Focusing on single strong architecture (Cite solution_lesson_node_00008)
    for model_name in Config.MODEL_ARCHS:
        logger.info(f"=== Processing Architecture: {model_name} ===")

        for fold in range(Config.NUM_FOLDS):
            logger.info(f"  Training Fold {fold}/{Config.NUM_FOLDS - 1}")

            # Data Splits
            train_df = df_train_full[df_train_full["fold"] != fold].reset_index(
                drop=True
            )
            val_df = df_train_full[df_train_full["fold"] == fold].reset_index(drop=True)

            # Datasets & Loaders
            train_ds = PathologyDataset(train_df, transforms=get_transforms("train"))
            val_ds = PathologyDataset(val_df, transforms=get_transforms("val"))

            train_loader = DataLoader(
                train_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,  # Safety for BN stability
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

            # Model Setup
            model = get_model(model_name, pretrained=True)
            model.to(device)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            criterion = nn.BCEWithLogitsLoss()

            # Scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
            )

            # Training
            best_auc = 0.0
            best_model_path = os.path.join(
                Config.WORKING_DIR, f"{model_name}_fold_{fold}.pth"
            )

            for epoch in range(Config.EPOCHS):
                avg_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, device
                )
                val_loss, val_auc = evaluate(model, val_loader, criterion, device)
                scheduler.step()

                logger.info(
                    f"    Epoch {epoch+1}/{Config.EPOCHS} - Loss: {avg_loss:.4f} - Val AUC: {val_auc:.4f}"
                )

                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(model.state_dict(), best_model_path)

            # Cleanup to free GPU memory
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()

    # 5. Validation on Hold-out Set
    logger.info("=== Validating on Hold-Out Set ===")

    # We need to generate predictions for the hold-out set
    # Strategy: Average predictions from 5 folds (Bagging) (Cite solution_lesson_node_00013)

    val_holdout_ds = PathologyDataset(df_val_holdout, transforms=get_transforms("val"))
    val_holdout_loader = DataLoader(
        val_holdout_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Since we only have one architecture now
    model_name = Config.MODEL_ARCHS[0]
    logger.info(f"  Generating hold-out predictions for {model_name}...")
    fold_preds = []

    for fold in range(Config.NUM_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"{model_name}_fold_{fold}.pth")
        model = get_model(model_name, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        preds = predict_tta(
            model, val_holdout_loader, device, tta_steps=Config.TTA_STEPS
        )
        fold_preds.append(preds)

        del model
        torch.cuda.empty_cache()

    # Average across folds
    final_val_probs = np.mean(fold_preds, axis=0)

    # Calculate Metric
    final_val_auc = calculate_auc(df_val_holdout["label"].values, final_val_probs)
    print(f"Final Validation Metric: {final_val_auc:.16f}")

    # 6. Failure Analysis
    logger.info("=== Performing Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(df_val_holdout["label"].values - final_val_probs)

    # We correlate error with input features (Brightness, Contrast, Red Channel)
    # Using a subset of validation images to ensure speed
    n_analysis = min(2000, len(df_val_holdout))
    # Use random indices for analysis
    analysis_indices = np.random.choice(len(df_val_holdout), n_analysis, replace=False)

    analysis_errors = errors[analysis_indices]
    analysis_paths = df_val_holdout.iloc[analysis_indices]["file_path"].values

    # Feature accumulators
    brightness_list = []
    contrast_list = []
    red_list = []

    for path in analysis_paths:
        try:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                brightness_list.append(np.mean(gray))
                contrast_list.append(np.std(gray))
                red_list.append(np.mean(img[:, :, 0]))
            else:
                brightness_list.append(0)
                contrast_list.append(0)
                red_list.append(0)
        except:
            brightness_list.append(0)
            contrast_list.append(0)
            red_list.append(0)

    # Correlations
    corr_brightness, _ = pearsonr(analysis_errors, brightness_list)
    corr_contrast, _ = pearsonr(analysis_errors, contrast_list)
    corr_red, _ = pearsonr(analysis_errors, red_list)

    print(f"Correlation (Error vs Brightness): {corr_brightness:.4f}")
    print(f"Correlation (Error vs Contrast): {corr_contrast:.4f}")
    print(f"Correlation (Error vs Red Channel): {corr_red:.4f}")

    # 7. Submission
    threshold = 0.9946321378935362
    if final_val_auc > threshold:
        logger.info("Metric threshold met. Generating submission...")

        test_ds = PathologyDataset(df_test, transforms=get_transforms("test"))
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        model_name = Config.MODEL_ARCHS[0]
        logger.info(f"  Generating test predictions for {model_name}...")
        fold_preds = []

        for fold in range(Config.NUM_FOLDS):
            model_path = os.path.join(
                Config.WORKING_DIR, f"{model_name}_fold_{fold}.pth"
            )
            model = get_model(model_name, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

            preds = predict_tta(model, test_loader, device, tta_steps=Config.TTA_STEPS)
            fold_preds.append(preds)

            del model
            torch.cuda.empty_cache()

        final_test_probs = np.mean(fold_preds, axis=0)

        # Save submission
        submission = pd.DataFrame({"id": df_test["id"], "label": final_test_probs})
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Saved final submission to {Config.SUBMISSION_PATH}")
    else:
        logger.info(
            f"Metric {final_val_auc} did not meet threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
