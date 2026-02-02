import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.utils import seed_everything, get_logger, calculate_log_loss
from library.dataset import get_dataset, DogCatDataset
from library.models import create_model
from library.augmentations import (
    get_train_transforms,
    get_valid_transforms,
    MixupCutmixCollator,
)
from library.stacking import StackingTrainer, predict_stacking

# Override Config for Fast Baseline Execution
Config.EPOCHS = 2
Config.N_FOLDS = 5
Config.BATCH_SIZE = 64

logger = get_logger("runfile")


def predict_with_tta(model, loader, device):
    """
    Performs inference with Test-Time Augmentation (Horizontal Flip).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Original Prediction
            out1 = model(images)
            prob1 = torch.sigmoid(out1).view(-1)

            # 2. Flipped Prediction (TTA)
            images_flipped = torch.flip(
                images, [3]
            )  # Flip width dimension (B, C, H, W)
            out2 = model(images_flipped)
            prob2 = torch.sigmoid(out2).view(-1)

            # Average probabilities
            avg_prob = (prob1 + prob2) / 2.0
            preds.append(avg_prob.cpu().numpy())

    return np.concatenate(preds)


def get_image_metadata(filepaths, input_dir):
    """
    Extracts width, height, and aspect ratio for failure analysis.
    """
    widths = []
    heights = []
    ratios = []

    for fp in filepaths:
        full_path = os.path.join(input_dir, fp)
        img = cv2.imread(full_path)
        if img is not None:
            h, w, _ = img.shape
            widths.append(w)
            heights.append(h)
            ratios.append(w / h)
        else:
            widths.append(0)
            heights.append(0)
            ratios.append(0)

    return np.array(widths), np.array(heights), np.array(ratios)


def calc_correlation(x, y):
    """
    Calculates Pearson correlation coefficient using NumPy.
    """
    if len(x) != len(y):
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    if den == 0:
        return 0.0
    return num / den


def run():
    seed_everything(Config.SEED)
    Config.setup_directories()
    logger.info("Starting Fast Baseline Run...")

    device = Config.DEVICE

    # Load Metadata
    train_df_full = pd.read_csv(Config.TRAIN_CSV)
    val_df_holdout = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Prepare K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Prepare DataLoaders for Holdout Val and Test (Fixed across folds)
    val_ds = DogCatDataset(val_df_holdout, transforms=get_valid_transforms())
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    test_ds = DogCatDataset(test_df, transforms=get_valid_transforms())
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Accumulator for holdout predictions: {model_name: accumulated_probs}
    holdout_preds_accumulator = {
        m: np.zeros(len(val_df_holdout)) for m in Config.MODEL_NAMES
    }

    # -------------------------------------------------------------------------
    # 1. Base Model Training & OOF Generation
    # -------------------------------------------------------------------------
    for model_name in Config.MODEL_NAMES:
        logger.info(f"Processing Model: {model_name}")

        oof_preds_full = np.zeros(len(train_df_full))

        for fold, (train_idx, valid_idx) in enumerate(
            skf.split(train_df_full, train_df_full["label"])
        ):
            logger.info(f"  Fold {fold+1}/{Config.N_FOLDS}")

            # Split Data
            train_fold = train_df_full.iloc[train_idx].reset_index(drop=True)
            valid_fold = train_df_full.iloc[valid_idx].reset_index(drop=True)

            # Datasets
            train_ds = DogCatDataset(train_fold, transforms=get_train_transforms())
            valid_ds_oof = DogCatDataset(valid_fold, transforms=get_valid_transforms())

            # Loaders
            train_loader = DataLoader(
                train_ds,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                collate_fn=MixupCutmixCollator(),
            )
            valid_loader_oof = DataLoader(
                valid_ds_oof,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )

            # Model, Optimizer, Scheduler
            model = create_model(model_name)
            model.to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
            )
            criterion = torch.nn.BCEWithLogitsLoss()

            # Training Loop
            for epoch in range(1, Config.EPOCHS + 1):
                model.train()
                for images, targets in train_loader:
                    images, targets = images.to(device), targets.to(device).float()
                    optimizer.zero_grad()
                    outputs = model(images).view(-1)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            # Predict OOF (with TTA)
            oof_p = predict_with_tta(model, valid_loader_oof, device)
            oof_preds_full[valid_idx] = oof_p

            # Save OOF for this fold
            df_oof_fold = valid_fold.copy()
            df_oof_fold["pred"] = oof_p
            df_oof_fold.to_csv(
                os.path.join(Config.OOF_DIR, f"{model_name}_fold_{fold}_oof.csv"),
                index=False,
            )

            # Predict Holdout Val (with TTA) - Accumulate average
            val_p = predict_with_tta(model, val_loader, device)
            holdout_preds_accumulator[model_name] += val_p / Config.N_FOLDS

            # Predict Test (with TTA) - Save per fold for stacking aggregation
            test_p = predict_with_tta(model, test_loader, device)
            df_test_fold = test_df.copy()
            df_test_fold["pred"] = test_p
            df_test_fold.to_csv(
                os.path.join(Config.OOF_DIR, f"{model_name}_fold_{fold}_test.csv"),
                index=False,
            )

            # Clean up
            del model, optimizer, scheduler, train_loader, valid_loader_oof
            torch.cuda.empty_cache()

        # Save Consolidated OOF for StackingTrainer
        df_oof_all = train_df_full.copy()
        df_oof_all["pred"] = oof_preds_full
        df_oof_all.to_csv(
            os.path.join(Config.OOF_DIR, f"{model_name}_oof.csv"), index=False
        )

    # -------------------------------------------------------------------------
    # 2. Stacking & Validation
    # -------------------------------------------------------------------------
    logger.info("Training Meta-Learner...")
    stacker = StackingTrainer()
    # Force recompute to use the newly generated OOF files
    meta_model = stacker.train(load_cached_data=False)

    # Prepare Holdout Validation Meta-Features
    val_meta_df = val_df_holdout.copy()
    for model_name in Config.MODEL_NAMES:
        val_meta_df[model_name] = holdout_preds_accumulator[model_name]

    X_val = val_meta_df[Config.MODEL_NAMES].values
    y_val = val_meta_df["label"].values

    # Meta-Model Inference on Holdout
    val_final_probs = meta_model.predict_proba(X_val)[:, 1]

    # Calculate Metric
    final_metric = calculate_log_loss(y_val, val_final_probs)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("Performing Failure Analysis...")

    # Calculate per-sample log loss
    epsilon = 1e-15
    val_final_probs_clipped = np.clip(val_final_probs, epsilon, 1 - epsilon)
    loss_per_sample = -(
        y_val * np.log(val_final_probs_clipped)
        + (1 - y_val) * np.log(1 - val_final_probs_clipped)
    )

    # Get Image Stats
    widths, heights, ratios = get_image_metadata(
        val_df_holdout["filepath"].values, Config.INPUT_DIR
    )

    # Correlations
    corr_w = calc_correlation(loss_per_sample, widths)
    corr_h = calc_correlation(loss_per_sample, heights)
    corr_r = calc_correlation(loss_per_sample, ratios)

    print(f"Error Correlation - Width: {corr_w:.4f}")
    print(f"Error Correlation - Height: {corr_h:.4f}")
    print(f"Error Correlation - Aspect Ratio: {corr_r:.4f}")

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.018199009307556684

    if final_metric < THRESHOLD:
        logger.info("Metric below threshold. Generating Submission...")
        predict_stacking(load_cached_data=False)
    else:
        logger.info(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    run()
