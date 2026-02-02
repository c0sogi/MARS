import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import seed_everything, get_logger, compute_qwk
from library.data import load_data_from_metadata, get_dataloaders
from library.models import RegressionModel, OrdinalModel
from library.engine import train_one_epoch, validate_one_epoch, AWP
from library.stacking import LGBMStacker, FeatureEngineer

# --- Configuration Overrides for Fast Execution ---
Config.EPOCHS = 1
Config.TRAIN_BATCH_SIZE = 4
Config.GRADIENT_ACCUMULATION_STEPS = 4
Config.N_FOLDS = 5
TRAIN_SAMPLE_SIZE = 1000  # Limit training data for speed to meet time limit


def run_inference_for_stacker(models, pred_dict, split_name):
    """
    Helper to run inference using the trained stacker models on a specific split.
    Mimics LGBMStacker.predict but allows specifying the split (e.g., 'val').
    """
    fe = FeatureEngineer()
    # Get meta features
    meta_df = fe.get_features(split_name, load_cached_data=True)

    # Prepare OOF dataframe
    data = {}
    for model_name, preds in pred_dict.items():
        data[f"pred_{model_name}"] = preds.flatten()
    oof_df = pd.DataFrame(data)

    # Reset indices to ensure alignment
    meta_df = meta_df.reset_index(drop=True)
    oof_df = oof_df.reset_index(drop=True)

    # Concat
    X = pd.concat([oof_df, meta_df], axis=1)

    # Predict
    final_preds = np.zeros(len(X))
    for model in models:
        preds = model.predict(X, num_iteration=model.best_iteration)
        final_preds += preds

    final_preds /= len(models)
    return final_preds, X  # Return X for failure analysis


def main():
    logger = get_logger("runfile")
    seed_everything(Config.SEED)

    logger.info("Starting Runfile Execution...")

    # 1. Load Data
    logger.info("Loading Metadata...")
    df_train_full = load_data_from_metadata("train")
    df_val_holdout = load_data_from_metadata("val")
    df_test = load_data_from_metadata("test")

    # Subsample Training Data for Speed
    if len(df_train_full) > TRAIN_SAMPLE_SIZE:
        logger.info(f"Subsampling training data to {TRAIN_SAMPLE_SIZE} samples.")
        df_train = df_train_full.sample(
            n=TRAIN_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
    else:
        df_train = df_train_full

    # 2. Initialize Arrays
    # OOF for Stacker Training (aligned with df_train)
    oof_reg = np.zeros(len(df_train))
    oof_ord = np.zeros(len(df_train))

    # Accumulators for Hold-out Validation (aligned with df_val_holdout)
    val_reg_accum = np.zeros(len(df_val_holdout))
    val_ord_accum = np.zeros(len(df_val_holdout))

    # Accumulators for Test (aligned with df_test)
    test_reg_accum = np.zeros(len(df_test))
    test_ord_accum = np.zeros(len(df_test))

    # 3. Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Pre-compute/Cache Test & Val Dataloaders (Common across folds)
    val_loader = get_dataloaders(
        df_val_holdout, "val_holdout", batch_size=Config.VALID_BATCH_SIZE, shuffle=False
    )
    test_loader = get_dataloaders(
        df_test,
        "test_set",
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        max_length=Config.INFERENCE_MAX_LENGTH,
    )

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(df_train, df_train["score"])
    ):
        logger.info(f"=== Fold {fold + 1}/{Config.N_FOLDS} ===")

        df_fold_train = df_train.iloc[train_idx].reset_index(drop=True)
        df_fold_valid = df_train.iloc[valid_idx].reset_index(drop=True)

        # Get Loaders for this fold
        train_loader = get_dataloaders(
            df_fold_train,
            f"train_fold_{fold}",
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
        )
        valid_loader = get_dataloaders(
            df_fold_valid,
            f"valid_fold_{fold}",
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
        )

        # --- Train Model A: Regression ---
        logger.info(f"Training Regression Model (Fold {fold+1})...")
        model_reg = RegressionModel(pretrained=True).to(Config.DEVICE)
        optimizer = AdamW(
            model_reg.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * Config.EPOCHS,
        )
        scaler = torch.amp.GradScaler("cuda")
        awp = AWP(model_reg, optimizer) if Config.USE_AWP else None

        for epoch in range(Config.EPOCHS):
            train_one_epoch(
                model_reg,
                train_loader,
                optimizer,
                scheduler,
                Config.DEVICE,
                epoch,
                criterion_mse,
                scaler,
                awp,
            )

        # Predict OOF (Fold Valid)
        _, _, preds_fold_reg, _ = validate_one_epoch(
            model_reg, valid_loader, Config.DEVICE, criterion_mse
        )
        oof_reg[valid_idx] = preds_fold_reg

        # Predict Hold-out Val
        _, _, preds_val_reg, _ = validate_one_epoch(
            model_reg, val_loader, Config.DEVICE, criterion_mse
        )
        val_reg_accum += preds_val_reg

        # Predict Test
        _, _, preds_test_reg, _ = validate_one_epoch(
            model_reg, test_loader, Config.DEVICE, criterion_mse
        )
        test_reg_accum += preds_test_reg

        del model_reg, optimizer, scaler, scheduler, awp
        torch.cuda.empty_cache()

        # --- Train Model B: Ordinal ---
        logger.info(f"Training Ordinal Model (Fold {fold+1})...")
        model_ord = OrdinalModel(pretrained=True).to(Config.DEVICE)
        optimizer = AdamW(
            model_ord.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * Config.EPOCHS,
        )
        scaler = torch.amp.GradScaler("cuda")
        awp = AWP(model_ord, optimizer) if Config.USE_AWP else None

        for epoch in range(Config.EPOCHS):
            train_one_epoch(
                model_ord,
                train_loader,
                optimizer,
                scheduler,
                Config.DEVICE,
                epoch,
                criterion_bce,
                scaler,
                awp,
            )

        # Predict OOF
        _, _, preds_fold_ord, _ = validate_one_epoch(
            model_ord, valid_loader, Config.DEVICE, criterion_bce
        )
        oof_ord[valid_idx] = preds_fold_ord

        # Predict Hold-out Val
        _, _, preds_val_ord, _ = validate_one_epoch(
            model_ord, val_loader, Config.DEVICE, criterion_bce
        )
        val_ord_accum += preds_val_ord

        # Predict Test
        _, _, preds_test_ord, _ = validate_one_epoch(
            model_ord, test_loader, Config.DEVICE, criterion_bce
        )
        test_ord_accum += preds_test_ord

        del model_ord, optimizer, scaler, scheduler, awp
        torch.cuda.empty_cache()

    # Average Predictions
    val_preds_reg = val_reg_accum / Config.N_FOLDS
    val_preds_ord = val_ord_accum / Config.N_FOLDS
    test_preds_reg = test_reg_accum / Config.N_FOLDS
    test_preds_ord = test_ord_accum / Config.N_FOLDS

    # 5. Stacking
    logger.info("Training Stacker...")
    stacker = LGBMStacker()

    # Train on Train OOFs
    train_oof_dict = {"reg": oof_reg, "ord": oof_ord}
    stacker.train(train_oof_dict, df_train["score"].values, n_folds=5)

    # 6. Validation Assessment
    logger.info("Validating Stacker on Hold-out Set...")
    val_pred_dict = {"reg": val_preds_reg, "ord": val_preds_ord}

    # Use helper to predict on val
    final_val_preds, val_X = run_inference_for_stacker(
        stacker.models, val_pred_dict, "val"
    )

    # Clip and Round for QWK
    final_val_preds_rounded = np.clip(np.round(final_val_preds), 1, 6).astype(int)
    val_targets = df_val_holdout["score"].values.astype(int)

    val_qwk = compute_qwk(val_targets, final_val_preds_rounded)
    print(f"Final Validation Metric: {val_qwk}")

    # 7. Failure Analysis
    logger.info("Performing Failure Analysis...")
    errors = np.abs(val_targets - final_val_preds)

    # Correlate errors with features in val_X
    # val_X contains ['pred_reg', 'pred_ord', 'char_count', 'word_count', ...]
    analysis_cols = [c for c in val_X.columns if c not in ["pred_reg", "pred_ord"]]

    print("\nError Correlations with Meta-Features:")
    for col in analysis_cols:
        if col in val_X.columns:
            corr, _ = pearsonr(errors, val_X[col])
            print(f"Correlation (Error vs {col}): {corr:.4f}")

    # 8. Submission
    THRESHOLD = 0.8246384329994252
    if val_qwk > THRESHOLD:
        logger.info(
            f"Validation score {val_qwk} > {THRESHOLD}. Generating submission..."
        )
        test_pred_dict = {"reg": test_preds_reg, "ord": test_preds_ord}
        stacker.run_inference_and_submit(test_pred_dict)
    else:
        logger.info(f"Validation score {val_qwk} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
