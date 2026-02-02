import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler

from library.config import Config
from library.utils import get_logger, seed_everything, calculate_metric
from library.data import get_dataframes, get_dataloader
from library.modeling import get_model
from library.engine import train_one_epoch, validate, predict
from library.calibration import Calibrator

logger = get_logger("pipeline")


def train_model_fold(model_name, fold, train_df, val_df, device):
    """
    Trains a single model on a specific fold using Progressive Resizing.

    Args:
        model_name (str): Name of the model architecture.
        fold (int): Current fold number.
        train_df (pd.DataFrame): Training data for this fold.
        val_df (pd.DataFrame): Validation data for this fold.
        device (torch.device): Compute device.

    Returns:
        tuple: (oof_preds, oof_targets, best_model_state_dict)
    """
    model_cfg = Config.MODELS[model_name]

    # Initialize Model
    model = get_model(model_name, pretrained=True)
    model.to(device)

    best_model_path = os.path.join(
        Config.WORKING_DIR, f"{model_name}_fold{fold}_best.pth"
    )
    criterion = nn.BCEWithLogitsLoss()

    if os.path.exists(best_model_path):
        logger.info(
            f"Found existing checkpoint at {best_model_path}. Skipping training."
        )
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        # Optimizer & Scaler
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=model_cfg["lr"], weight_decay=Config.WEIGHT_DECAY
        )
        scaler = GradScaler()

        # Scheduler setup
        total_epochs = sum(p["epochs"] for p in model_cfg["phases"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=model_cfg["min_lr"]
        )

        best_val_loss = float("inf")

        # Progressive Resizing Loop
        current_epoch = 0
        for phase_idx, phase in enumerate(model_cfg["phases"]):
            img_size = phase["img_size"]
            phase_epochs = phase["epochs"]
            logger.info(
                f"Model {model_name} | Fold {fold} | Phase {phase_idx+1}: "
                f"Img Size {img_size}x{img_size} | Epochs {phase_epochs}"
            )

            # Re-initialize DataLoaders for new image size
            train_loader = get_dataloader(
                train_df, img_size, model_cfg["batch_size"], mode="train"
            )
            val_loader = get_dataloader(
                val_df, img_size, model_cfg["batch_size"], mode="val"
            )

            for epoch in range(phase_epochs):
                current_epoch += 1

                train_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, device, scaler
                )
                val_loss, _, _ = validate(model, val_loader, criterion, device)

                scheduler.step()

                logger.info(
                    f"  Epoch {current_epoch}/{total_epochs} | "
                    f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_model_path)

        # Load Best Model
        logger.info(
            f"Loading best model from {best_model_path} with Val Loss {best_val_loss:.6f}"
        )
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Final Validation on Target Resolution (Last Phase)
    final_img_size = model_cfg["phases"][-1]["img_size"]
    val_loader = get_dataloader(
        val_df, final_img_size, model_cfg["batch_size"], mode="val"
    )
    _, oof_preds, oof_targets = validate(model, val_loader, criterion, device)

    return oof_preds, oof_targets, model


def generate_calibrated_submission(oof_data, test_predictions, test_ids):
    """
    Generates submission file using Quality Gating, Calibration, and Ensembling.

    Args:
        oof_data (dict): Dictionary containing OOF predictions and targets per model.
        test_predictions (dict): Dictionary containing list of test predictions per model.
        test_ids (np.array): Array of test image IDs.
    """
    logger.info("\nStarting Calibration and Ensemble Aggregation...")

    final_test_preds = []
    valid_models_count = 0

    for model_name in Config.MODELS:
        if model_name not in oof_data:
            continue

        # Concatenate OOF data across folds
        y_pred_oof = np.concatenate(oof_data[model_name]["y_pred"]).flatten()
        y_true_oof = np.concatenate(oof_data[model_name]["y_true"]).flatten()

        # Calculate raw Log Loss
        raw_loss = calculate_metric(y_true_oof, y_pred_oof)
        logger.info(f"Model {model_name} - Raw OOF Log Loss: {raw_loss:.10f}")

        # Quality Gating
        if raw_loss > Config.OOF_THRESHOLD:
            logger.warning(
                f"Model {model_name} discarded (Loss {raw_loss:.6f} > {Config.OOF_THRESHOLD})"
            )
            continue

        # 1. Average Raw Test Predictions across folds (Bagging)
        # test_predictions[model_name] is a list of arrays (one per fold)
        avg_raw_test_probs = np.mean(test_predictions[model_name], axis=0).flatten()

        # 2. Train Calibrator
        if Config.CALIBRATION_METHOD == "IsotonicRegression":
            calibrator = Calibrator(method="isotonic")
            calibrator.fit(y_pred_oof, y_true_oof)

            # 3. Apply Calibration to Aggregated Test Predictions
            calibrated_test_probs = calibrator.transform(avg_raw_test_probs)

            final_test_preds.append(calibrated_test_probs)
            valid_models_count += 1
            logger.info(f"Model {model_name} - Calibrated and included in ensemble.")
        else:
            final_test_preds.append(avg_raw_test_probs)
            valid_models_count += 1
            logger.info(f"Model {model_name} - Included (Raw).")

    # Fallback if all models fail gating
    if valid_models_count == 0:
        logger.error(
            "No models passed quality gating! Using raw average of all models."
        )
        for model_name in Config.MODELS:
            avg_raw_test_probs = np.mean(test_predictions[model_name], axis=0).flatten()
            final_test_preds.append(avg_raw_test_probs)

    # Ensemble Averaging (Arithmetic Mean of Calibrated Probabilities)
    ensemble_preds = np.mean(final_test_preds, axis=0)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids.astype(int), "label": ensemble_preds})

    # Sort by ID
    submission_df = submission_df.sort_values("id")

    # Save
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(sub_path, index=False)
    logger.info(f"Submission saved to {sub_path}")


def run_kfold_ensemble(debug=False):
    """
    Orchestrates the full Stratified K-Fold Heterogeneous Ensemble pipeline.
    """
    # Setup
    Config.setup(debug=debug)
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    train_df_part, val_df_part, test_df = get_dataframes()
    full_train_df = pd.concat([train_df_part, val_df_part]).reset_index(drop=True)

    # Storage
    oof_data = {}
    test_predictions = {}
    test_ids = None

    # K-Fold Splitter
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    for model_name in Config.MODELS:
        logger.info(f"\n{'='*20} Processing Model: {model_name} {'='*20}")

        oof_data[model_name] = {"y_true": [], "y_pred": []}
        test_predictions[model_name] = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(full_train_df, full_train_df["label"])
        ):
            logger.info(f"--- Fold {fold + 1}/{Config.N_FOLDS} ---")

            fold_train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = full_train_df.iloc[val_idx].reset_index(drop=True)

            # Train
            oof_preds, oof_targets, model = train_model_fold(
                model_name, fold, fold_train_df, fold_val_df, device
            )

            # Store OOF
            oof_data[model_name]["y_pred"].append(oof_preds)
            oof_data[model_name]["y_true"].append(oof_targets)

            # Inference on Test Set
            # Use the final resolution for inference
            final_img_size = Config.MODELS[model_name]["phases"][-1]["img_size"]
            test_loader = get_dataloader(
                test_df,
                final_img_size,
                Config.MODELS[model_name]["batch_size"],
                mode="test",
            )

            fold_test_preds, fold_test_ids = predict(
                model, test_loader, device, use_tta=Config.USE_TTA
            )

            test_predictions[model_name].append(fold_test_preds)

            if test_ids is None:
                test_ids = fold_test_ids

            # Cleanup
            del model, oof_preds, oof_targets, fold_test_preds
            torch.cuda.empty_cache()
            gc.collect()

    # Generate Submission
    generate_calibrated_submission(oof_data, test_predictions, test_ids)
