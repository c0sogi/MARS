import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import get_logger, calculate_metric, seed_everything
from library.data import get_dataloader, get_dataframes

# Initialize logger
logger = get_logger("modeling")


def get_model(model_name, pretrained=True):
    """
    Creates a model instance based on the configuration.
    Wraps timm to provide a binary classification head.
    """
    if model_name not in Config.MODELS:
        raise ValueError(f"Model {model_name} not found in Config.MODELS")

    model_config = Config.MODELS[model_name]
    backbone_name = model_config["backbone"]

    # Create model using timm
    # num_classes=1 ensures the final layer projects to a single logit
    # in_chans=3 for RGB images
    model = timm.create_model(
        backbone_name, pretrained=pretrained, num_classes=1, in_chans=3
    )

    return model


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """
    Trains the model for one epoch using Mixed Precision.
    """
    model.train()
    total_loss = 0.0
    dataset_size = len(loader.dataset)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # [B, 1]

        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / dataset_size
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Validates the model. Returns average loss and predictions.
    """
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    avg_loss = total_loss / dataset_size
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    return avg_loss, preds, targets


def inference(model, loader, device, use_tta=False):
    """
    Runs inference on the test set. Supports TTA (Horizontal Flip).
    """
    model.eval()
    preds = []
    ids = []

    with torch.no_grad():
        for images, img_ids in loader:
            images = images.to(device)

            # Forward pass 1 (Original)
            logits = model(images)
            probs = torch.sigmoid(logits)

            if use_tta:
                # Forward pass 2 (Horizontal Flip)
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
                probs_flip = torch.sigmoid(logits_flip)

                # Average probabilities
                probs = (probs + probs_flip) / 2.0

            preds.append(probs.cpu().numpy())
            ids.extend(img_ids.numpy())

    return np.concatenate(preds), np.array(ids)


def run_training_and_inference():
    """
    Executes the full Stratified K-Fold Heterogeneous Ensemble pipeline:
    1. Data Loading & Splitting
    2. Progressive Resizing Training per Model per Fold
    3. OOF Generation & Calibration (Isotonic Regression)
    4. Test Inference & Submission Generation
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    train_df_part, val_df_part, test_df = get_dataframes()
    # Combine for proper K-Fold
    full_train_df = pd.concat([train_df_part, val_df_part]).reset_index(drop=True)

    # Prepare storage for OOF and Test predictions
    # Structure: oof_preds[model_key] = { 'y_true': [], 'y_pred': [] }
    # Structure: test_preds[model_key] = list of arrays (one per fold)
    oof_data = {}
    test_predictions = {}

    # K-Fold Splitter
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Iterate over each model architecture defined in Config
    for model_name, model_cfg in Config.MODELS.items():
        logger.info(f"\n{'='*20} Processing Model: {model_name} {'='*20}")

        oof_data[model_name] = {"y_true": [], "y_pred": []}
        test_predictions[model_name] = []

        # Iterate over Folds
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(full_train_df, full_train_df["label"])
        ):
            logger.info(f"--- Fold {fold + 1}/{Config.N_FOLDS} ---")

            fold_train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = full_train_df.iloc[val_idx].reset_index(drop=True)

            # Initialize Model
            model = get_model(model_name, pretrained=True)
            model.to(device)

            # Optimizer & Scaler
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=model_cfg["lr"], weight_decay=Config.WEIGHT_DECAY
            )
            scaler = GradScaler()
            criterion = nn.BCEWithLogitsLoss()

            # Calculate total epochs for scheduler
            total_epochs = sum(p["epochs"] for p in model_cfg["phases"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=model_cfg["min_lr"]
            )

            best_val_loss = float("inf")
            best_model_path = os.path.join(
                Config.WORKING_DIR, f"{model_name}_fold{fold}_best.pth"
            )

            # Progressive Resizing Loop
            current_epoch = 0
            for phase_idx, phase in enumerate(model_cfg["phases"]):
                img_size = phase["img_size"]
                phase_epochs = phase["epochs"]
                logger.info(
                    f"Phase {phase_idx+1}: Image Size {img_size}x{img_size} for {phase_epochs} epochs"
                )

                # Re-initialize DataLoaders for new image size
                train_loader = get_dataloader(
                    fold_train_df, img_size, model_cfg["batch_size"], mode="train"
                )
                val_loader = get_dataloader(
                    fold_val_df, img_size, model_cfg["batch_size"], mode="val"
                )

                for epoch in range(phase_epochs):
                    current_epoch += 1
                    train_loss = train_one_epoch(
                        model, train_loader, optimizer, criterion, device, scaler
                    )
                    val_loss, val_preds, val_targets = validate(
                        model, val_loader, criterion, device
                    )

                    scheduler.step()

                    logger.info(
                        f"Epoch {current_epoch}/{total_epochs} | "
                        f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), best_model_path)

            # Load Best Model for this Fold
            model.load_state_dict(torch.load(best_model_path, map_location=device))

            # Generate OOF Predictions (at target resolution, which is the last phase size)
            final_img_size = model_cfg["phases"][-1]["img_size"]
            val_loader = get_dataloader(
                fold_val_df, final_img_size, model_cfg["batch_size"], mode="val"
            )
            _, oof_preds, oof_targets = validate(model, val_loader, criterion, device)

            # Store OOF
            oof_data[model_name]["y_pred"].append(oof_preds)
            oof_data[model_name]["y_true"].append(oof_targets)

            # Generate Test Predictions
            test_loader = get_dataloader(
                test_df, final_img_size, model_cfg["batch_size"], mode="test"
            )
            test_probs, test_ids = inference(
                model, test_loader, device, use_tta=Config.USE_TTA
            )
            test_predictions[model_name].append(test_probs)

            # Cleanup
            del (
                model,
                optimizer,
                scaler,
                scheduler,
                train_loader,
                val_loader,
                test_loader,
            )
            torch.cuda.empty_cache()
            gc.collect()

    # --- Calibration & Ensemble Aggregation ---
    logger.info("\nStarting Calibration and Ensemble Aggregation...")

    final_test_preds = []
    valid_models_count = 0

    for model_name in Config.MODELS:
        # Concatenate OOF data across folds
        y_pred_oof = np.concatenate(oof_data[model_name]["y_pred"]).flatten()
        y_true_oof = np.concatenate(oof_data[model_name]["y_true"]).flatten()

        # Calculate raw Log Loss
        raw_loss = calculate_metric(y_true_oof, y_pred_oof)
        logger.info(f"Model {model_name} - Raw OOF Log Loss: {raw_loss:.6f}")

        # Quality Gating
        if raw_loss > Config.OOF_THRESHOLD:
            logger.warning(
                f"Model {model_name} discarded (Loss > {Config.OOF_THRESHOLD})"
            )
            continue

        # Train Isotonic Regression Calibrator
        if Config.CALIBRATION_METHOD == "IsotonicRegression":
            iso_reg = IsotonicRegression(out_of_bounds="clip")
            # IsotonicRegression expects 1D array
            iso_reg.fit(y_pred_oof, y_true_oof)

            # Apply to Test Predictions
            # Average raw predictions across folds first?
            # Strategy: Calibrate each fold's prediction?
            # Better: Average raw probs across folds, then calibrate using the global calibrator.
            # (Assuming the distribution shift is consistent)
            avg_raw_test_probs = np.mean(test_predictions[model_name], axis=0).flatten()
            calibrated_test_probs = iso_reg.transform(avg_raw_test_probs)

            final_test_preds.append(calibrated_test_probs)
            valid_models_count += 1
            logger.info(f"Model {model_name} - Included in ensemble.")

        else:
            # No calibration
            avg_raw_test_probs = np.mean(test_predictions[model_name], axis=0).flatten()
            final_test_preds.append(avg_raw_test_probs)
            valid_models_count += 1

    if valid_models_count == 0:
        logger.error(
            "No models passed quality gating! Using raw average of all models as fallback."
        )
        for model_name in Config.MODELS:
            avg_raw_test_probs = np.mean(test_predictions[model_name], axis=0).flatten()
            final_test_preds.append(avg_raw_test_probs)

    # Ensemble Averaging
    ensemble_preds = np.mean(final_test_preds, axis=0)

    # Save Submission
    submission_df = pd.DataFrame({"id": test_ids.astype(int), "label": ensemble_preds})

    # Ensure IDs are sorted
    submission_df = submission_df.sort_values("id")

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(sub_path, index=False)
    logger.info(f"Submission saved to {sub_path}")
