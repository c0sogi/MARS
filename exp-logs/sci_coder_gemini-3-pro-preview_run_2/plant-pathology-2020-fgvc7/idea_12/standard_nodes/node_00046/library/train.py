import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import (
    seed_everything,
    get_logger,
    worker_init_fn,
    calculate_pos_weights,
)
from library.dataset import get_data, AppleDataset
from library.model import AppleDiseaseModel
from library.loss import WeightedBCELoss


def train_one_epoch(loader, model, criterion, optimizer, device, epoch, logger):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        # Forward pass
        # Model handles Multi-Sample Dropout internally during training
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(loader, model, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns: Mean ROC AUC, Average Loss
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for metric calculation
            preds = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Calculate ROC AUC for each column (Rust, Scab)
    # Handle edge case where a fold might (rarely) miss a class in validation
    try:
        auc_scores = []
        for i in range(all_targets.shape[1]):
            # Check if class exists in targets
            if len(np.unique(all_targets[:, i])) > 1:
                score = roc_auc_score(all_targets[:, i], all_preds[:, i])
                auc_scores.append(score)
            else:
                # Fallback if only one class is present (should not happen with StratifiedKFold)
                auc_scores.append(0.5)

        mean_auc = np.mean(auc_scores)
    except Exception as e:
        print(f"Metric calculation error: {e}")
        mean_auc = 0.5

    return mean_auc, avg_loss


def run_training():
    """
    Main training orchestration function.
    """
    seed_everything(Config.SEED)
    logger = get_logger("training")

    # 1. Load and Prepare Data
    # We combine train and val metadata to perform our own 5-Fold CV
    logger.info("Loading and merging datasets for Cross-Validation...")
    train_df_part = get_data("train")
    val_df_part = get_data("val")
    full_df = pd.concat([train_df_part, val_df_part], axis=0).reset_index(drop=True)

    if Config.DEBUG:
        logger.info(
            f"DEBUG Mode: Subsetting data to {Config.DEBUG_SUBSET_SIZE} samples."
        )
        full_df = full_df.head(Config.DEBUG_SUBSET_SIZE)

    # 2. Iterate over Models in Ensemble
    for model_cfg in Config.MODELS:
        model_name = model_cfg["name"]
        logger.info(f"\n{'='*40}\nStarting Training for Model: {model_name}\n{'='*40}")

        # Sanitize model name for file saving
        safe_model_name = model_name.replace(".", "_")

        # 3. Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=Config.FOLDS, shuffle=True, random_state=Config.SEED
        )

        # We split based on 'stratify_label' which is present in the metadata
        fold_iterator = skf.split(full_df, full_df["stratify_label"])

        for fold, (train_idx, val_idx) in enumerate(fold_iterator):
            logger.info(f"\n--- Fold {fold + 1}/{Config.FOLDS} ---")

            # Split Data
            train_sub = full_df.iloc[train_idx].reset_index(drop=True)
            val_sub = full_df.iloc[val_idx].reset_index(drop=True)

            # Dataset & Loader
            train_dataset = AppleDataset(
                train_sub, img_size=model_cfg["img_size"], mode="train"
            )
            val_dataset = AppleDataset(
                val_sub, img_size=model_cfg["img_size"], mode="val"
            )

            # Cite debug_lesson_1: Adjust batching strategy for small debug subsets
            drop_last = True
            if len(train_dataset) < model_cfg["batch_size"]:
                logger.warning(
                    f"Training set size ({len(train_dataset)}) is smaller than batch size "
                    f"({model_cfg['batch_size']}). Disabling drop_last."
                )
                drop_last = False

            train_loader = DataLoader(
                train_dataset,
                batch_size=model_cfg["batch_size"],
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                worker_init_fn=worker_init_fn,
                pin_memory=True,
                drop_last=drop_last,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=model_cfg["batch_size"],
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                worker_init_fn=worker_init_fn,
                pin_memory=True,
            )

            # Calculate Class Weights for this fold
            pos_weights = calculate_pos_weights(train_sub).to(Config.DEVICE)

            # Initialize Model
            model = AppleDiseaseModel(
                model_name=model_name,
                pretrained=True,
                num_classes=Config.NUM_TARGETS,
                gem_p=model_cfg["gem_p"],
                num_msd=model_cfg["num_msd"],
                msd_dropout=model_cfg["msd_dropout"],
            ).to(Config.DEVICE)

            # Loss, Optimizer, Scheduler
            criterion = WeightedBCELoss(
                pos_weights=pos_weights, smoothing=Config.LABEL_SMOOTHING
            )
            optimizer = optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            # Standard Scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS, eta_min=1e-6
            )

            # SWA Setup
            if Config.USE_SWA:
                swa_model = AveragedModel(model)
                swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)
                swa_start = Config.SWA_START_EPOCH
            else:
                swa_model = None
                swa_start = Config.EPOCHS + 1  # Never start

            best_auc = 0.0
            best_model_path = os.path.join(
                Config.WORKING_DIR, f"best_model_{safe_model_name}_fold_{fold}.pth"
            )

            # Training Loop
            for epoch in range(Config.EPOCHS):
                start_time = time.time()

                train_loss = train_one_epoch(
                    train_loader,
                    model,
                    criterion,
                    optimizer,
                    Config.DEVICE,
                    epoch,
                    logger,
                )

                # SWA Logic
                if Config.USE_SWA and epoch >= swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step()

                # Validation
                # We validate the base model during training to track progress
                val_auc, val_loss = validate(
                    val_loader, model, criterion, Config.DEVICE
                )

                elapsed = time.time() - start_time

                # Logging
                logger.info(
                    f"Epoch {epoch+1}/{Config.EPOCHS} | "
                    f"Time: {elapsed:.1f}s | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val AUC: {val_auc:.6f}"
                )

                # Save Best Base Model
                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"  >>> New Best AUC: {best_auc:.6f} (Saved)")

            # End of Fold Processing

            # Handle SWA Finalization
            if Config.USE_SWA:
                logger.info("Finalizing SWA Model...")
                # Update BN statistics for the SWA model
                update_bn(train_loader, swa_model, device=Config.DEVICE)

                # Evaluate SWA Model
                swa_auc, swa_loss = validate(
                    val_loader, swa_model, criterion, Config.DEVICE
                )
                logger.info(
                    f"SWA Model Results - Val AUC: {swa_auc:.6f} | Val Loss: {swa_loss:.4f}"
                )

                # Save SWA Model (overwriting best if desired, or as separate)
                # Strategy: If SWA is better, or just save it as the 'final' model for this fold.
                # Given the strategy emphasizes convergence, we often trust SWA more for generalization.
                # We will save SWA model as the primary checkpoint for inference if it exists.
                swa_path = os.path.join(
                    Config.WORKING_DIR, f"swa_model_{safe_model_name}_fold_{fold}.pth"
                )
                torch.save(swa_model.state_dict(), swa_path)

                # For this solution, we will stick to the 'best_model' found during training
                # UNLESS SWA performed better or we want to force SWA usage.
                # Let's keep both, but the inference script will likely look for a specific pattern.
                # We'll rename the SWA model to 'best_model' if it outperforms,
                # otherwise we keep the best base model.
                if swa_auc > best_auc:
                    logger.info(
                        "SWA outperformed best base model. Replacing best model file."
                    )
                    torch.save(swa_model.state_dict(), best_model_path)
                else:
                    logger.info(
                        "SWA did not outperform best base model. Keeping best base model."
                    )

            # Clean up to save memory
            del model, optimizer, scheduler, criterion
            if Config.USE_SWA:
                del swa_model, swa_scheduler
            torch.cuda.empty_cache()

    logger.info("Training Complete.")


if __name__ == "__main__":
    # This block is not required by the prompt instructions but good for local testing
    pass
