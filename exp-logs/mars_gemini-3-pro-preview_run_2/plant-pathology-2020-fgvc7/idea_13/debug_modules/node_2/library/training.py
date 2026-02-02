import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from library.config import Config
from library.utils import (
    seed_everything,
    get_class_weights,
    reconstruct_4_class_probabilities,
    calculate_metric,
    save_checkpoint,
    get_device,
)
from library.data import get_train_val_loaders, get_folds_data
from library.models import AppleNet


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Label smoothing factor
    smoothing = Config.LABEL_SMOOTHING

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # Apply label smoothing to targets manually
        # targets are binary (0 or 1).
        # smoothed = target * (1 - alpha) + 0.5 * alpha
        with torch.no_grad():
            smoothed_targets = targets * (1.0 - smoothing) + 0.5 * smoothing

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, smoothed_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    Reconstructs 4-class probabilities to calculate the competition metric.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_rust_probs = []
    all_scab_probs = []
    all_rust_targets = []
    all_scab_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            # Calculate loss on raw targets (or smoothed, but usually raw for val)
            # We use raw targets for validation loss to measure true error
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for the 2 binary classes
            probs = torch.sigmoid(outputs)

            all_rust_probs.append(probs[:, 0].cpu().numpy())
            all_scab_probs.append(probs[:, 1].cpu().numpy())
            all_rust_targets.append(targets[:, 0].cpu().numpy())
            all_scab_targets.append(targets[:, 1].cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate results
    rust_probs = np.concatenate(all_rust_probs)
    scab_probs = np.concatenate(all_scab_probs)
    rust_targets = np.concatenate(all_rust_targets)
    scab_targets = np.concatenate(all_scab_targets)

    # Reconstruct 4-class probabilities and targets for metric calculation
    # The competition metric is Mean Column-wise ROC AUC on the 4 classes
    y_pred = reconstruct_4_class_probabilities(rust_probs, scab_probs)
    y_true = reconstruct_4_class_probabilities(rust_targets, scab_targets)

    score = calculate_metric(y_true, y_pred)

    return epoch_loss, score


def run_fold(fold_idx, model_config):
    """
    Runs the training pipeline for a specific fold and model configuration.
    """
    seed_everything(Config.SEED)
    device = get_device()

    model_name = model_config["name"]
    print(f"--- Starting Fold {fold_idx} for model {model_name} ---")

    # 1. Data Loaders
    train_loader, val_loader = get_train_val_loaders(
        fold_idx, model_config["img_size"], model_config["batch_size"]
    )

    # 2. Class Weights
    # Get training data for this fold to calculate weights
    full_df = get_folds_data(load_cached_data=True)
    train_df = full_df[full_df["fold"] != fold_idx]

    # Force re-computation or ensure we use weights for this specific split
    # We pass load_cached_data=False to ensure we calculate weights for this specific training set
    pos_weight = get_class_weights(train_df, load_cached_data=False).to(device)

    # 3. Model
    model = AppleNet(
        model_name=model_name,
        pretrained=True,
        dropout_rates=model_config.get("dropout_rates"),
    ).to(device)

    # 4. Optimizer & Criterion
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 5. SWA Setup
    use_swa = Config.USE_SWA
    swa_start = Config.SWA_START_EPOCH
    swa_model = AveragedModel(model).to(device) if use_swa else None
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR) if use_swa else None

    # Standard Scheduler
    # We run standard scheduler until SWA starts (or for all epochs if SWA is off)
    scheduler_epochs = swa_start if use_swa else Config.EPOCHS
    scheduler = CosineAnnealingLR(
        optimizer, T_max=scheduler_epochs, eta_min=Config.MIN_LR
    )

    # 6. Training Loop
    best_score = 0.0
    best_loss = float("inf")
    patience_counter = 0

    # Construct unique save name for this model and fold
    model_save_name = f"best_model_{model_name}_fold_{fold_idx}.pth"

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Determine if we are in SWA phase
        in_swa_phase = use_swa and (epoch >= swa_start)

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Update SWA or Scheduler
        if in_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()

            # Update BN for validation
            # This ensures the SWA model has correct batch norm statistics for the validation pass
            update_bn(train_loader, swa_model, device=device)

            # Validate SWA model
            val_loss, val_score = validate(swa_model, val_loader, criterion, device)
            current_model_state = swa_model.state_dict()
        else:
            scheduler.step()
            # Validate standard model
            val_loss, val_score = validate(model, val_loader, criterion, device)
            current_model_state = model.state_dict()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_score:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save Best Model
        # We prioritize AUC. If AUC is equal, look at loss.
        if val_score > best_score:
            best_score = val_score
            best_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": current_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": best_score,
                },
                filename=model_save_name,
            )
            print(f"  >>> New Best Model Saved (Score: {best_score:.6f})")
        else:
            patience_counter += 1

        # Early Stopping
        # If performance doesn't improve for PATIENCE epochs, stop.
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Fold {fold_idx} finished. Best AUC: {best_score:.6f}")

    # Clean up to save memory
    del model, swa_model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_score
