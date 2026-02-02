import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from library.config import Config
from library.utils import compute_roc_auc, save_checkpoint, set_seed

# =========================================================================
# Mixup Helpers
# =========================================================================


def mixup_data(x, y, alpha=1.0, device=Config.DEVICE):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss for mixed inputs.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =========================================================================
# Core Functions
# =========================================================================


def train_one_epoch(model, dataloader, optimizer, criterion, device, mixup_alpha):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Apply Mixup
        if mixup_alpha > 0:
            images, targets_a, targets_b, lam = mixup_data(
                images, labels, mixup_alpha, device
            )
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """
    Validates the model and computes ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            num_batches += 1

            # Store predictions (sigmoid applied for probability) and targets
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        auc_score = compute_roc_auc(all_targets, all_preds)
    else:
        auc_score = 0.0

    return avg_loss, auc_score


def run_training(
    model,
    train_loader,
    val_loader,
    swa_start_epoch,
    save_path,
    mixup_alpha,
    patience=10,
):
    """
    Orchestrates the training process including SWA and Early Stopping.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        swa_start_epoch: Epoch to start Stochastic Weight Averaging.
        save_path: Path to save the final SWA model.
        mixup_alpha: Alpha parameter for Mixup.
        patience: Early stopping patience (only active before SWA).
    """
    device = Config.DEVICE
    set_seed(Config.SEED)

    # Optimizer: AdamW (Correcting previous failure with SGD)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    # SWA Setup
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)

    # Tracking
    best_val_auc = 0.0
    epochs_no_improve = 0
    base_model_save_path = save_path.replace("_swa.pth", "_base_best.pth")

    print(
        f"Starting training. SWA starts at epoch {swa_start_epoch}. Mixup alpha: {mixup_alpha}"
    )

    start_time = time.time()

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, mixup_alpha
        )

        # SWA Logic
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            phase = "SWA"
        else:
            phase = "Base"

        # Validation (Always validate base model to track progress)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} [{phase}] - "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val AUC: {val_auc:.10f}"
        )

        # Checkpointing and Early Stopping (Only applicable before SWA starts)
        if epoch < swa_start_epoch:
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_no_improve = 0
                save_checkpoint(model, optimizer, epoch, base_model_save_path)
                # print(f"  New best base model saved to {base_model_save_path}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. (Pre-SWA)")
                # Important: If we stop early before SWA, we must ensure we have a valid model.
                # However, the strategy mandates SWA. If we stop early, we might skip SWA.
                # To adhere strictly to the strategy: if performance degrades significantly before SWA,
                # we might just load the best model and proceed to SWA, or stop.
                # Given the prompt "Explicitly disable Early Stopping during the SWA phase",
                # we implies we CAN stop before. But usually SWA requires running late.
                # We will just break here.
                break
        else:
            # During SWA, we do not stop early. We continue to collect averages.
            pass

    total_time = time.time() - start_time
    print(f"Training loop completed in {total_time:.2f}s.")

    # Finalize SWA
    if epoch >= swa_start_epoch:
        print("Updating SWA BatchNorm statistics...")
        update_bn(train_loader, swa_model, device=device)

        # Validate SWA Model
        # Note: update_bn puts model in train mode, need to switch to eval for validation
        # But validate() function handles .eval() internally.
        swa_val_loss, swa_val_auc = validate(swa_model, val_loader, criterion, device)
        print(
            f"Final SWA Model - Val Loss: {swa_val_loss:.6f}, Val AUC: {swa_val_auc:.10f}"
        )

        # Save SWA Model
        save_checkpoint(swa_model, None, epoch, save_path)
        print(f"SWA model saved to {save_path}")
    else:
        print("SWA phase was not reached. SWA model not saved.")
        # Fallback: Ensure the requested save_path exists by copying the best base model
        if os.path.exists(base_model_save_path):
            best_state = torch.load(base_model_save_path)
            torch.save(best_state, save_path)
            print(f"Fallback: Best base model copied to {save_path}")

    return swa_model
