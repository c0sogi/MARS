import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import get_score


def dice_loss(logits, targets, smooth=1e-6):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        logits: (B, 1, H, W) Raw output from the model (before sigmoid).
        targets: (B, 1, H, W) Binary ground truth masks.
        smooth: Smoothing factor to prevent division by zero.
    Returns:
        Tensor: 1 - Dice Coefficient.
    """
    probs = torch.sigmoid(logits)

    # Flatten to (B, -1)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def train_one_epoch(model, loader, optimizer, scaler, device):
    """
    Trains the model for one epoch.
    """
    model.train()

    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    seg_loss_sum = 0.0
    sample_count = 0

    cls_criterion = nn.BCEWithLogitsLoss()

    for images, targets, masks in loader:
        images = images.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        with autocast():
            cls_logits, seg_logits = model(images)

            # 1. Classification Loss
            loss_cls = cls_criterion(cls_logits, targets)

            # 2. Segmentation Loss (Dice)
            # Calculate Dice loss per sample (B,)
            d_loss = dice_loss(seg_logits, masks)

            # 3. Masking Logic
            # We supervise segmentation if:
            # a) The image has a ground truth mask (sum > 0)
            # b) The image is a confirmed negative (all classification targets are 0)
            has_mask = masks.view(batch_size, -1).sum(dim=1) > 0
            is_negative = targets.sum(dim=1) == 0

            # valid_mask is 1.0 where we want to apply loss, 0.0 otherwise
            valid_mask = (has_mask | is_negative).float()

            if valid_mask.sum() > 0:
                # Average loss only over valid samples
                loss_seg = (d_loss * valid_mask).sum() / valid_mask.sum()
            else:
                loss_seg = torch.tensor(0.0, device=device)

            # Total Loss
            loss = loss_cls + Config.AUX_LOSS_WEIGHT * loss_seg

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics (weighted by batch size for accurate epoch average)
        total_loss_sum += loss.item() * batch_size
        cls_loss_sum += loss_cls.item() * batch_size
        seg_loss_sum += loss_seg.item() * batch_size
        sample_count += batch_size

    avg_loss = total_loss_sum / sample_count
    avg_cls = cls_loss_sum / sample_count
    avg_seg = seg_loss_sum / sample_count

    return avg_loss, avg_cls, avg_seg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set and returns the macro-averaged AUC.
    """
    model.eval()
    preds = []
    targets_list = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)

            cls_logits, _ = model(images)
            probs = torch.sigmoid(cls_logits)

            preds.append(probs.cpu().numpy())
            targets_list.append(targets.numpy())

    preds = np.concatenate(preds, axis=0)
    targets_list = np.concatenate(targets_list, axis=0)

    score = get_score(targets_list, preds)
    return score


def predict(model, loader, device):
    """
    Runs inference on the test set.
    """
    model.eval()
    preds = []
    uids = []

    with torch.no_grad():
        for images, batch_uids in loader:
            images = images.to(device)

            cls_logits, _ = model(images)
            probs = torch.sigmoid(cls_logits)

            preds.append(probs.cpu().numpy())
            uids.extend(batch_uids)

    preds = np.concatenate(preds, axis=0)
    return uids, preds


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=Config.EPOCHS,
    patience=5,
):
    """
    Main training loop with Early Stopping.
    """
    scaler = GradScaler()
    best_score = -np.inf
    best_epoch = -1
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        train_loss, train_cls, train_seg = train_one_epoch(
            model, train_loader, optimizer, scaler, device
        )
        val_score = validate(model, val_loader, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}: Train Loss={train_loss}, Cls Loss={train_cls}, Seg Loss={train_seg}, Val AUC={val_score}"
        )

        if scheduler is not None:
            scheduler.step()

        # Checkpoint and Early Stopping
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved at epoch {epoch+1} with AUC {val_score}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered at epoch {epoch+1} (No improvement for {patience} epochs)."
            )
            break

    print(f"Training finished. Best AUC: {best_score} at Epoch {best_epoch}")


def generate_submission(model, test_loader, device, output_path=Config.SUBMISSION_PATH):
    """
    Loads the best model, runs inference on the test set, and saves the submission file.
    """
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}...")
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            "Warning: Best model not found. Using current model weights for inference."
        )

    print("Running inference on test set...")
    uids, preds = predict(model, test_loader, device)

    df_sub = pd.DataFrame(preds, columns=Config.LABELS)
    df_sub.insert(0, "StudyInstanceUID", uids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
