import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from library.config import CFG
from library.utils import seed_everything, calculate_metric, get_class_weights
from library.dataset import get_loaders
from library.models import AppleNet


def train_one_epoch(epoch, model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using AMP and Gradient Accumulation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Initialize GradScaler for AMP
    scaler = GradScaler(enabled=CFG.use_amp)

    optimizer.zero_grad()

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Apply Label Smoothing manually to targets
        # y_smooth = y * (1 - alpha) + 0.5 * alpha
        if CFG.label_smoothing > 0:
            labels_smooth = (
                labels * (1 - CFG.label_smoothing) + 0.5 * CFG.label_smoothing
            )
        else:
            labels_smooth = labels

        # Forward pass with AMP
        with autocast(enabled=CFG.use_amp):
            logits = model(images)
            loss = criterion(logits, labels_smooth)
            # Normalize loss for gradient accumulation
            loss = loss / CFG.gradient_accumulation_steps

        # Backward pass with Scaler
        scaler.scale(loss).backward()

        # Step Optimizer (Gradient Accumulation)
        if (batch_idx + 1) % CFG.gradient_accumulation_steps == 0 or (
            batch_idx + 1
        ) == len(loader):
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Gradient Clipping
            if CFG.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Update stats (scale loss back up for reporting)
        running_loss += loss.item() * CFG.gradient_accumulation_steps * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(epoch, model, loader, criterion, device):
    """
    Validates the model for one epoch.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # Forward pass with AMP
            with autocast(enabled=CFG.use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            preds_list.append(probs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate predictions and labels
    preds_all = np.concatenate(preds_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)

    # Calculate ROC AUC
    val_score = calculate_metric(labels_all, preds_all)

    return epoch_loss, val_score


def run_fold(fold, train_df, val_df, model_name, img_size):
    """
    Runs training for a specific fold.

    Args:
        fold (int): Current fold number.
        train_df (pd.DataFrame): Training data for this fold.
        val_df (pd.DataFrame): Validation data for this fold.
        model_name (str): Name of the backbone model.
        img_size (int): Input image resolution.
    """
    seed_everything(CFG.seed)

    device = torch.device(CFG.device)

    # --- Data Loaders ---
    # We pass empty test_df as we don't need test loader during training loop
    # We can pass val_df as test_df to get_loaders to satisfy the signature if needed,
    # but get_loaders takes (train, val, test).
    # We'll just pass val_df for test_df placeholder since we won't use the test loader here.
    train_loader, val_loader, _ = get_loaders(
        train_df, val_df, val_df, img_size, CFG.batch_size
    )

    # --- Model ---
    model = AppleNet(model_name=model_name, pretrained=True)
    model.to(device)

    # --- Loss Function ---
    # Calculate class weights if enabled
    pos_weight = None
    if CFG.use_class_weights:
        # Calculate weights based on training data distribution
        weights = get_class_weights(train_df, CFG.target_cols)
        pos_weight = weights.to(device)

    # BCEWithLogitsLoss combines Sigmoid and BCE.
    # It supports pos_weight for imbalance.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )

    # --- Training Loop ---
    best_score = -np.inf
    best_loss = np.inf
    patience_counter = 0

    print(
        f"Starting training for Fold {fold} | Model: {model_name} | Image Size: {img_size}"
    )
    print(f"Training samples: {len(train_df)} | Validation samples: {len(val_df)}")

    for epoch in range(CFG.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_score = valid_one_epoch(
            epoch, model, val_loader, criterion, device
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{CFG.epochs} "
            f"[Time: {elapsed:.0f}s] "
            f"Train Loss: {train_loss:.5f} "
            f"Val Loss: {val_loss:.5f} "
            f"Val AUC: {val_score:.10f}"
        )

        # Checkpointing based on ROC AUC
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0

            # Save best model
            # Sanitize model name for filename
            safe_model_name = model_name.replace(".", "_")
            save_path = os.path.join(
                CFG.output_dir, f"best_model_{safe_model_name}_fold_{fold}.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Model Saved! Best AUC: {best_score:.10f}")

        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= CFG.early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Fold {fold} finished. Best Validation AUC: {best_score:.10f}")

    # Clean up to save memory
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_score
