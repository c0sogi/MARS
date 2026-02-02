import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from library.config import Config
from library.utils import AverageMeter, get_score, Logger


class AppleLoss(nn.Module):
    """
    Custom Loss wrapper to handle Label Smoothing with BCEWithLogitsLoss.
    """

    def __init__(self, smoothing=0.05):
        super(AppleLoss, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # Apply label smoothing to multi-label targets
        # y_ls = y * (1 - alpha) + 0.5 * alpha
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler):
    """
    Trains the model for one epoch.
    """
    model.train()
    loss_meter = AverageMeter()
    criterion = AppleLoss(smoothing=Config.LABEL_SMOOTHING)

    for step, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Automatic Mixed Precision
        with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()

        # Gradient Clipping
        if Config.MAX_GRAD_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def valid_one_epoch(model, data_loader, device):
    """
    Validates the model for one epoch.
    """
    model.eval()
    loss_meter = AverageMeter()
    # Use standard BCE for validation to measure true error against hard targets
    criterion = nn.BCEWithLogitsLoss()

    preds = []
    valid_labels = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, targets)

            loss_meter.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            preds.append(probs.cpu().numpy())
            valid_labels.append(targets.cpu().numpy())

    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    # Calculate F1 Score
    score = get_score(valid_labels, preds, threshold=Config.THRESHOLD)

    return loss_meter.avg, score


def fit(
    model, train_loader, val_loader, optimizer, scheduler, device, epochs=Config.EPOCHS
):
    """
    Main training loop with Early Stopping and Model Checkpointing.
    """
    logger = Logger(Config.LOG_PATH)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)

    best_score = -np.inf
    patience = 5
    patience_counter = 0

    logger.print(f"Start Training on device: {device}")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, scaler
        )

        # Validate
        val_loss, val_score = valid_one_epoch(model, val_loader, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - start_time

        logger.print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Time: {elapsed:.0f}s - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val F1: {val_score:.6f}"
        )

        # Model Checkpointing & Early Stopping
        if val_score > best_score:
            best_score = val_score
            logger.print(
                f"Validation Score Improved ({best_score:.6f}). Saving model..."
            )
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            logger.print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.print("Early stopping triggered.")
            break

    logger.print(f"Training Complete. Best Validation F1: {best_score:.6f}")


def predict(model, test_loader, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    preds = []

    # Run Inference
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                outputs = model(images)
                probs = torch.sigmoid(outputs)
            preds.append(probs.cpu().numpy())

    preds = np.concatenate(preds)

    # Convert probabilities to labels
    # Thresholding
    binary_preds = (preds > Config.THRESHOLD).astype(int)

    # Map indices to class names
    submission_rows = []
    # We assume test_loader is not shuffled, so order matches test_loader.dataset.df
    image_ids = test_loader.dataset.df["image"].values

    for i, row in enumerate(binary_preds):
        image_id = image_ids[i]
        labels = []
        for idx, is_present in enumerate(row):
            if is_present:
                labels.append(Config.CLASSES[idx])

        # Join labels with space
        label_str = " ".join(labels)
        # If no label predicted, it might be appropriate to leave empty or default.
        # Based on dataset, 'healthy' is a class. If model is good, it picks one.
        # If empty, we leave it empty (format requirement is space delimited list).

        submission_rows.append({"image": image_id, "labels": label_str})

    # Create DataFrame and save
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
