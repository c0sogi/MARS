import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.utils import SoftTargetCrossEntropy, get_logger
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, device, loss_fn):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        dataloader: DataLoader for training data.
        optimizer: Optimizer instance.
        device: Device to train on (cpu or cuda).
        loss_fn: Loss function (SoftTargetCrossEntropy).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        dataloader: DataLoader for validation data.
        device: Device to evaluate on.
        loss_fn: Loss function (CrossEntropyLoss).

    Returns:
        tuple: (Average Loss, Accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    dataset_size = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        running_loss += loss.item() * batch_size

        # Get predictions
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    epoch_acc = correct_predictions.double() / dataset_size

    return epoch_loss, epoch_acc.item()


@torch.no_grad()
def predict_with_tta(model, dataloader, device):
    """
    Generates predictions using Test Time Augmentation (Horizontal Flip).

    Args:
        model: The neural network model.
        dataloader: DataLoader for test data.
        device: Device to predict on.

    Returns:
        tuple: (List of image_ids, List of predicted labels)
    """
    model.eval()
    image_ids = []
    predictions = []

    for images, ids in dataloader:
        images = images.to(device)

        # Forward pass: Original Image
        outputs_orig = model(images)

        # Forward pass: Horizontally Flipped Image
        # Tensor shape is [B, C, H, W], so we flip on dim 3 (width)
        images_flipped = torch.flip(images, dims=[3])
        outputs_flip = model(images_flipped)

        # Average logits
        outputs_avg = (outputs_orig + outputs_flip) / 2.0

        # Get final predictions
        _, preds = torch.max(outputs_avg, 1)

        image_ids.extend(ids)
        predictions.extend(preds.cpu().numpy())

    return image_ids, predictions


def run_training(model, train_loader, val_loader, test_loader, cfg: Config):
    """
    Main execution function for training, evaluation, and submission generation.

    Args:
        model: The neural network model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        test_loader: DataLoader for testing.
        cfg: Configuration object.
    """
    # Setup logging
    log_path = os.path.join(cfg.working_dir, "train.log")
    logger = get_logger(log_path)

    device = cfg.device
    model = model.to(device)

    # Initialize Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # Initialize Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
    )

    # Loss Functions
    # Train: SoftTargetCrossEntropy for MixUp/CutMix
    train_criterion = SoftTargetCrossEntropy()
    # Val: Standard CrossEntropyLoss for integer labels
    val_criterion = nn.CrossEntropyLoss()

    # Early Stopping Variables
    best_acc = 0.0
    best_loss = float("inf")
    patience_counter = 0

    logger.info(f"Starting training on device: {device}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")

    for epoch in range(cfg.epochs):
        start_time = time.time()

        # Training Step
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, train_criterion
        )

        # Validation Step
        val_loss, val_acc = evaluate(model, val_loader, device, val_criterion)

        # Scheduler Step
        scheduler.step()

        end_time = time.time()
        epoch_time = end_time - start_time

        # Logging
        logger.info(f"Epoch {epoch+1}/{cfg.epochs} | Time: {epoch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.6f}")
        logger.info(f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.10f}")

        # Checkpoint & Early Stopping Logic
        # Priority: Higher Accuracy, then Lower Loss
        improved = False
        if val_acc > best_acc:
            improved = True
        elif val_acc == best_acc and val_loss < best_loss:
            improved = True

        if improved:
            best_acc = val_acc
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), cfg.best_model_path)
            logger.info(f"Metric improved. Model saved to {cfg.best_model_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{cfg.patience}")

        if patience_counter >= cfg.patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training complete. Best Validation Accuracy: {best_acc:.10f}")

    # ==========================================
    # Inference & Submission
    # ==========================================
    logger.info("Starting inference on test set with TTA...")

    # Load the best model weights
    if os.path.exists(cfg.best_model_path):
        model.load_state_dict(torch.load(cfg.best_model_path, map_location=device))
    else:
        logger.warning("Best model file not found. Using current model weights.")

    # Generate predictions
    test_ids, test_preds = predict_with_tta(model, test_loader, device)

    # Create submission dataframe
    submission_df = pd.DataFrame({"image_id": test_ids, "label": test_preds})

    # Save submission
    # Ensure directory exists (Config creates working_dir, but submission path might be in subdir)
    os.makedirs(os.path.dirname(cfg.submission_path), exist_ok=True)
    submission_df.to_csv(cfg.submission_path, index=False)

    logger.info(f"Submission saved to {cfg.submission_path}")

    return submission_df
