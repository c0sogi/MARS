import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import get_logger, seed_everything
from library.dataset import get_dataloaders
from library.model import get_model, get_optimizer, get_scheduler, get_loss_fn

logger = get_logger("train")


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model.
        loader: DataLoader for training data.
        optimizer: Optimizer.
        criterion: Loss function.
        device: 'cuda' or 'cpu'.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Accumulate loss weighted by batch size
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size

    epoch_loss = running_loss / count
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using Log Loss.

    Args:
        model: PyTorch model.
        loader: DataLoader for validation data.
        device: 'cuda' or 'cpu'.

    Returns:
        float: Log Loss score.
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            # Get probability of class 1 (Dog)
            dog_probs = probs[:, 1].cpu().numpy()

            preds.extend(dog_probs)
            targets.extend(labels.numpy())

    # Calculate Log Loss
    # labels=[0, 1] ensures correct calculation even if a batch misses a class
    metric = log_loss(targets, preds, labels=[0, 1])
    return metric


def predict(model, loader, device, use_tta=False):
    """
    Generates predictions for the test set.

    Args:
        model: PyTorch model.
        loader: DataLoader for test data.
        device: 'cuda' or 'cpu'.
        use_tta (bool): Whether to use Test Time Augmentation (Horizontal Flip).

    Returns:
        tuple: (ids, probabilities)
    """
    model.eval()
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # Forward pass 1: Original
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            if use_tta:
                # Forward pass 2: Horizontal Flip
                # Flip along width dimension (dim 3: N, C, H, W)
                images_flipped = torch.flip(images, dims=[3])
                outputs_flipped = model(images_flipped)
                probs_flipped = torch.softmax(outputs_flipped, dim=1)

                # Average probabilities
                probs = (probs + probs_flipped) / 2.0

            # Store predictions (Class 1: Dog)
            dog_probs = probs[:, 1].cpu().numpy()

            all_ids.extend(ids.numpy())
            all_probs.extend(dog_probs)

    return all_ids, all_probs


def run_training():
    """
    Orchestrates the training process, evaluation, and submission generation.
    """
    # 1. Setup
    seed_everything(Config.SEED)

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders()

    # Initialize Model, Optimizer, Scheduler, Loss
    model = get_model()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    criterion = get_loss_fn()

    # Training Loop Variables
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Early Stopping Parameters
    patience = 3
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )

        # Validate
        val_loss = validate(model, val_loader, Config.DEVICE)

        # Logging (Full precision)
        logger.info(
            f"Epoch {epoch + 1}/{Config.EPOCHS} - Train Loss: {train_loss} - Val Log Loss: {val_loss}"
        )

        # Update Scheduler
        scheduler.step()

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Validation loss improved. Saved model to {best_model_path}")
        else:
            patience_counter += 1
            logger.info(
                f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
            )
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    # 2. Submission Generation
    logger.info("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

    logger.info(f"Generating predictions (TTA={Config.USE_TTA})...")
    ids, probs = predict(model, test_loader, Config.DEVICE, use_tta=Config.USE_TTA)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": ids, "label": probs})

    # Save Submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
