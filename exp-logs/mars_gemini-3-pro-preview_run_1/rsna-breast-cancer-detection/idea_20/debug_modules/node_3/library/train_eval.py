import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging

from library.config import Config
from library.utils import get_logger, probabilistic_f1, save_checkpoint, seed_everything
from library.data import get_dataloaders
from library.model import SiameseEfficientNet

# Initialize Logger
logger = get_logger("train_eval")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device.

    Returns:
        tuple: (average_loss, pF1_score)
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    # Iterate without progress bar
    for batch_idx, (inputs, targets) in enumerate(loader):
        # Unpack inputs
        img_target = inputs["target"].to(device, non_blocking=True)
        img_contra = inputs["contra"].to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).view(-1, 1)

        # Forward pass
        optimizer.zero_grad()
        logits = model(img_target, img_contra)

        # Loss calculation
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Optimization step (No gradient clipping as per requirements)
        optimizer.step()

        # Metrics accumulation
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs)

    # Calculate epoch metrics
    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    epoch_pf1 = probabilistic_f1(all_targets, all_probs)

    return epoch_loss, epoch_pf1


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Torch device.

    Returns:
        tuple: (average_loss, pF1_score)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            img_target = inputs["target"].to(device, non_blocking=True)
            img_contra = inputs["contra"].to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).view(-1, 1)

            logits = model(img_target, img_contra)
            loss = criterion(logits, targets)

            running_loss += loss.item() * targets.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs)

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.vstack(all_targets)
    all_probs = np.vstack(all_probs)
    epoch_pf1 = probabilistic_f1(all_targets, all_probs)

    return epoch_loss, epoch_pf1


def inference(model, loader, device):
    """
    Generates predictions for the test set and saves submission.

    Args:
        model: The trained PyTorch model.
        loader: DataLoader for test data.
        device: Torch device.
    """
    model.eval()
    results = []

    logger.info("Starting inference on test set...")

    with torch.no_grad():
        for inputs, prediction_ids in loader:
            img_target = inputs["target"].to(device, non_blocking=True)
            img_contra = inputs["contra"].to(device, non_blocking=True)

            logits = model(img_target, img_contra)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for pid, prob in zip(prediction_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Aggregate by prediction_id (Max probability across views)
    # This handles the case where multiple images map to the same prediction_id (e.g. different views)
    df_sub = df_results.groupby("prediction_id")["cancer"].max().reset_index()

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Submission shape: {df_sub.shape}")


def run_training(debug=Config.DEBUG, epochs=Config.NUM_EPOCHS):
    """
    Main function to run the training pipeline.

    Args:
        debug (bool): If True, runs on a small subset of data.
        epochs (int): Number of training epochs.
    """
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, load_cached_data=True
    )

    # Initialize Model
    model = SiameseEfficientNet().to(device)

    # Loss Function
    # Weighted BCE Loss to handle class imbalance (approx 1:47)
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Training Loop
    best_pf1 = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")

        # Train
        train_loss, train_pf1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        logger.info(f"Train Loss: {train_loss} | Train pF1: {train_pf1}")

        # Validate
        val_loss, val_pf1 = evaluate(model, val_loader, criterion, device)
        # Printing full precision as requested
        logger.info(f"Val Loss: {val_loss}")
        logger.info(f"Val pF1: {val_pf1}")

        # Scheduler Step
        scheduler.step()

        # Checkpointing & Early Stopping
        is_best = val_pf1 > best_pf1
        if is_best:
            best_pf1 = val_pf1
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_pf1": best_pf1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
            )
            logger.info(f"New Best Model Saved! pF1: {best_pf1}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            logger.info(
                f"Early stopping triggered after {patience_counter} epochs without improvement."
            )
            break

    # Load best model for inference
    if os.path.exists(best_model_path):
        logger.info("Loading best model for inference...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        logger.warning("Best model not found, using current model weights.")

    # Run Inference
    inference(model, test_loader, device)
