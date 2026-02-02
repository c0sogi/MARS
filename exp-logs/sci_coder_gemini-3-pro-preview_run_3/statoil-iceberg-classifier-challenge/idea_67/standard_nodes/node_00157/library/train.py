import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

import library.config as config
from library.utils import set_seed, setup_logging
from library.model import LeakyAttentiveIsomorphicCNN
from library.data_loader import get_fold_loaders, get_test_loader

# Initialize logger
logger = setup_logging("train.log")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device (CPU/GPU).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for batch in loader:
        images = batch["image"].to(device)
        angles = batch["angle"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, angles)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Torch device (CPU/GPU).

    Returns:
        float: Average loss for the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            logits = model(images, angles)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def run_fold_training(fold_idx, load_cached_data=True):
    """
    Executes the training pipeline for a single fold with Early Stopping.

    Args:
        fold_idx (int): The index of the fold to train.
        load_cached_data (bool): Whether to load pre-processed data from cache.

    Returns:
        float: The best validation loss achieved.
    """
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training for Fold {fold_idx} on device: {device}")

    # Get DataLoaders
    train_loader, val_loader = get_fold_loaders(
        fold_idx, load_cached_data=load_cached_data
    )

    # Initialize Model
    model = LeakyAttentiveIsomorphicCNN().to(device)

    # Optimizer (AdamW with constant LR)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Loss Function (BCEWithLogitsLoss includes Sigmoid)
    criterion = nn.BCEWithLogitsLoss()

    # Checkpoint Path
    best_model_path = os.path.join(
        config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
    )

    # Early Stopping Tracking
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        logger.info(
            f"Fold {fold_idx} | Epoch {epoch + 1}/{config.NUM_EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            logger.info(f"Early stopping counter: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    logger.info(f"Fold {fold_idx} training finished. Best Val Loss: {best_val_loss}")
    return best_val_loss


def train_all_folds(load_cached_data=True):
    """
    Sequentially trains all folds.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        list: List of best validation losses for each fold.
    """
    losses = []
    for fold in range(config.NUM_FOLDS):
        loss = run_fold_training(fold, load_cached_data=load_cached_data)
        losses.append(loss)

    avg_loss = sum(losses) / len(losses)
    logger.info(f"All folds finished. Average Best Val Loss: {avg_loss}")
    return losses


def generate_submission(load_cached_data=True):
    """
    Generates submission file by averaging predictions from all fold models.
    Saves the result to SUBMISSION_PATH.
    """
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting submission generation...")

    # Load Test Data
    test_loader = get_test_loader(load_cached_data=load_cached_data)

    # Container for accumulated probabilities: key = id, value = list of probs
    id_to_probs = {}

    # Iterate over folds
    for fold_idx in range(config.NUM_FOLDS):
        logger.info(f"Inference with model from Fold {fold_idx}")

        # Load Model
        model = LeakyAttentiveIsomorphicCNN().to(device)
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )

        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} not found. Skipping fold.")
            continue

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                angles = batch["angle"].to(device)
                ids = batch["id"]

                # Forward pass (Logits)
                logits = model(images, angles)

                # Convert to Probability
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                # Store by ID
                for img_id, prob in zip(ids, probs):
                    if img_id not in id_to_probs:
                        id_to_probs[img_id] = []
                    id_to_probs[img_id].append(prob)

    # Average predictions
    final_preds = []
    for img_id, probs in id_to_probs.items():
        if len(probs) > 0:
            avg_prob = sum(probs) / len(probs)
            final_preds.append({"id": img_id, "is_iceberg": avg_prob})
        else:
            logger.warning(f"No predictions found for ID {img_id}")

    # Create DataFrame
    df_sub = pd.DataFrame(final_preds)

    # Ensure directory exists
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Save
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {config.SUBMISSION_PATH}")
