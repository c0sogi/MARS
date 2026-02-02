import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_logger
from library.model import get_model
from library.dataset import get_dataloaders


def train_one_epoch(model, dataloader, criterion, optimizer, device, mixup_alpha=0.0):
    """
    Trains the model for one epoch, optionally using MixUp.
    Cite solution_lesson_node_00001: Implementing MixUp as active regularization.

    Args:
        model (torch.nn.Module): The neural network.
        dataloader (DataLoader): Training data loader.
        criterion (loss): Loss function.
        optimizer (torch.optim): Optimizer.
        device (torch.device): Computation device.
        mixup_alpha (float): Alpha parameter for Beta distribution.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # MixUp Implementation
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(images.size(0)).to(device)

            mixed_images = lam * images + (1 - lam) * images[index]
            y_a, y_b = labels, labels[index]

            outputs = model(mixed_images)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

            # For accuracy monitoring, we compare against the dominant label
            # This is an approximation for monitoring purposes
            _, preds = torch.max(outputs, 1)
            # We count it correct if it matches the label with higher weight
            if lam > 0.5:
                correct_predictions += torch.sum(preds == y_a.data)
            else:
                correct_predictions += torch.sum(preds == y_b.data)

        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    return epoch_loss, epoch_acc.item()


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The neural network.
        dataloader (DataLoader): Validation data loader.
        criterion (loss): Loss function.
        device (torch.device): Computation device.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    return epoch_loss, epoch_acc.item()


def train_model(cfg=Config):
    """
    Main function to handle the training loop, validation, and early stopping.

    Args:
        cfg (Config): Configuration class.

    Returns:
        model: The trained model with best weights loaded.
    """
    logger = get_logger(os.path.join(cfg.WORKING_DIR, "train.log"))
    device = torch.device(cfg.DEVICE)

    logger.info(f"Using device: {device}")

    # Data Loaders
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # Model
    model = get_model(cfg.NUM_CLASSES, device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )

    # Training Loop
    best_acc = 0.0
    patience = 3
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(cfg.NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            mixup_alpha=cfg.MIXUP_ALPHA,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}")
        logger.info(f"Train Loss: {train_loss:.10f} | Train Acc: {train_acc:.10f}")
        logger.info(f"Val Loss:   {val_loss:.10f} | Val Acc:   {val_acc:.10f}")

        # Early Stopping and Model Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
            logger.info(
                f"Validation accuracy improved. Model saved to {cfg.MODEL_SAVE_PATH}"
            )
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    logger.info(f"Training complete. Best Validation Accuracy: {best_acc:.10f}")

    # Load best weights
    if os.path.exists(cfg.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))

    return model


def predict(cfg=Config):
    """
    Loads the best model, performs inference on the test set, and saves the submission file.

    Args:
        cfg (Config): Configuration class.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    logger = get_logger(os.path.join(cfg.WORKING_DIR, "inference.log"))
    device = torch.device(cfg.DEVICE)

    logger.info("Starting inference...")

    # Load Test Data
    _, _, test_loader = get_dataloaders(cfg)

    # Load Model
    model = get_model(cfg.NUM_CLASSES, device)

    # Load Weights
    if os.path.exists(cfg.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))
        logger.info(f"Loaded model weights from {cfg.MODEL_SAVE_PATH}")
    else:
        logger.warning(
            f"Model weights not found at {cfg.MODEL_SAVE_PATH}. Using random weights (untrained)."
        )

    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # Load Test Metadata to map IDs
    test_df = pd.read_csv(cfg.TEST_CSV)

    if cfg.DEBUG:
        test_df = test_df.head(cfg.DEBUG_SAMPLE_SIZE)

    # Ensure lengths match
    if len(predictions) != len(test_df):
        logger.error(
            f"Mismatch: {len(predictions)} predictions vs {len(test_df)} test samples."
        )

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"image_id": test_df["image_id"], "label": predictions}
    )

    # Save
    os.makedirs(cfg.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(cfg.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {cfg.SUBMISSION_PATH}")

    return submission_df
