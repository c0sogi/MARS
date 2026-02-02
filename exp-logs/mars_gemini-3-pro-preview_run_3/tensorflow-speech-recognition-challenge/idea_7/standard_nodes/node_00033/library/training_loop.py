import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, setup_logger
from library.dataset import get_dataloaders, get_test_dataloader
from library.network import MR_SK_CRNN


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch with optional MixUp.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    use_mixup = hasattr(Config, "MIXUP_ALPHA") and Config.MIXUP_ALPHA > 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_mixup:
            # MixUp
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
            index = torch.randperm(inputs.size(0)).to(device)

            mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
            labels_a, labels_b = labels, labels[index]

            outputs = model(mixed_inputs)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(
                outputs, labels_b
            )
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Accuracy calculation (using the stronger label for mixed samples or standard for normal)
        _, preds = torch.max(outputs, 1)
        if use_mixup:
            # For accuracy monitoring during MixUp, we compare against the original labels
            # This is an approximation but sufficient for monitoring
            correct_predictions += (
                lam * (preds == labels_a).float()
                + (1 - lam) * (preds == labels_b).float()
            ).sum()
        else:
            correct_predictions += torch.sum(preds == labels.data)

        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = (
        correct_predictions.double() / total_samples
        if isinstance(correct_predictions, torch.Tensor)
        else correct_predictions / total_samples
    )

    return epoch_loss, (
        epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc
    )


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples

    return epoch_loss, epoch_acc.item()


def train_model(debug=False):
    """
    Main function to handle the training pipeline including early stopping.
    """
    # Setup
    Config.setup_directories()
    logger = setup_logger("training", os.path.join(Config.WORKING_DIR, "training.log"))
    set_seed(Config.SEED)
    device = Config.DEVICE

    logger.info(f"Using device: {device}")
    logger.info("Initializing DataLoaders...")

    # Load Data
    train_loader, val_loader = get_dataloaders(debug=debug, load_cached_data=True)

    # Initialize Model
    logger.info("Initializing Model (MR_SK_CRNN)...")
    model = MR_SK_CRNN().to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Training Loop Variables
    best_acc = 0.0
    patience_counter = 0

    logger.info("Starting training loop...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log Metrics
        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            logger.info(
                f"New best validation accuracy: {best_acc:.6f}. Saving model..."
            )
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

            if patience_counter >= Config.PATIENCE:
                logger.info("Early stopping triggered.")
                break

    logger.info(f"Training complete. Best Validation Accuracy: {best_acc:.6f}")


def predict_submission(debug=False):
    """
    Generates predictions for the test set using the best saved model.
    """
    logger = setup_logger(
        "inference", os.path.join(Config.WORKING_DIR, "inference.log")
    )
    device = Config.DEVICE

    logger.info("Loading best model for inference...")
    model = MR_SK_CRNN().to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    logger.info("Loading Test DataLoader...")
    test_loader = get_test_dataloader(debug=debug, load_cached_data=True)

    predictions = []

    logger.info("Generating predictions...")
    with torch.no_grad():
        for inputs, fnames in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get predicted class indices
            _, preds = torch.max(outputs, 1)

            # Convert to CPU list
            preds_list = preds.cpu().numpy()

            for fname, pred_idx in zip(fnames, preds_list):
                label_str = Config.ID2LABEL[pred_idx]
                predictions.append({"fname": fname, "label": label_str})

    # Create DataFrame
    df_submission = pd.DataFrame(predictions)

    # Save to CSV
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(f"Total predictions generated: {len(df_submission)}")
