import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.utils import set_seed, get_logger, save_checkpoint, load_checkpoint
from library.dataset import make_loader
from library.model import SPCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        angles = batch["angle"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, angles)
        loss = criterion(logits, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            labels = batch["label"].to(device)

            logits = model(images, angles)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    ids = []
    preds = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            batch_ids = batch["id"]

            logits = model(images, angles)
            probs = torch.sigmoid(logits)

            ids.extend(batch_ids)
            preds.extend(probs.cpu().numpy())

    return ids, preds


def run_training(
    epochs=75,
    patience=12,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    seed=42,
    work_dir="./working/idea_55",
):
    """
    Main function to run the training pipeline.
    """
    set_seed(seed)

    # Define directories
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    log_dir = os.path.join(work_dir, "logs")
    submission_dir = os.path.join(work_dir, "submission")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # Setup Logger
    logger = get_logger(os.path.join(log_dir, "train.log"))
    logger.info("Starting training pipeline...")

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Data
    # Note: make_loader handles caching internally in ./working/idea_55/
    logger.info("Initializing data loaders...")
    train_loader, val_loader, test_loader = make_loader(
        batch_size=batch_size, num_workers=2, load_cached_data=True
    )

    # Initialize Model
    logger.info("Initializing SPCNN model...")
    model = SPCNN().to(device)

    # Optimizer and Loss
    # Using AdamW with constant learning rate as per strategy
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    fold = 0  # Default fold index for the fixed split

    logger.info(f"Starting training for {epochs} epochs with patience {patience}...")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss:.10f}, Train Acc: {train_acc:.10f} - "
            f"Val Loss: {val_loss:.10f}, Val Acc: {val_acc:.10f}"
        )

        # Checkpoint and Early Stopping
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                is_best=True,
                checkpoint_dir=checkpoint_dir,
                fold=fold,
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with Val Loss {best_loss:.10f}"
            )
            break

    # Inference
    logger.info("Loading best model for inference...")
    best_model_path = os.path.join(checkpoint_dir, f"model_best_fold_{fold}.pth")

    # Load checkpoint
    try:
        load_checkpoint(best_model_path, model, device=device)
        logger.info(f"Successfully loaded model from {best_model_path}")
    except FileNotFoundError:
        logger.error(
            f"Best model not found at {best_model_path}. Using current model state."
        )

    logger.info("Generating predictions on test set...")
    ids, probs = predict(model, test_loader, device)

    # Save Submission
    sub_df = pd.DataFrame({"id": ids, "is_iceberg": probs})

    sub_path = os.path.join(submission_dir, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    logger.info(f"Submission saved to {sub_path}")
