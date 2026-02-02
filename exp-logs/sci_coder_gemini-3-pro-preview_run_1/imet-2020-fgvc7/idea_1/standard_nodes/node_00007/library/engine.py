import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import calculate_micro_f1
from library.model import ArtworkResNet


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer instance.
        criterion (Loss): Loss function.
        device (str): Device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (Loss): Loss function.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (average_loss, micro_f1_score)
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            num_batches += 1

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        f1_score = calculate_micro_f1(
            all_preds, all_targets, threshold=Config.THRESHOLD
        )
    else:
        f1_score = 0.0

    return avg_loss, f1_score


def run_training(train_loader, val_loader):
    """
    Main training loop with early stopping and model checkpointing.

    Args:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
    """
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Initialize model
    model = ArtworkResNet(
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
        freeze_backbone=Config.FREEZE_BACKBONE,
    )
    model = model.to(device)

    # Define Loss Function with Positive Weight
    # Create a vector of weights for each class
    pos_weight = torch.full((Config.NUM_CLASSES,), Config.POS_WEIGHT).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()

        # Print full precision metrics
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Micro F1: {val_f1}")

        # Checkpointing based on F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"Validation F1 improved. Saving model to {Config.MODEL_SAVE_PATH}")
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement in F1. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training completed. Best Val F1: {best_f1}")


def inference(test_loader):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        test_loader (DataLoader): Test data loader.
    """
    device = Config.DEVICE

    # Load model structure
    model = ArtworkResNet(num_classes=Config.NUM_CLASSES, pretrained=False)

    # Load best weights
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Train the model first."
        )

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    results = []

    print("Starting inference on test set...")

    with torch.no_grad():
        for images, img_ids in test_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Convert to numpy
            probs_np = probs.cpu().numpy()

            # Process batch
            for i in range(len(img_ids)):
                img_id = img_ids[i]
                sample_probs = probs_np[i]

                # Get indices where probability > threshold
                predicted_indices = np.where(sample_probs > Config.THRESHOLD)[0]

                # Format as space-separated string
                if len(predicted_indices) > 0:
                    pred_str = " ".join(map(str, predicted_indices))
                else:
                    # If no attribute exceeds threshold, leave empty or handle as needed.
                    # Competition usually expects empty string or specific default.
                    # Based on sample submission, it's just IDs.
                    pred_str = ""

                results.append({"id": img_id, "attribute_ids": pred_str})

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
