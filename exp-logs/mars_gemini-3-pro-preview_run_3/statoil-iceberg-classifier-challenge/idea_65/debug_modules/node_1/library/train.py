import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, get_device
from library.model import CMSDI_CNN
from library.data import get_data_loaders


def train_epoch(model, loader, optimizer, device):
    """
    Trains the model for one epoch.
    Handles the Multi-Sample Dropout logic where the model returns (Batch, Num_Samples).
    """
    model.train()
    running_loss = 0.0

    criterion = nn.BCEWithLogitsLoss()

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # In training mode, model returns [Batch, Num_Samples]
        logits_stack = model(images, angles)

        # Expand labels to match the shape of logits_stack for multi-sample loss
        # labels: [Batch] -> [Batch, 1] -> [Batch, Num_Samples]
        labels_expanded = labels.unsqueeze(1).expand_as(logits_stack)

        # Calculate loss
        loss = criterion(logits_stack, labels_expanded)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    In eval mode, the model returns averaged logits (Batch, 1).
    """
    model.eval()
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            # Forward pass
            # In eval mode, model returns averaged logits [Batch, 1]
            preds = model(images, angles)

            # Calculate loss against standard labels [Batch, 1]
            loss = criterion(preds, labels.unsqueeze(1))

            running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def run_fold(fold, device):
    """
    Runs the training and validation loop for a single fold.
    Implements Early Stopping and Checkpointing.
    """
    print(f"\nStarting Fold {fold}/{Config.NUM_FOLDS - 1}")

    # Data Loaders
    train_loader, val_loader = get_data_loaders(fold)

    # Model
    model = CMSDI_CNN().to(device)

    # Optimizer (AdamW with constant LR as per Idea)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Early Stopping variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, device)

        elapsed = time.time() - start_time

        print(
            f"Fold {fold} | Epoch {epoch + 1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved to {best_model_path}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch + 1}. Best Val Loss: {best_val_loss:.8f}"
            )
            break

    return best_val_loss


def train_models():
    """
    Main function to train models across all folds.
    """
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    fold_scores = []

    for fold in range(Config.NUM_FOLDS):
        if Config.DEBUG and fold > 0:
            print("Debug mode: Skipping remaining folds.")
            break

        best_loss = run_fold(fold, device)
        fold_scores.append(best_loss)

    print("\nTraining Complete.")
    print("Cross-Validation Scores (Log Loss):")
    for i, score in enumerate(fold_scores):
        print(f"Fold {i}: {score:.8f}")
    print(f"Average CV Score: {np.mean(fold_scores):.8f}")
