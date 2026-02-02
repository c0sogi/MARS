import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from library.config import Config, seed_everything
from library.utils import (
    AverageMeter,
    calculate_roc_auc,
    save_checkpoint,
    save_submission,
)
from library.model import CactusResNet
from library.dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for images, targets in loader:
        images = images.to(device)
        # Reshape targets to (B, 1) for BCEWithLogitsLoss
        targets = targets.to(device).unsqueeze(1)

        # Apply Mixup if enabled
        if Config.MIXUP_ALPHA > 0:
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
            index = torch.randperm(images.size(0)).to(device)
            images = lam * images + (1 - lam) * images[index]
            targets = lam * targets + (1 - lam) * targets[index]

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets_dev = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets_dev)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to convert logits to probabilities
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    # Calculate AUC
    auc = calculate_roc_auc(np.array(all_targets), np.array(all_preds))
    return losses.avg, auc


def inference_tta(model, loader, device):
    """
    Performs inference with Test-Time Augmentation (TTA).
    Averages predictions from:
    1. Original Image
    2. Horizontally Flipped Image
    3. Vertically Flipped Image
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Original
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (flip width dimension, dim 3)
            images_h = torch.flip(images, [3])
            out_h = model(images_h)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip (flip height dimension, dim 2)
            images_v = torch.flip(images, [2])
            out_v = model(images_v)
            prob_v = torch.sigmoid(out_v)

            # Average probabilities
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0
            all_preds.extend(avg_prob.cpu().numpy().flatten())

    return all_preds


def run_training(debug=Config.DEBUG, save_checkpoint_path=None):
    """
    Main function to orchestrate training, validation, and submission generation.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Define checkpoint path
    if save_checkpoint_path is None:
        save_checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Load Data
    dataloaders = get_dataloaders(debug=debug)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # Initialize Model
    model = CactusResNet(num_classes=Config.NUM_CLASSES).to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN)

    # Training Loop Variables
    best_auc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Log metrics (Full precision for AUC)
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save the best model
            save_checkpoint(model, save_checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Val AUC: {best_auc}")

    # Load best weights for inference
    model.load_state_dict(best_model_wts)

    # Generate Submission
    print("Generating predictions on test set with TTA...")
    predictions = inference_tta(model, test_loader, device)

    # Retrieve IDs from the dataset dataframe
    test_ids = test_loader.dataset.df["id"].values

    # Save submission
    save_submission(test_ids, predictions, Config.SUBMISSION_PATH)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return model
