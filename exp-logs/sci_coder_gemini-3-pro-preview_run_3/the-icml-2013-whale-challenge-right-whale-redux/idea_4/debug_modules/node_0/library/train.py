import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    calculate_auc,
)
from library.data import get_dataloaders
from library.model import WhaleClassifier
from library.losses import get_loss_module, MixupLoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        # Mixup Augmentation
        if Config.MIXUP_ALPHA > 0:
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
        else:
            lam = 1.0

        # Create mixed inputs
        index = torch.randperm(batch_size).to(device)
        mixed_images = lam * images + (1 - lam) * images[index, :]

        # Targets for mixup loss
        y_a, y_b = targets, targets[index]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(mixed_images)

        # Compute Loss
        # criterion is expected to be MixupLoss here
        loss = criterion(outputs, y_a, y_b, lam)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            outputs = model(images)

            # Compute Loss (Standard WeightedBCE, not Mixup)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    auc_score = calculate_auc(all_targets, all_preds)

    return avg_loss, auc_score


def predict_test_set(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    clips = []
    probs = []

    with torch.no_grad():
        for images, batch_clips in loader:
            images = images.to(device)

            outputs = model(images)
            batch_probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            clips.extend(batch_clips)
            probs.extend(batch_probs)

    return clips, probs


def run_training(epochs=Config.EPOCHS, load_cached_data=True, patience=5):
    """
    Main execution function for training and inference.

    Args:
        epochs (int): Number of training epochs.
        load_cached_data (bool): Whether to use cached preprocessed data.
        patience (int): Early stopping patience.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    print(f"Initializing model: {Config.BACKBONE}")
    model = WhaleClassifier()
    model = model.to(device)

    # 4. Loss Functions
    # Base criterion for validation (Weighted BCE)
    base_criterion = get_loss_module().to(device)
    # Mixup criterion for training
    train_criterion = MixupLoss(base_criterion).to(device)

    # 5. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.MIN_LR)

    # 6. Training Loop
    best_score = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, train_criterion, device
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, base_criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging (Full precision)
        print(
            f"Epoch {epoch}/{epochs} | LR: {current_lr:.8f} | "
            f"Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Save Best Model
        if val_auc > best_score:
            print(
                f"Validation AUC improved from {best_score:.10f} to {val_auc:.10f}. Saving checkpoint..."
            )
            best_score = val_auc
            save_checkpoint(model, optimizer, epoch, best_score, Config.BEST_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    # 7. Inference
    print("Training complete. Loading best model for inference...")

    # Reload best model weights
    checkpoint = load_checkpoint(Config.BEST_MODEL_PATH, model, device=device)
    print(
        f"Loaded checkpoint from epoch {checkpoint['epoch']} with AUC {checkpoint['score']:.10f}"
    )

    print("Generating predictions on test set...")
    test_clips, test_probs = predict_test_set(model, test_loader, device)

    # 8. Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df = pd.DataFrame({"clip": test_clips, "probability": test_probs})

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
