import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import seed_everything, calculate_map5
from library.dataset import get_dataloaders
from library.model import WhaleDenseNet


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # ElasticFace head requires labels during training to apply margin
        logits = model(images, labels)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set using Test-Time Augmentation (TTA).
    TTA Strategy: Average logits of Original + Horizontal Flip.

    Args:
        model: The PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        tuple: (Average Validation Loss, MAP@5 Score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # 1. Forward pass - Original
            # Pass labels=None to ensure inference mode (no margin applied, just scaled cosine)
            logits_orig = model(images, labels=None)

            # 2. Forward pass - Horizontal Flip TTA
            # Flip along width dimension (dim 3 for NCHW tensor)
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip, labels=None)

            # 3. Average Logits
            logits_avg = (logits_orig + logits_flip) / 2.0

            # Compute Loss using averaged logits for monitoring
            loss = criterion(logits_avg, labels)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Get Top 5 Predictions
            # logits_avg: (B, NumClasses)
            _, top_indices = torch.topk(logits_avg, k=5, dim=1)

            # Store predictions and targets
            # Convert to CPU lists
            all_preds.extend(top_indices.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    val_loss = running_loss / dataset_size

    # Calculate MAP@5
    # calculate_map5 accepts lists of integers for both preds and targets
    map5 = calculate_map5(all_preds, all_targets)

    return val_loss, map5


def fit_model(seed_val):
    """
    Full training loop for a single model in the Independent Convergence Ensemble (ICE).

    Args:
        seed_val (int): The random seed for this specific ensemble member.

    Returns:
        float: The best MAP@5 score achieved.
    """
    # 1. Setup Environment
    print(f"Starting training for Seed: {seed_val}")

    # Set seed for reproducibility
    seed_everything(seed_val)

    # Create specific output directory for this seed
    save_dir = os.path.join(Config.WORKING_DIR, f"seed_{seed_val}")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # load_cached_data=True allows reusing the class mapping generated previously
    train_loader, val_loader, _, class_to_idx, _ = get_dataloaders(
        load_cached_data=True, verbose=True
    )

    # 3. Model Initialization
    # Initialize model with correct number of classes
    model = WhaleDenseNet(
        num_classes=len(class_to_idx),
        embedding_dim=Config.EMBEDDING_DIM,
        pretrained=True,
    )
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.MIN_LR
    )

    # 5. Loss Function
    # CrossEntropyLoss with Label Smoothing as per strategy
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    # 6. Training Loop
    best_map5 = 0.0
    patience_counter = 0

    for epoch in range(1, Config.MAX_EPOCHS + 1):
        start_time = time.time()

        # Train Step
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation Step (with TTA)
        val_loss, val_map5 = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch}/{Config.MAX_EPOCHS} "
            f"[Time: {elapsed:.2f}s] "
            f"[LR: {current_lr:.6f}] "
            f"Train Loss: {train_loss:.8f} "
            f"Val Loss: {val_loss:.8f} "
            f"Val MAP@5: {val_map5:.16f}"
        )

        # 7. Checkpointing & Early Stopping
        # Save latest checkpoint
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_map5": best_map5,
            "class_to_idx": class_to_idx,
        }
        torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth.tar"))

        # Save best model
        if val_map5 > best_map5:
            best_map5 = val_map5
            patience_counter = 0
            torch.save(checkpoint, os.path.join(save_dir, "model_best.pth.tar"))
            print(f"  >>> New Best MAP@5: {best_map5:.16f}. Saved model.")
        else:
            patience_counter += 1
            print(
                f"  >>> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"Training finished for Seed {seed_val}. Best MAP@5: {best_map5:.16f}")
    return best_map5
