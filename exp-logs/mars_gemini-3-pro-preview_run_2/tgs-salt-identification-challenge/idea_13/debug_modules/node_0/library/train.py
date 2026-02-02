import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from library.config import Config
from library.dataset import get_dataloaders
from library.model import SaltNet
from library.losses import BCELovaszLoss
from library.utils import do_kaggle_metric


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    running_score = 0.0
    dataset_size = 0

    for images, masks, depths, _ in loader:
        batch_size = images.size(0)

        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Model expects (x, z)
        outputs = model(images, depths)

        # Loss calculation (BCELovaszLoss expects logits)
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metric calculation
        # Apply sigmoid to logits for metric calculation (0.5 threshold)
        preds = torch.sigmoid(outputs)
        score = do_kaggle_metric(preds, masks)

        running_loss += loss.item() * batch_size
        running_score += score * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    epoch_score = running_score / dataset_size

    return epoch_loss, epoch_score


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    running_score = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, masks, depths, _ in loader:
            batch_size = images.size(0)

            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            # Forward pass
            outputs = model(images, depths)

            # Loss
            loss = criterion(outputs, masks)

            # Metric
            preds = torch.sigmoid(outputs)
            score = do_kaggle_metric(preds, masks)

            running_loss += loss.item() * batch_size
            running_score += score * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    epoch_score = running_score / dataset_size

    return epoch_loss, epoch_score


def train_model(load_cached_data=True):
    """
    Main function to train the Residual-Injection Wide-LinkNet.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Training on device: {device}")

    # Ensure directories exist
    Config.setup()

    # 2. Data
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Model
    model = SaltNet()
    model = model.to(device)

    # 4. Optimization
    criterion = BCELovaszLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=1e-6
    )

    # 5. Training Loop
    best_score = -np.inf
    best_loss = np.inf

    # Early Stopping parameters
    patience = 15
    patience_counter = 0

    start_time = time.time()

    print(f"Starting training for {Config.EPOCHS} epochs...")
    print("-" * 60)

    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss, train_score = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        duration = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {duration:.0f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train mAP: {train_score:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val mAP: {val_score}"
        )

        # Checkpointing (Save Best Model based on mAP)
        if val_score > best_score:
            print(
                f"Validation mAP improved from {best_score:.4f} to {val_score:.4f}. Saving model..."
            )
            best_score = val_score
            best_loss = val_loss
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in mAP. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Validation mAP: {best_score}")
    print(f"Best Model Saved to: {Config.BEST_MODEL_PATH}")

    return model
