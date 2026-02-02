import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm.auto import tqdm

from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import get_loaders
from library.model import AppleDiseaseModel


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """
    Trains the model for one epoch using Gradient Accumulation.
    """
    model.train()

    dataset_size = 0
    running_loss = 0.0

    # Gradient Accumulation setup
    accum_steps = Config.GRADIENT_ACCUM_STEPS
    optimizer.zero_grad()

    # Iterate over batches
    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        batch_size = images.size(0)

        # Forward pass
        logits = model(images)
        loss = model.get_loss(logits, targets)

        # Normalize loss for gradient accumulation
        loss = loss / accum_steps
        loss.backward()

        # Gradient Clipping
        if Config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Step optimizer and zero gradients only after accum_steps
        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += (loss.item() * accum_steps) * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size

    return epoch_loss


def valid_one_epoch(model, dataloader, device):
    """
    Validates the model on the validation set.
    """
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)

            batch_size = images.size(0)

            logits = model(images)
            loss = model.get_loss(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate F1 Score
    epoch_f1 = get_score(all_targets, all_preds, threshold=Config.THRESHOLD)

    return epoch_loss, epoch_f1


def train_loop():
    """
    Main training loop orchestrator.
    """
    seed_everything(Config.SEED)

    # Directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Data Loaders
    train_loader, val_loader, _ = get_loaders()

    # Model
    device = torch.device(Config.DEVICE)
    model = AppleDiseaseModel()
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    # Using CosineAnnealingLR.
    # T_max is set to EPOCHS.
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # Training Loop variables
    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0.0
    patience = 5
    patience_counter = 0

    print(f"Starting training on device: {device}")
    print(
        f"Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}, Accum Steps: {Config.GRADIENT_ACCUM_STEPS}"
    )

    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch
        )

        # Validate
        val_loss, val_f1 = valid_one_epoch(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Print metrics
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val F1 Score: {val_f1}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']}")

        # Checkpointing
        if val_f1 > best_score:
            print(
                f"Validation Score Improved ({best_score} ---> {val_f1}). Saving model..."
            )
            best_score = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0

            # Save best model to disk
            save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Val F1 Score: {best_score}")

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model
