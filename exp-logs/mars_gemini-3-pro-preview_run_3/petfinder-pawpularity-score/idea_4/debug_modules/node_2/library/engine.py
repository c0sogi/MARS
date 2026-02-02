import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import get_score
from library.models import AdaptiveBackbone
from library.ensemble import extract_features_and_cache


def train_one_epoch(model, dataloader, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training DataLoader.
        optimizer: Optimizer instance.
        device: Torch device.
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    dataset_size = 0
    criterion = nn.MSELoss()

    for batch in dataloader:
        images = batch["image"].to(device)
        # Targets are in range [1, 100], model output (Sigmoid) is [0, 1].
        # Normalize targets for training.
        targets = batch["target"].to(device) / 100.0
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)  # Shape: (B, 1)

        # Compute loss
        loss = criterion(outputs, targets.view(-1, 1))

        # Backward pass
        loss.backward()

        # Gradient Clipping
        if Config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item() * batch_size
        dataset_size += batch_size

    return total_loss / dataset_size


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation DataLoader.
        device: Torch device.

    Returns:
        tuple: (average_loss, rmse_score)
    """
    model.eval()
    total_loss = 0.0
    dataset_size = 0
    criterion = nn.MSELoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)  # Range [1, 100]
            batch_size = images.size(0)

            # Forward pass
            outputs = model(images)  # Range [0, 1]

            # Compute Loss (on normalized scale to be consistent with training)
            loss = criterion(outputs, targets.view(-1, 1) / 100.0)
            total_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Rescale predictions to [1, 100] for metric calculation
            preds = outputs.cpu().numpy() * 100.0
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    avg_loss = total_loss / dataset_size
    rmse = get_score(np.array(all_targets), np.array(all_preds))

    return avg_loss, rmse


def run_fine_tuning(
    train_loader,
    val_loader,
    epochs=Config.EPOCHS,
    learning_rate=Config.LEARNING_RATE,
):
    """
    Runs the Stage 1 Fine-Tuning process.

    Args:
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        epochs: Number of epochs to train.
        learning_rate: Learning rate for the optimizer.

    Returns:
        model: The fine-tuned model with the best validation weights loaded.
    """
    device = Config.DEVICE
    print(f"Starting fine-tuning on {device} for {epochs} epochs...")

    # Initialize Model
    model = AdaptiveBackbone(pretrained=True).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (Cosine Annealing)
    # T_max is the total number of steps
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_rmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scheduler)
        val_loss, val_rmse = validate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val RMSE: {val_rmse}"
        )

        # Save best model
        if val_rmse < best_rmse:
            print(f"RMSE improved from {best_rmse} to {val_rmse}. Saving model...")
            best_rmse = val_rmse
            torch.save(model.state_dict(), best_model_path)

    print(f"Fine-tuning complete. Best RMSE: {best_rmse}")
    print(f"Loading best weights from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model


def extract_features(model, dataloader, mode, tta=False, load_cached_data=True):
    """
    Extracts features using the fine-tuned model.
    Wraps the caching and TTA logic provided in library.ensemble.

    Args:
        model: The fine-tuned AdaptiveBackbone model.
        dataloader: DataLoader for the dataset.
        mode: 'train', 'valid', or 'test'.
        tta: Whether to apply Test-Time Augmentation.
        load_cached_data: Whether to try loading from cache first.

    Returns:
        tuple: (features, targets, ids)
    """
    return extract_features_and_cache(
        model=model,
        dataloader=dataloader,
        device=Config.DEVICE,
        mode=mode,
        tta=tta,
        load_cached_data=load_cached_data,
    )
