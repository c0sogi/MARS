import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset

from library.config import Config
from library.utils import seed_everything, get_label_encoder, calculate_map5
from library.dataset import HotelDataset, get_transforms
from library.model import HotelResNet


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images, labels)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    return running_loss / count if count > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run evaluation on.

    Returns:
        tuple: (Average Loss, MAP@5 Score)
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

            # Get top 5 predictions (indices)
            # outputs shape: (Batch, NumClasses)
            _, topk_indices = torch.topk(outputs, k=5, dim=1)

            all_preds.append(topk_indices.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    if count == 0:
        return 0.0, 0.0

    avg_loss = running_loss / count

    # Concatenate results for metric calculation
    all_preds = np.vstack(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate MAP@5
    # calculate_map5 expects predictions as list of lists/arrays and targets as list/array
    map5 = calculate_map5(all_preds, all_targets)

    return avg_loss, map5


def run_training(
    debug=Config.DEBUG,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
):
    """
    Orchestrates the training process.

    Args:
        debug (bool): If True, runs on a small subset of data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loaders.
        load_cached_data (bool): Whether to load processed data from cache.

    Returns:
        float: The best validation MAP@5 score achieved.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing training on device: {device}")

    # ---------------------------------------------------------
    # 1. Prepare Data
    # ---------------------------------------------------------
    # Get Label Encoder
    encoder = get_label_encoder(Config.TRAIN_CSV, load_cached_data=load_cached_data)
    num_classes = len(encoder.id_to_class)

    # Initialize Datasets
    train_dataset = HotelDataset(
        csv_path=Config.TRAIN_CSV,
        root_dir=Config.INPUT_DIR,
        label_encoder=encoder,
        transform=get_transforms("train"),
        is_test=False,
        load_cached_data=load_cached_data,
    )

    val_dataset = HotelDataset(
        csv_path=Config.VAL_CSV,
        root_dir=Config.INPUT_DIR,
        label_encoder=encoder,
        transform=get_transforms(
            "val"
        ),  # Using same transforms logic as defined in library
        is_test=False,
        load_cached_data=load_cached_data,
    )

    # Handle Debug Mode
    if debug:
        print(f"Debug mode enabled. Subsetting to {Config.DEBUG_SAMPLE_SIZE} samples.")
        train_indices = list(range(min(len(train_dataset), Config.DEBUG_SAMPLE_SIZE)))
        val_indices = list(range(min(len(val_dataset), Config.DEBUG_SAMPLE_SIZE)))

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")

    # ---------------------------------------------------------
    # 2. Prepare Model, Loss, Optimizer
    # ---------------------------------------------------------
    model = HotelResNet(num_classes=num_classes, pretrained=Config.PRETRAINED)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    best_map5 = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_map5 = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MAP@5: {val_map5}"
        )

        # Early Stopping & Checkpointing
        if val_map5 > best_map5:
            best_map5 = val_map5
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch + 1}. Best MAP@5: {best_map5}"
            )
            break

    print(f"Training complete. Best Validation MAP@5: {best_map5}")
    return best_map5
