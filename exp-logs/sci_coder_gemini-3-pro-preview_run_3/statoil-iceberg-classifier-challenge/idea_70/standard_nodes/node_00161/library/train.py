import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.model import AGICNN
from library.dataset import get_fold_datasets


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run on.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, angles, labels in train_loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass: Model expects (x, angle)
        outputs = model(images, angles)

        # BCEWithLogitsLoss expects float targets
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Accuracy
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run on.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, angles, labels in val_loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = correct / total
    return val_loss, val_acc


def run_fold(fold, X, y, angles, ids):
    """
    Runs the full training and validation loop for a specific fold.

    Args:
        fold (int): The fold index (0-4).
        X (np.ndarray): Full image data array.
        y (np.ndarray): Full label array.
        angles (np.ndarray): Full angle array.
        ids (np.ndarray): Full ID array.

    Returns:
        float: The best validation loss achieved.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fold {fold}: Starting training on device {device}")

    # Prepare datasets for this fold (stratified split + leak-free imputation)
    train_ds, val_ds = get_fold_datasets(
        X, y, angles, ids, fold=fold, num_folds=Config.NUM_FOLDS, seed=Config.SEED
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize Model
    model = AGICNN()
    model.to(device)

    # Optimizer: AdamW with constant learning rate
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function: Binary Cross Entropy with Logits
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop State
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Fold {fold} | Epoch {epoch + 1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
            f"Val Loss: {val_loss} | Val Acc: {val_acc} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint (saves best model as a copy automatically)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "fold": fold,
            },
            is_best=is_best,
            fold=fold,
        )

        if patience_counter >= Config.PATIENCE:
            print(f"Fold {fold}: Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"Fold {fold}: Finished. Best Validation Loss: {best_val_loss}")
    return best_val_loss
