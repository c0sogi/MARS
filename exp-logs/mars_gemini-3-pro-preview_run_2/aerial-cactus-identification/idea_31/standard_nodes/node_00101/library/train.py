import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, calculate_roc_auc, ModelCheckpoint, print_metrics
from library.dataset import CactusDataset, get_transforms
from library.model import WideSERes2NeXt


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to train on.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to evaluate on.

    Returns:
        tuple: (average validation loss, validation ROC AUC score)
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to convert logits to probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu())
            all_preds.append(probs.cpu())

    val_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    val_auc = calculate_roc_auc(all_labels, all_preds)

    return val_loss, val_auc


def run_training(seed, max_epochs=Config.EPOCHS, patience=5):
    """
    Executes the full training pipeline for a specific seed.

    Args:
        seed (int): The random seed for this run.
        max_epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for improvement before early stopping.
    """
    # 1. Set Reproducibility
    set_seed(seed)

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Initialize Datasets and Loaders
    train_dataset = CactusDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        phase="train",
        transform=get_transforms("train"),
        max_samples=Config.MAX_TRAIN_SAMPLES,
    )

    val_dataset = CactusDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        phase="val",
        transform=get_transforms("val"),
        max_samples=Config.MAX_TRAIN_SAMPLES,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Helps with BatchNorm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Initialize Model
    model = WideSERes2NeXt()
    model = model.to(device)

    # 5. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=Config.ETA_MIN
    )

    # 6. Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # 7. Checkpointing and Tracking
    checkpoint = ModelCheckpoint(seed=seed, mode="max")
    patience_counter = 0

    print(f"Starting training for Seed {seed} on {device}")

    for epoch in range(1, max_epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Log Metrics
        print_metrics(epoch, train_loss, val_loss, val_auc)

        # Checkpoint Logic & Early Stopping
        is_best = checkpoint.step(val_auc, model)

        if is_best:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} for seed {seed}")
            break

    print(f"Training finished for Seed {seed}. Best Val AUC: {checkpoint.best_score}")
