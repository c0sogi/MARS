import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, compute_metrics
from library.dataset import PathologyDataset, get_transforms
from library.model import TumorClassifier


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training set.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    for images, labels in dataloader:
        images = images.to(device)
        # BCEWithLogitsLoss expects target shape (N, *) to match input shape.
        # Model output is (N, 1), so we unsqueeze labels to (N, 1).
        labels = labels.to(device).unsqueeze(1)

        # Cite {solution_lesson_node_00006}: Label Smoothing
        # Smooth labels: y_ls = y * (1 - epsilon) + 0.5 * epsilon
        if Config.LABEL_SMOOTHING > 0:
            labels_smooth = (
                labels * (1 - Config.LABEL_SMOOTHING) + 0.5 * Config.LABEL_SMOOTHING
            )
        else:
            labels_smooth = labels

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels_smooth)

        loss.backward()
        optimizer.step()

        # Accumulate batch loss scaled by batch size
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (Average validation loss, Validation AUC)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(dataloader.dataset)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)

            # Move to CPU and store for metric calculation
            all_preds.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    val_loss = running_loss / dataset_size
    val_auc = compute_metrics(all_labels, all_preds)

    return val_loss, val_auc


def run_training(
    epochs: int = Config.EPOCHS,
    batch_size: int = Config.BATCH_SIZE,
    learning_rate: float = Config.LEARNING_RATE,
    weight_decay: float = Config.WEIGHT_DECAY,
    patience: int = Config.EARLY_STOPPING_PATIENCE,
    num_workers: int = Config.NUM_WORKERS,
    debug: bool = Config.DEBUG,
):
    """
    Orchestrates the training process.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        patience (int): Number of epochs to wait for improvement before early stopping.
        num_workers (int): Number of subprocesses for data loading.
        debug (bool): If True, runs on a subset of data.

    Returns:
        str: Path to the saved best model checkpoint.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")

    # 2. Data Loading
    train_dataset = PathologyDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        transform=get_transforms("train"),
        debug=debug,
    )
    val_dataset = PathologyDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        transform=get_transforms("val"),
        debug=debug,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = TumorClassifier(
        pretrained=Config.PRETRAINED, dropout_rate=Config.DROPOUT_RATE
    )
    model = model.to(device)

    # 4. Optimizer, Criterion, Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # Cosine Annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Print Metrics (Full Precision)
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Checkpoint & Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation AUC improved. Saved model to {best_model_path}")
        else:
            patience_counter += 1
            print(
                f"No improvement in validation AUC. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")
    return best_model_path
