import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from library.config import Config
from library.dataset import CervicalSpineDataset
from library.model import CervicalSpineMIL
from library.loss import HierarchicalCompoundLoss


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)

        # Prepare targets dictionary as expected by HierarchicalCompoundLoss
        targets = {
            "vertebrae": batch["labels"]["vertebrae"].to(device),
            "patient_overall": batch["labels"]["patient_overall"].to(device),
        }

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)

            targets = {
                "vertebrae": batch["labels"]["vertebrae"].to(device),
                "patient_overall": batch["labels"]["patient_overall"].to(device),
            }

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

    return running_loss / len(loader)


def run_training(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=5,
    load_cached_data=True,
):
    """
    Main driver for the training pipeline with Early Stopping.

    Args:
        num_epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        learning_rate (float): Initial learning rate.
        patience (int): Number of epochs to wait for improvement before early stopping.
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    # 1. Setup
    Config.setup_reproducibility()
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    train_dataset = CervicalSpineDataset(
        Config.TRAIN_METADATA_PATH, phase="train", load_cached_data=load_cached_data
    )
    val_dataset = CervicalSpineDataset(
        Config.VAL_METADATA_PATH, phase="val", load_cached_data=load_cached_data
    )

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

    # 3. Model, Optimizer, Scheduler, Loss
    model = CervicalSpineMIL(pretrained=Config.PRETRAINED).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    # T_max is calculated based on the total expected epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(num_epochs * Config.T_MAX_MULTIPLIER)
    )

    criterion = HierarchicalCompoundLoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Print full precision metrics
        print(f"Epoch {epoch+1}: Train Loss {train_loss}, Val Loss {val_loss}")

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Validation loss improved. Saved model to {Config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered. No improvement for {patience} epochs."
                )
                break

    print(f"Training completed. Best Validation Loss: {best_val_loss}")
