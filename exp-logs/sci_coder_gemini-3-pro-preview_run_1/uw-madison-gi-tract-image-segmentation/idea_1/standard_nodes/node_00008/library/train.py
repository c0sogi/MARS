import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.utils import set_seed
from library.dataset import UWMadisonDataset
from library.model import UNetResNet18, BCEDiceLoss, train_one_epoch, validate


def run_training(
    epochs=15, batch_size=32, fraction=1.0, lr=1e-4, patience=5, img_size=256
):
    """
    Executes the training pipeline for the U-Net ResNet18 model.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for dataloaders.
        fraction (float): Fraction of the dataset to use (for debugging).
        lr (float): Learning rate for the optimizer.
        patience (int): Number of epochs to wait for improvement before early stopping.
        img_size (int): Target image size for resizing.
    """
    # Ensure reproducibility
    set_seed(42)

    # Setup directories and device
    working_dir = "./working/idea_1"
    os.makedirs(working_dir, exist_ok=True)
    checkpoint_path = os.path.join(working_dir, "best_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # Initialize Datasets
    # The dataset class handles metadata loading and caching internally
    train_dataset = UWMadisonDataset(mode="train", fraction=fraction, img_size=img_size)
    val_dataset = UWMadisonDataset(mode="val", fraction=fraction, img_size=img_size)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize Model, Loss, Optimizer, and Scheduler
    model = UNetResNet18(num_classes=3).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # Training Loop Variables
    best_dice = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        start_time = time.time()

        # Execute Training Step
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Execute Validation Step
        val_loss, val_dice, val_hd = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_dice)

        duration = time.time() - start_time

        # Print metrics with full precision (no formatting)
        print(f"Epoch {epoch+1} completed in {duration} seconds")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Dice: {val_dice}")
        print(f"Val Hausdorff: {val_hd}")

        # Checkpointing
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early Stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training finished. Best Validation Dice: {best_dice}")
