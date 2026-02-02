import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, get_device, save_checkpoint
from library.dataset import get_dataloaders
from library.model import get_model, freeze_backbone, unfreeze_all


def train_one_epoch(model, loader, optimizer, criterion, device, debug=False):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

        # Break early if debugging
        if debug and i >= 10:
            break

    epoch_loss = running_loss / count if count > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device, debug=False):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

            if debug and i >= 10:
                break

    epoch_loss = running_loss / count if count > 0 else 0.0
    return epoch_loss


def run_fold_training(
    fold_idx,
    warmup_epochs=Config.WARMUP_EPOCHS,
    finetune_epochs=Config.FINETUNE_EPOCHS,
    lr_finetune=Config.LR,
    weight_decay=Config.WEIGHT_DECAY,
    patience=5,
    debug=False,
):
    """
    Executes the training pipeline for a specific fold using the Two-Phase Transfer Learning strategy.

    Args:
        fold_idx (int): The fold index (0 to N_FOLDS-1).
        warmup_epochs (int): Number of epochs to train only the head.
        finetune_epochs (int): Number of epochs to fine-tune the whole model.
        lr_finetune (float): Learning rate for the fine-tuning phase.
        weight_decay (float): Weight decay for the optimizer.
        patience (int): Early stopping patience.
        debug (bool): If True, runs a short loop for debugging.

    Returns:
        float: The best validation loss achieved.
    """
    print(f"Starting training for Fold {fold_idx}")
    set_seed(Config.SEED)
    device = get_device()

    # 1. Prepare Data
    train_loader, val_loader, classes = get_dataloaders(fold_idx)

    # 2. Initialize Model
    model = get_model(Config.MODEL_NAME, Config.NUM_CLASSES, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # ==========================
    # Phase 1: Warm-up
    # ==========================
    if warmup_epochs > 0:
        print(f"Phase 1: Warm-up for {warmup_epochs} epochs...")
        freeze_backbone(model)

        # Use a standard learning rate for head initialization (e.g., 1e-3)
        optimizer_warmup = optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=weight_decay
        )

        for epoch in range(warmup_epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer_warmup, criterion, device, debug
            )
            val_loss = validate(model, val_loader, criterion, device, debug)
            print(
                f"[Warmup Epoch {epoch+1}/{warmup_epochs}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

    # ==========================
    # Phase 2: Fine-tuning
    # ==========================
    print(f"Phase 2: Fine-tuning for {finetune_epochs} epochs...")
    unfreeze_all(model)

    # Re-initialize optimizer for all parameters with the conservative fine-tuning LR
    optimizer = optim.AdamW(
        model.parameters(), lr=lr_finetune, weight_decay=weight_decay
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)

    best_val_loss = float("inf")
    early_stop_counter = 0

    save_path = os.path.join(Config.OUTPUT_DIR, f"model_fold_{fold_idx}.pth")

    for epoch in range(finetune_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, debug
        )
        val_loss = validate(model, val_loader, criterion, device, debug)

        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"[Finetune Epoch {epoch+1}/{finetune_epochs}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            print(
                f"Validation Loss improved from {best_val_loss:.6f} to {val_loss:.6f}. Saving model..."
            )
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(
                f"No improvement. Early stopping counter: {early_stop_counter}/{patience}"
            )

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Fold {fold_idx} training complete. Best Val Loss: {best_val_loss:.6f}")
    return best_val_loss
