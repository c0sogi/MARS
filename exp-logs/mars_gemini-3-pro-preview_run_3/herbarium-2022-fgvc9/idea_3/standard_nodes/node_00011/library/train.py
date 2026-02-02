import os
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import set_seed, AverageMeter, Mixup
from library.dataset import get_dataloader
from library.model import PlantConvNeXt


def train_one_epoch(
    epoch, model, loader, optimizer, criterion, device, scaler, scheduler, mixup_fn
):
    """
    Trains the model for one epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply Mixup/CutMix
        # This returns mixed images and soft targets (probabilities)
        images, targets = mixup_fn(images, targets)

        with autocast(enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(enabled=Config.USE_AMP):
                outputs = model(images)
                # Validation targets are class indices (LongTensor)
                loss = criterion(outputs, targets)

            # Get predictions
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            loss_meter.update(loss.item(), images.size(0))

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Macro F1 Score
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    return loss_meter.avg, macro_f1


def run_training(
    debug=Config.DEBUG,
    epochs=Config.EPOCHS,
    patience=5,
    save_path=Config.BEST_MODEL_PATH,
):
    """
    Main function to run the training pipeline.

    Args:
        debug (bool): Whether to run in debug mode (fewer samples).
        epochs (int): Number of training epochs.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 2. Data Loaders
    print("Initializing DataLoaders...")
    train_loader = get_dataloader("train", debug=debug)
    val_loader = get_dataloader("val", debug=debug)

    # 3. Model
    print(f"Initializing Model: {Config.MODEL_NAME}")
    model = PlantConvNeXt()
    model = model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR requires total steps
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 5. Loss Function
    # CrossEntropyLoss supports both class indices (Validation) and soft probabilities (Train w/ Mixup)
    criterion = nn.CrossEntropyLoss()

    # 6. AMP Scaler & Mixup
    scaler = GradScaler(enabled=Config.USE_AMP)
    mixup_fn = Mixup()

    # 7. Training Loop
    best_f1 = -1.0
    early_stop_counter = 0

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            scheduler,
            mixup_fn,
        )

        # Validate
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch}/{epochs} completed in {elapsed} seconds.")
        print("Train Loss:", train_loss)
        print("Val Loss:", val_loss)
        print("Val Macro F1:", val_f1)

        # Checkpoint & Early Stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    print("Training finished.")
    print("Best Validation Macro F1:", best_f1)
