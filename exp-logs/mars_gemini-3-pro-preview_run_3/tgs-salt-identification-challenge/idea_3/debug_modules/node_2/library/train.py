import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from library.config import Config
from library.dataset import get_dataloaders, set_seed
from library.model import ResNeXtUNet
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.utils import AverageMeter, calc_map, iou_metric


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    losses = AverageMeter()
    ious = AverageMeter()

    for inputs, masks in loader:
        inputs = inputs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)

        # Calculate loss
        loss = criterion(logits, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        losses.update(loss.item(), inputs.size(0))

        # Calculate approximate IoU for monitoring (threshold 0.5)
        with torch.no_grad():
            preds_bin = (torch.sigmoid(logits) > 0.5).float()
            batch_iou = iou_metric(preds_bin.cpu().numpy(), masks.cpu().numpy())
            ious.update(batch_iou, inputs.size(0))

    return losses.avg, ious.avg


def validate(model, loader, criterion, device):
    """
    Validates the model. Returns average loss, mAP at 0.5, and raw predictions/targets.
    """
    model.eval()
    losses = AverageMeter()

    # Store all predictions and targets for global mAP calculation and threshold optimization
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, masks in loader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            logits = model(inputs)
            loss = criterion(logits, masks)

            losses.update(loss.item(), inputs.size(0))

            # Store probabilities
            preds_prob = torch.sigmoid(logits)
            all_preds.append(preds_prob.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate mAP at standard threshold 0.5
    map_score = calc_map(all_preds, all_targets, threshold=0.5)

    return losses.avg, map_score, all_preds, all_targets


def optimize_threshold(preds, targets):
    """
    Sweeps over thresholds to find the one that maximizes mAP.
    """
    thresholds = np.arange(0.2, 0.85, 0.05)
    best_threshold = 0.5
    best_map = 0.0

    for t in thresholds:
        score = calc_map(preds, targets, threshold=t)
        if score > best_map:
            best_map = score
            best_threshold = t

    return best_threshold, best_map


def run_training():
    """
    Main training loop.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Model
    model = ResNeXtUNet(n_classes=1, pretrained=True)
    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # Loss Functions
    criterion_bce_dice = BCEDiceLoss(alpha=0.5)
    criterion_lovasz = LovaszHingeLoss()

    best_map = 0.0
    best_epoch = 0
    early_stopping_patience = 10
    patience_counter = 0

    # To store the best threshold found during validation of the best model
    final_best_threshold = 0.5

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")
    print(f"Switching to Lovasz Loss at epoch {Config.LOVASZ_EPOCH_START}")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Select Loss Function
        if epoch < Config.LOVASZ_EPOCH_START:
            criterion = criterion_bce_dice
            loss_name = "BCE+Dice"
        else:
            criterion = criterion_lovasz
            loss_name = "Lovasz"

        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_map_05, val_preds, val_targets = validate(
            model, val_loader, criterion, device
        )

        # Optimize Threshold (on validation set)
        # We optimize threshold every epoch to track 'potential' best performance
        current_best_thresh, current_best_map = optimize_threshold(
            val_preds, val_targets
        )

        # Update Scheduler based on mAP (using the optimized one for better signal)
        scheduler.step(current_best_map)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{Config.EPOCHS} [{loss_name}] | "
            f"Time: {elapsed:.0f}s | "
            f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val mAP(0.5): {val_map_05} | "
            f"Opt mAP: {current_best_map} @ {current_best_thresh:.2f}"
        )

        # Save Best Model
        # We use the optimized mAP to determine the best model
        if current_best_map > best_map:
            best_map = current_best_map
            best_epoch = epoch
            final_best_threshold = current_best_thresh
            patience_counter = 0

            save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_map": best_map,
                    "best_threshold": final_best_threshold,
                },
                save_path,
            )
            # Also save to working root for easy access
            torch.save(
                model.state_dict(), os.path.join(Config.WORKING_DIR, "best_model.pth")
            )
            print(f"  >>> Model Saved! New Best mAP: {best_map}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(
        f"Training complete. Best mAP: {best_map} at epoch {best_epoch} with threshold {final_best_threshold:.2f}"
    )

    # Load best weights before returning
    best_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, final_best_threshold
