import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from library.config import Config
from library.utils import seed_everything, AverageMeter, metric_score, InverseScaler
from library.loss import LaplaceNLLLoss
from library.data import get_dataloaders
from library.model import RODSNet


def train_one_epoch(epoch, model, loader, criterion, optimizer, device):
    """
    Handles the training of a single epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (imgs, tabular, targets) in enumerate(loader):
        imgs = imgs.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(imgs, tabular)

        # Calculate loss
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), imgs.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    losses = AverageMeter()
    scores = AverageMeter()

    inverse_scaler = InverseScaler()

    # Get global stats for manual target inverse scaling
    # target_orig = target_scaled * std + mean
    mean_val, std_val = Config.get_target_stats()

    with torch.no_grad():
        for imgs, tabular, targets in loader:
            imgs = imgs.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)

            # Forward pass
            preds = model(imgs, tabular)
            loss = criterion(preds, targets)
            losses.update(loss.item(), imgs.size(0))

            # --- Metric Calculation ---
            # 1. Prepare Predictions
            pred_mean_norm = preds[:, 0]
            pred_raw_sigma = preds[:, 1]

            # Convert raw sigma to normalized sigma (softplus)
            # The loss function uses softplus + epsilon, we do the same for consistency
            pred_sigma_norm = F.softplus(pred_raw_sigma) + 1e-6

            # Inverse scale predictions to ml
            pred_mean_orig, pred_sigma_orig = inverse_scaler(
                pred_mean_norm, pred_sigma_norm
            )

            # 2. Prepare Targets
            # Targets from loader are Z-scored. Convert back to ml.
            targets_orig = targets * std_val + mean_val

            # 3. Calculate Score
            # Flatten tensors for metric calculation
            score = metric_score(
                targets_orig.view(-1), pred_mean_orig.view(-1), pred_sigma_orig.view(-1)
            )
            scores.update(score, imgs.size(0))

    return losses.avg, scores.avg


def run_training(debug=False):
    """
    Main training loop.
    """
    seed_everything(Config.SEED)

    # Data
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # Model
    device = torch.device(Config.DEVICE)
    model = RODSNet().to(device)

    # Optimizer with Differential Learning Rates
    # Group 1: Backbone parameters (frozen then unfrozen parts)
    backbone_params = list(model.visual_stream.backbone.parameters())
    backbone_ids = list(map(id, backbone_params))

    # Group 2: Head parameters (Clinical stream + Visual stream MLP/Projection)
    head_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # Loss
    criterion = LaplaceNLLLoss()

    print(f"Starting training on {device} for {Config.NUM_EPOCHS} epochs...")

    best_score = -float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, device
        )
        val_loss, val_score = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Score: {val_score}"
        )

        # Checkpoint
        if val_score > best_score:
            best_score = val_score
            save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with score: {best_score}")

    print(f"Training complete. Best Validation Score: {best_score}")
