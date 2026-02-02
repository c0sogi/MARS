import os
import time
import torch
import torch.nn as nn
import numpy as np
from library import config, utils, models


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = utils.AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device).unsqueeze(1)  # Ensure target shape is (B, 1)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), data.size(0))

    return losses.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns: avg_loss, auc, predictions, targets
    """
    model.eval()
    losses = utils.AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device).unsqueeze(1)

            output = model(data)
            loss = criterion(output, target)
            losses.update(loss.item(), data.size(0))

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(output)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(target.cpu().numpy())

    # Concatenate results
    if len(preds_list) > 0:
        preds = np.concatenate(preds_list)
        targets = np.concatenate(targets_list)
    else:
        preds = np.array([])
        targets = np.array([])

    # Calculate AUC
    auc = utils.calculate_roc_auc(targets, preds)

    return losses.avg, auc, preds, targets


def train_fold(fold_idx, model_name, train_loader, val_loader, device, num_epochs=None):
    """
    Trains a single fold for a specific model architecture.
    Implements Multi-Objective Checkpointing: saves separate models for Best AUC and Best Loss.
    """
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS

    print(f"\n[Fold {fold_idx}] Initializing {model_name}...")

    # Initialize Model
    model = models.get_model(model_name, pretrained=config.PRETRAINED)
    model = model.to(device)

    # Initialize Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    # Initialize Scheduler (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Trackers for Multi-Objective Checkpointing
    best_auc = -1.0
    best_loss = float("inf")

    # Checkpoint paths
    ckpt_dir = config.CHECKPOINT_DIR
    path_auc = os.path.join(ckpt_dir, f"{model_name}_fold_{fold_idx}_auc.pth")
    path_loss = os.path.join(ckpt_dir, f"{model_name}_fold_{fold_idx}_loss.pth")

    # Early Stopping parameters
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        t0 = time.time()

        # Train Step
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validation Step
        val_loss, val_auc, _, _ = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        dt = time.time() - t0
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Time: {dt}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_auc}"
        )

        improved = False

        # Checkpoint: Best AUC (Discriminator Objective)
        if val_auc > best_auc:
            best_auc = val_auc
            print(
                f"  >>> New Best AUC: {best_auc} (Saving to {os.path.basename(path_auc)})"
            )
            torch.save(model.state_dict(), path_auc)
            improved = True

        # Checkpoint: Best Loss (Calibrator Objective)
        if val_loss < best_loss:
            best_loss = val_loss
            print(
                f"  >>> New Best Loss: {val_loss} (Saving to {os.path.basename(path_loss)})"
            )
            torch.save(model.state_dict(), path_loss)
            improved = True

        # Dual-Objective Early Stopping (Cite solution_lesson_node_00082)
        if improved:
            counter = 0  # Reset patience if EITHER metric improves
        else:
            counter += 1

        # Early Stopping Check
        if counter >= patience:
            print(f"  >>> Early stopping triggered after {epoch+1} epochs.")
            break

    # Cleanup
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return best_auc, best_loss


def predict(model, loader, device):
    """
    Runs inference on a loader using a trained model.
    Returns: probabilities (np.array)
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            output = model(data)
            probs = torch.sigmoid(output)
            preds_list.append(probs.cpu().numpy())

    if len(preds_list) > 0:
        return np.concatenate(preds_list)
    else:
        return np.array([])
