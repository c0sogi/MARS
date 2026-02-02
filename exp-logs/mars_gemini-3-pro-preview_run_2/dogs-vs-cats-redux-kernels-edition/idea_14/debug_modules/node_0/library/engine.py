import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import get_score


def mixup_data(x, y, alpha=0.2, device="cuda"):
    """
    Applies Mixup augmentation.
    Returns mixed inputs and mixed targets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    # y is already float (BCE targets), so we mix directly
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y


def cutmix_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies CutMix augmentation.
    Returns mixed inputs and mixed targets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    # Get random bounding box
    W = x.size(2)
    H = x.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Adjust lambda to exact pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    # Apply CutMix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    mixed_y = lam * y + (1 - lam) * y[index]

    return x, mixed_y


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup or CutMix
        # Config says MIXUP_PROB = 1.0, so we always apply one.
        # We'll split 50/50 between Mixup and CutMix.
        p = np.random.rand()
        if p < 0.5:
            images, labels = mixup_data(
                images, labels, alpha=Config.MIXUP_ALPHA, device=device
            )
        else:
            images, labels = cutmix_data(
                images, labels, alpha=Config.CUTMIX_ALPHA, device=device
            )

        optimizer.zero_grad()

        # Forward pass
        # Model returns logits (num_classes=1)
        outputs = model(images).squeeze(1)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, data_loader, device):
    """
    Validates the model on the validation set.
    Returns average loss and log loss score.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    preds = []
    targets = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            # Apply sigmoid for metric calculation
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / num_batches

    # Calculate Log Loss
    score = get_score(np.array(targets), np.array(preds))

    return avg_loss, score


def inference_fn(model, data_loader, device):
    """
    Runs inference on the test set with TTA (Horizontal Flip).
    Returns a dictionary mapping IDs to predicted probabilities.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for images, ids in data_loader:
            images = images.to(device)

            # 1. Original
            out_orig = model(images).squeeze(1)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip (TTA)
            images_flip = torch.flip(images, dims=[3])
            out_flip = model(images_flip).squeeze(1)
            prob_flip = torch.sigmoid(out_flip)

            # Average
            avg_prob = (prob_orig + prob_flip) / 2.0

            # Store
            avg_prob_np = avg_prob.cpu().numpy()
            ids_np = ids.numpy() if isinstance(ids, torch.Tensor) else ids

            for i, img_id in enumerate(ids_np):
                results[img_id] = avg_prob_np[i]

    return results


def train_fold(
    model, train_loader, val_loader, optimizer, scheduler, device, fold_id, model_name
):
    """
    Orchestrates the training for a single fold, including checkpointing for Model Soup.
    """
    best_score = float("inf")

    # Directory to save checkpoints for this specific model and fold
    # Using Config.CHECKPOINT_DIR

    print(f"Starting training for {model_name} - Fold {fold_id}")

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss, val_score = validate(model, val_loader, device)

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val LogLoss: {val_score:.6f}"
        )

        # Save checkpoints for Model Soup (epochs 18, 19, 20)
        if epoch in Config.SOUP_EPOCHS:
            ckpt_name = f"{model_name}_fold_{fold_id}_ep{epoch}.pth"
            save_path = os.path.join(Config.CHECKPOINT_DIR, ckpt_name)
            torch.save(model.state_dict(), save_path)
            print(f"  Saved Soup Checkpoint: {ckpt_name}")

        # Save Best Model (Standard practice, though Soup is the goal)
        if val_score < best_score:
            best_score = val_score
            ckpt_name = f"best_{model_name}_fold_{fold_id}.pth"
            save_path = os.path.join(Config.CHECKPOINT_DIR, ckpt_name)
            torch.save(model.state_dict(), save_path)
            # print(f"  New Best Score! Saved: {ckpt_name}")

    print(f"Fold {fold_id} finished. Best Val LogLoss: {best_score:.6f}")
    return best_score
