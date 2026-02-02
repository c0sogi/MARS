import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import get_auc_score
from library.loss import MultiTaskLoss


def train_one_epoch(
    model, optimizer, dataloader, device, epoch, ema_model=None, scheduler=None
):
    """
    Trains the model for one epoch using Automatic Mixed Precision (AMP) and Multi-Task Loss.
    """
    model.train()

    criterion = MultiTaskLoss()
    scaler = torch.cuda.amp.GradScaler()

    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["targets"].to(device)
        masks = batch["mask"].to(device)
        mask_validity = batch["mask_validity"].to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            cls_logits, seg_logits = model(images)
            loss_dict = criterion(cls_logits, targets, seg_logits, masks, mask_validity)
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        if ema_model is not None:
            ema_model.update(model)

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss and the average AUC score.
    """
    model.eval()

    criterion = MultiTaskLoss()

    running_loss = 0.0
    dataset_size = 0

    preds = []
    valid_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["targets"].to(device)
            masks = batch["mask"].to(device)
            mask_validity = batch["mask_validity"].to(device)

            batch_size = images.size(0)

            cls_logits, seg_logits = model(images)
            loss_dict = criterion(cls_logits, targets, seg_logits, masks, mask_validity)
            loss = loss_dict["loss"]

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to classification logits for AUC calculation
            batch_preds = torch.sigmoid(cls_logits)

            preds.append(batch_preds.cpu().numpy())
            valid_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    preds = np.concatenate(preds, axis=0)
    valid_targets = np.concatenate(valid_targets, axis=0)

    auc_score = get_auc_score(valid_targets, preds)

    return epoch_loss, auc_score


def inference_fn(model, dataloader, device):
    """
    Generates predictions for the test set using Test Time Augmentation (TTA).
    TTA: Average of original image prediction and horizontally flipped image prediction.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)

            # 1. Forward pass with original images
            cls_logits, _ = model(images)
            probs_orig = torch.sigmoid(cls_logits)

            # 2. Forward pass with horizontally flipped images (TTA)
            # Flip along width dimension (dim 3 for NCHW tensor)
            images_flip = torch.flip(images, dims=[3])
            cls_logits_flip, _ = model(images_flip)
            probs_flip = torch.sigmoid(cls_logits_flip)

            # 3. Average probabilities
            avg_probs = (probs_orig + probs_flip) / 2.0

            preds.append(avg_probs.cpu().numpy())

    return np.concatenate(preds, axis=0)
