import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_auc


def train_one_epoch(epoch, model, train_loader, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()

    losses = AverageMeter()

    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    for step, (images, labels, _) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply Label Smoothing
        # y_ls = y * (1 - alpha) + 0.5 * alpha
        smooth_labels = (
            labels * (1.0 - Config.LABEL_SMOOTHING) + 0.5 * Config.LABEL_SMOOTHING
        )

        optimizer.zero_grad()

        outputs = model(images)
        # Flatten outputs to match label shape (B,)
        loss = criterion(outputs.view(-1), smooth_labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    if scheduler is not None:
        scheduler.step()

    print(f"Epoch {epoch}: Train Loss: {losses.avg}")
    return losses.avg


def validate(model, val_loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    losses = AverageMeter()
    all_targets = []
    all_preds = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            preds = torch.sigmoid(outputs.view(-1))

            loss = criterion(outputs.view(-1), labels)

            losses.update(loss.item(), images.size(0))

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    auc_score = calculate_auc(np.array(all_targets), np.array(all_preds))

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation AUC: {auc_score}")

    return losses.avg, auc_score


def predict_tta(model, test_loader, device):
    """
    Generates predictions using Test Time Augmentation (TTA).
    TTA Strategy: Original, Horizontal Flip, Vertical Flip, Rotate 90.
    Returns a DataFrame with 'id' and 'label' (probability).
    """
    model.eval()

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for images, _, ids in test_loader:
            images = images.to(device, non_blocking=True)

            # 1. Original
            out_orig = model(images)
            prob_orig = torch.sigmoid(out_orig.view(-1))

            # 2. Horizontal Flip
            images_h = torch.flip(images, dims=[3])
            out_h = model(images_h)
            prob_h = torch.sigmoid(out_h.view(-1))

            # 3. Vertical Flip
            images_v = torch.flip(images, dims=[2])
            out_v = model(images_v)
            prob_v = torch.sigmoid(out_v.view(-1))

            # 4. Rotate 90
            images_r = torch.rot90(images, k=1, dims=[2, 3])
            out_r = model(images_r)
            prob_r = torch.sigmoid(out_r.view(-1))

            # Average predictions
            avg_prob = (prob_orig + prob_h + prob_v + prob_r) / 4.0

            ids_list.extend(ids)
            preds_list.extend(avg_prob.cpu().numpy())

    # Create DataFrame
    df_preds = pd.DataFrame({"id": ids_list, "label": preds_list})

    return df_preds
