import torch
import torch.nn as nn
import numpy as np
from torch.optim.swa_utils import AveragedModel, update_bn
from library.config import Config
from library.utils import AverageMeter, compute_multilabel_auc


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        # Apply Mixup
        mixed_images, targets_a, targets_b, lam = mixup_data(
            images, targets, Config.MIXUP_ALPHA, device
        )

        optimizer.zero_grad()

        outputs = model(mixed_images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and macro-averaged ROC AUC.
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid for AUC calculation
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    auc = compute_multilabel_auc(all_targets, all_preds)

    print(f"Validation Results - Loss: {losses.avg}, AUC: {auc}")

    return losses.avg, auc


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA).
    """

    def __init__(self, model, swa_start_epoch, device):
        self.swa_start_epoch = swa_start_epoch
        self.device = device
        # avg_fn=None uses the default running average
        self.swa_model = AveragedModel(model).to(device)
        self.active = False

    def on_epoch_end(self, model, epoch):
        """
        Updates the SWA model if the current epoch is past the start epoch.
        """
        if epoch >= self.swa_start_epoch:
            self.active = True
            self.swa_model.update_parameters(model)

    def finalize(self, loader):
        """
        Updates Batch Normalization statistics for the SWA model.
        Should be called at the end of training.
        """
        if self.active:
            print("Updating SWA Batch Normalization statistics...")
            update_bn(loader, self.swa_model, device=self.device)

    def get_model(self):
        """
        Returns the averaged model if SWA was active, otherwise None.
        """
        return self.swa_model if self.active else None


def inference(model, loader, device, use_tta=False):
    """
    Generates predictions for the given loader.
    Supports Test-Time Augmentation (Horizontal Flip).
    Returns raw probabilities (after Sigmoid).
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # Standard forward pass
            logits = model(images)
            preds = torch.sigmoid(logits)

            if use_tta:
                # Horizontal Flip TTA
                # Flip along the width dimension (last dimension)
                images_flipped = torch.flip(images, dims=[-1])
                logits_flipped = model(images_flipped)
                preds_flipped = torch.sigmoid(logits_flipped)

                # Average probabilities
                preds = (preds + preds_flipped) / 2.0

            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds, axis=0)
