import time
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import numpy as np

from library.config import Config
from library.utils import AverageMeter, calculate_roc_auc
from library.data_manager import mixup_data


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA) during training.
    """

    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.swa_model = AveragedModel(model).to(device)

        # SWA specific scheduler
        # We use a constant LR for SWA as defined in Config
        self.swa_scheduler = SWALR(
            optimizer, swa_lr=Config.SWA_LR, anneal_epochs=3, anneal_strategy="cos"
        )
        self.active = False

    def step(self, epoch):
        """
        Updates SWA model and scheduler if in SWA phase.
        Returns True if SWA step was performed.
        """
        if epoch >= Config.SWA_START_EPOCH:
            self.active = True
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
            return True
        return False

    def finalize(self, loader):
        """
        Updates BatchNorm statistics for the SWA model.
        Returns the finalized SWA model.
        """
        print("Updating SWA Batch Normalization statistics...")

        # Custom update_bn wrapper to handle the specific input signature (img, fsize)
        # torch.optim.swa_utils.update_bn expects a loader that yields (x, ...)
        # Our loader yields (img, label, fsize, id).
        # We need to manually perform the forward passes to update BN stats.

        self.swa_model.train()
        with torch.no_grad():
            for i, (images, _, fsizes, _) in enumerate(loader):
                images = images.to(self.device)
                fsizes = fsizes.to(self.device)

                # Forward pass through the averaged model to update running statistics
                # We ignore the output
                _ = self.swa_model(images, fsizes)

        return self.swa_model


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scheduler=None):
    """
    Trains the model for one epoch using Mixup and Auxiliary Loss.
    """
    model.train()

    losses = AverageMeter()

    # Check if we are in SWA phase to decide on scheduler stepping if it's per-iteration
    # (Here we assume epoch-based scheduling for the main scheduler, handled outside)

    for i, (images, labels, fsizes, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        fsizes = fsizes.to(device)

        # Apply Mixup
        # Note: mixup_data returns mixed images, soft labels, and mixed metadata
        images, labels, fsizes = mixup_data(
            images, labels, fsizes, alpha=Config.MIXUP_ALPHA, device=device
        )

        # Forward pass
        # Model expects (x, fsize) due to FiLM modulation
        outputs = model(images, fsizes)

        logits = outputs["logits"]
        aux_logits = outputs["aux_logits"]

        # Calculate Loss
        # Weighted sum: Main + 0.4 * Aux
        # Labels are soft targets, so we use the criterion (BCEWithLogits) directly
        loss_main = criterion(logits, labels.unsqueeze(1))
        loss_aux = criterion(aux_logits, labels.unsqueeze(1))

        loss = loss_main + 0.4 * loss_aux

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    # Step the main scheduler if provided and not in SWA phase
    # (Logic usually handled in the main loop, but if per-epoch scheduler is passed here)
    if scheduler is not None and epoch < Config.SWA_START_EPOCH:
        scheduler.step()

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    losses = AverageMeter()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, labels, fsizes, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            fsizes = fsizes.to(device)

            # Forward pass (No Mixup)
            outputs = model(images, fsizes)
            logits = outputs["logits"]

            # Calculate Loss (Main head only for validation metric)
            loss = criterion(logits, labels.unsqueeze(1))
            losses.update(loss.item(), images.size(0))

            # Store predictions for AUC
            probs = torch.sigmoid(logits).cpu().numpy()
            preds_list.extend(probs)
            targets_list.extend(labels.cpu().numpy())

    preds_arr = np.array(preds_list)
    targets_arr = np.array(targets_list)

    roc_auc = calculate_roc_auc(targets_arr, preds_arr)

    return losses.avg, roc_auc
