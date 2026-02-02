import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import roc_auc_score
import numpy as np

from library.config import Config
from library.utils import MetricMonitor, get_logger
from library.dataset import mixup_data

logger = get_logger(__name__)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the mixup loss: lambda * loss(a) + (1 - lambda) * loss(b).
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion_cls,
    criterion_aux,
    device,
    epoch,
    mixup_alpha=0.2,
    mtl_weight=0.1,
):
    """
    Trains the model for one epoch using Mixup and Multi-Task Learning.
    """
    model.train()
    monitor = MetricMonitor()

    for batch_idx, (images, labels, film_inputs, mtl_targets, _) in enumerate(
        train_loader
    ):
        images = images.to(device)
        labels = labels.to(device).float()
        film_inputs = film_inputs.to(device)
        mtl_targets = mtl_targets.to(device).float()

        # Combine labels and auxiliary targets for unified Mixup
        # Stack to shape (B, 2)
        combined_targets = torch.stack([labels, mtl_targets], dim=1)

        # Apply Mixup
        mixed_images, targets_a, targets_b, lam = mixup_data(
            images, combined_targets, mixup_alpha, device
        )

        # Unpack targets back to specific tasks
        # Shape becomes (B, 1) for BCE/MSE compatibility
        labels_a = targets_a[:, 0].unsqueeze(1)
        labels_b = targets_b[:, 0].unsqueeze(1)
        mtl_a = targets_a[:, 1].unsqueeze(1)
        mtl_b = targets_b[:, 1].unsqueeze(1)

        # Forward pass
        logits, quality_pred = model(mixed_images, film_inputs)

        # Calculate Multi-Task Loss
        loss_cls = mixup_criterion(criterion_cls, logits, labels_a, labels_b, lam)
        loss_aux = mixup_criterion(criterion_aux, quality_pred, mtl_a, mtl_b, lam)

        loss = loss_cls + mtl_weight * loss_aux

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        monitor.update("Loss", loss.item())
        monitor.update("ClsLoss", loss_cls.item())
        monitor.update("AuxLoss", loss_aux.item())

    return (
        monitor.get_avg("Loss"),
        monitor.get_avg("ClsLoss"),
        monitor.get_avg("AuxLoss"),
    )


def validate(model, val_loader, criterion_cls, criterion_aux, device, mtl_weight=0.1):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    monitor = MetricMonitor()
    preds = []
    targets = []

    with torch.no_grad():
        for batch_idx, (images, labels, film_inputs, mtl_targets, _) in enumerate(
            val_loader
        ):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            film_inputs = film_inputs.to(device)
            mtl_targets = mtl_targets.to(device).float().unsqueeze(1)

            # Forward pass
            logits, quality_pred = model(images, film_inputs)

            # Calculate Loss (No Mixup)
            loss_cls = criterion_cls(logits, labels)
            loss_aux = criterion_aux(quality_pred, mtl_targets)
            loss = loss_cls + mtl_weight * loss_aux

            monitor.update("Loss", loss.item())

            # Store predictions for AUC
            probs = torch.sigmoid(logits)
            preds.extend(probs.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    # Calculate ROC AUC
    try:
        auc = roc_auc_score(targets, preds)
    except ValueError:
        auc = 0.5  # Fallback if only one class is present in batch (unlikely with stratified split)

    return monitor.get_avg("Loss"), auc


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA) lifecycle:
    - Initialization of AveragedModel
    - SWA Scheduler stepping
    - Custom BatchNorm update for multi-input models
    """

    def __init__(self, model, optimizer, config):
        self.config = config
        self.device = config.DEVICE
        self.enabled = config.USE_SWA
        self.start_epoch = config.SWA_START_EPOCH

        if self.enabled:
            self.swa_model = AveragedModel(model).to(self.device)
            self.swa_scheduler = SWALR(optimizer, swa_lr=config.SWA_LR)
        else:
            self.swa_model = None
            self.swa_scheduler = None

    def on_epoch_end(self, model, epoch, main_scheduler=None):
        """
        Called at the end of each epoch to update SWA model or step main scheduler.
        """
        if not self.enabled:
            if main_scheduler:
                main_scheduler.step()
            return

        if epoch >= self.start_epoch:
            # Update SWA model parameters
            self.swa_model.update_parameters(model)
            # Step SWA scheduler
            self.swa_scheduler.step()
        else:
            # Step standard scheduler
            if main_scheduler:
                main_scheduler.step()

    def finalize(self, loader):
        """
        Finalizes SWA by updating BatchNorm statistics.
        Uses a custom loop to handle the (image, film_input) signature.
        """
        if not self.enabled:
            return

        logger.info("SWA: Updating BatchNorm statistics...")
        self._update_bn(loader)

    def get_model(self):
        """Returns the averaged model if SWA is enabled, else None."""
        return self.swa_model if self.enabled else None

    def _update_bn(self, loader):
        """
        Custom implementation of torch.optim.swa_utils.update_bn
        to support models with multiple inputs (images, film_inputs).
        """
        model = self.swa_model
        device = self.device

        # Reset BN statistics
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum
                module.momentum = None
                module.num_batches_tracked *= 0

        model.train()
        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(device)
                film_inputs = batch[2].to(device)

                # Forward pass to accumulate stats
                model(images, film_inputs)

        # Restore momenta
        for module in momenta.keys():
            module.momentum = momenta[module]
