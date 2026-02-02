import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, update_bn

from library.config import Config
from library.utils import MetricMonitor, calculate_roc_auc
from library.data import mixup_data


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup and Multi-Task Loss.
    """
    model.train()
    metric_monitor = MetricMonitor()

    # Loss functions
    # BCEWithLogitsLoss is used for binary classification (supports soft targets from Mixup)
    criterion_cls = nn.BCEWithLogitsLoss()
    # MSELoss is used for the auxiliary file size regression task
    criterion_reg = nn.MSELoss()

    for batch_idx, (images, labels, fsizes, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)
        fsizes = fsizes.to(device).float().view(-1, 1)

        # Apply Mixup to inputs and targets (both class and regression targets)
        mixed_images, mixed_labels, mixed_fsizes = mixup_data(
            images, labels, fsizes, alpha=Config.MIXUP_ALPHA, device=device
        )

        optimizer.zero_grad()

        # Forward pass: returns (class_logits, quality_pred)
        class_logits, quality_pred = model(mixed_images)

        # Calculate Multi-Task Loss
        loss_cls = criterion_cls(class_logits, mixed_labels)
        loss_reg = criterion_reg(quality_pred, mixed_fsizes)

        # Weighted sum of losses
        total_loss = loss_cls + (Config.AUX_LOSS_WEIGHT * loss_reg)

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        # Update metrics
        metric_monitor.update("Loss", total_loss.item())
        metric_monitor.update("ClsLoss", loss_cls.item())
        metric_monitor.update("RegLoss", loss_reg.item())

    return metric_monitor


def validate(model, val_loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    metric_monitor = MetricMonitor()

    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels, fsizes, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            fsizes = fsizes.to(device).float().view(-1, 1)

            class_logits, quality_pred = model(images)

            loss_cls = criterion_cls(class_logits, labels)
            loss_reg = criterion_reg(quality_pred, fsizes)
            total_loss = loss_cls + (Config.AUX_LOSS_WEIGHT * loss_reg)

            metric_monitor.update("Loss", total_loss.item())
            metric_monitor.update("ClsLoss", loss_cls.item())
            metric_monitor.update("RegLoss", loss_reg.item())

            # Store predictions for AUC calculation
            preds = torch.sigmoid(class_logits)
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    # Concatenate all batches to calculate global AUC
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    auc = calculate_roc_auc(all_targets, all_preds)

    metric_monitor.update("AUC", auc)

    return metric_monitor


class SWAHandler:
    """
    Handles Stochastic Weight Averaging (SWA) updates and finalization.
    """

    def __init__(self, model):
        self.swa_model = AveragedModel(model)
        self.start_epoch = Config.SWA_START_EPOCH

    def step(self, model, epoch):
        """
        Updates the SWA model parameters if the current epoch is within the SWA range.
        """
        if epoch >= self.start_epoch:
            self.swa_model.update_parameters(model)

    def finalize(self, model, loader, device):
        """
        Finalizes SWA by updating Batch Norm statistics and copying weights back to the main model.
        """
        # Update BN statistics using the provided loader (usually train loader)
        # The loader returns (img, label, fsize, id), update_bn uses the first element (img)
        update_bn(loader, self.swa_model, device=device)

        # Copy the averaged parameters back to the original model
        model.load_state_dict(self.swa_model.module.state_dict())
