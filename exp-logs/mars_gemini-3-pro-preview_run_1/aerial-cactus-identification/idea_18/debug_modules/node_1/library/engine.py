import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel

from library.config import Config
from library.utils import MetricMonitor, get_logger

logger = get_logger("engine")


def mixup_data(x, y, f_norm, f_log, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to inputs and targets.
    Mixes images, FiLM features (file_size_norm), class labels, and regression targets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    # Mix the FiLM input as well so the model conditions on "mixed quality"
    mixed_f_norm = lam * f_norm + (1 - lam) * f_norm[index, :]

    y_a, y_b = y, y[index]
    f_log_a, f_log_b = f_log, f_log[index]

    return mixed_x, mixed_f_norm, y_a, y_b, f_log_a, f_log_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the mixup-weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """
    Training loop for one epoch with Mixup and MTL.
    """
    model.train()
    metric_monitor = MetricMonitor(float_precision=6)

    # Loss functions
    cls_criterion = nn.BCEWithLogitsLoss()
    mtl_criterion = nn.MSELoss()

    for batch in train_loader:
        # Move data to device
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float().view(-1, 1)
        fsize_norm = batch["file_size_norm"].to(device, non_blocking=True)
        fsize_log = (
            batch["file_size_log"].to(device, non_blocking=True).float().view(-1, 1)
        )

        batch_size = images.size(0)

        # Apply Mixup
        mixed_images, mixed_f_norm, y_a, y_b, f_log_a, f_log_b, lam = mixup_data(
            images,
            labels,
            fsize_norm,
            fsize_log,
            alpha=Config.MIXUP_ALPHA,
            device=device,
        )

        # Forward pass
        outputs = model(mixed_images, file_size_norm=mixed_f_norm)
        logits = outputs["logits"]

        # Calculate Losses
        # 1. Classification Loss
        loss_cls = mixup_criterion(cls_criterion, logits, y_a, y_b, lam)

        # 2. Auxiliary Regression Loss (MTL)
        loss_mtl = torch.tensor(0.0, device=device)
        if "mtl_pred" in outputs and Config.MTL_LOSS_WEIGHT > 0:
            loss_mtl = mixup_criterion(
                mtl_criterion, outputs["mtl_pred"], f_log_a, f_log_b, lam
            )

        # Combined Loss
        loss = loss_cls + Config.MTL_LOSS_WEIGHT * loss_mtl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Metrics
        metric_monitor.update("Loss", loss.item(), batch_size)
        metric_monitor.update("ClsLoss", loss_cls.item(), batch_size)
        if Config.MTL_LOSS_WEIGHT > 0:
            metric_monitor.update("MtlLoss", loss_mtl.item(), batch_size)

    return metric_monitor.metrics


def evaluate(model, val_loader, device):
    """
    Evaluation loop with 4-view Test Time Augmentation (TTA).
    Views: Original, H-Flip, V-Flip, HV-Flip (Rotation).
    """
    model.eval()
    metric_monitor = MetricMonitor(float_precision=6)

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float().view(-1, 1)
            fsize_norm = batch["file_size_norm"].to(device, non_blocking=True)

            batch_size = images.size(0)

            # TTA Accumulator
            probs_accum = torch.zeros_like(labels)

            # 1. Original View
            out1 = model(images, file_size_norm=fsize_norm)
            probs_accum += torch.sigmoid(out1["logits"])

            # 2. Horizontal Flip
            out2 = model(torch.flip(images, [3]), file_size_norm=fsize_norm)
            probs_accum += torch.sigmoid(out2["logits"])

            # 3. Vertical Flip
            out3 = model(torch.flip(images, [2]), file_size_norm=fsize_norm)
            probs_accum += torch.sigmoid(out3["logits"])

            # 4. H+V Flip (180 degree rotation)
            out4 = model(torch.flip(images, [2, 3]), file_size_norm=fsize_norm)
            probs_accum += torch.sigmoid(out4["logits"])

            # Average predictions
            avg_probs = probs_accum / 4.0

            # Calculate Validation Loss (Soft BCE using averaged probabilities)
            # Clamp to prevent log(0)
            avg_probs_clamped = torch.clamp(avg_probs, 1e-7, 1 - 1e-7)
            loss = nn.BCELoss()(avg_probs_clamped, labels)

            metric_monitor.update("Loss", loss.item(), batch_size)

            preds_list.append(avg_probs.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    all_preds = np.concatenate(preds_list)
    all_targets = np.concatenate(targets_list)

    # Compute ROC AUC
    # Handle edge cases (e.g., single class in batch or dummy test targets)
    if len(np.unique(all_targets)) > 1:
        auc_score = roc_auc_score(all_targets, all_preds)
    else:
        auc_score = 0.5

    return metric_monitor.metrics, auc_score, all_preds, all_targets


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA).
    """

    def __init__(self, model, start_epoch, device):
        self.swa_model = AveragedModel(model).to(device)
        self.start_epoch = start_epoch
        self.device = device
        self.active = False

    def update(self, model, epoch):
        """Updates SWA weights if current epoch >= start_epoch."""
        if epoch >= self.start_epoch:
            self.swa_model.update_parameters(model)
            self.active = True

    def update_bn(self, loader):
        """
        Custom Batch Normalization update for SWA.
        Handles the dictionary-based loader and model signature.
        """
        if not self.active:
            return

        # Reset BN statistics
        for module in self.swa_model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                module.num_batches_tracked *= 0

        self.swa_model.train()
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                fsize_norm = batch["file_size_norm"].to(self.device)

                # Forward pass updates running stats
                self.swa_model(images, file_size_norm=fsize_norm)

    def get_model(self):
        return self.swa_model
