import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel, SWALR
from library.utils import AverageMeter, get_logger
from library.config import Config

# Initialize Logger
logger = get_logger(name="engine")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler=None,
):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    # Binary Cross Entropy with Logits
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets, fsizes) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        fsizes = fsizes.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass with dual inputs (image + metadata)
        outputs = model(images, fsizes)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    # Step scheduler if it's per-epoch (not SWA scheduler, handled separately)
    if scheduler is not None:
        scheduler.step()

    return loss_meter.avg


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC.
    """
    model.eval()
    loss_meter = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets, fsizes in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            fsizes = fsizes.to(device, non_blocking=True)

            outputs = model(images, fsizes)
            loss = criterion(outputs, targets)

            loss_meter.update(loss.item(), images.size(0))

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate ROC AUC
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle edge case with single class in batch/set
        auc_score = 0.5

    return loss_meter.avg, auc_score


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA).
    Wraps AveragedModel and handles custom BN updates for dual-input models.
    """

    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, config: Config
    ):
        self.use_swa = config.USE_SWA
        self.start_epoch = config.SWA_START_EPOCH
        self.device = config.DEVICE

        if self.use_swa:
            self.swa_model = AveragedModel(model)
            self.swa_scheduler = SWALR(optimizer, swa_lr=config.SWA_LR)
        else:
            self.swa_model = None
            self.swa_scheduler = None

    def step(self, epoch: int, model: nn.Module):
        """
        Updates SWA model parameters if current epoch >= start_epoch.
        """
        if not self.use_swa:
            return

        if epoch >= self.start_epoch:
            self.swa_model.update_parameters(model)
            self.swa_scheduler.step()

    def update_bn(self, loader: torch.utils.data.DataLoader):
        """
        Custom update_bn implementation to handle (image, fsize) input signature.
        Standard torch.optim.swa_utils.update_bn fails with multi-argument forward methods.
        """
        if not self.use_swa:
            return

        logger.info("Updating SWA Batch Normalization statistics...")

        swa_model = self.swa_model
        device = self.device

        # Reset BN running stats
        momenta = {}
        for module in swa_model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum
                module.momentum = None
                module.num_batches_tracked *= 0

        swa_model.train()

        with torch.no_grad():
            for images, _, fsizes in loader:
                images = images.to(device, non_blocking=True)
                fsizes = fsizes.to(device, non_blocking=True)

                # Forward pass updates BN running stats
                swa_model(images, fsizes)

        # Restore momenta
        for module in swa_model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.momentum = momenta[module]

        logger.info("SWA BN update complete.")

    def get_model(self):
        return self.swa_model


def save_checkpoint(state, is_best, filepath):
    """
    Saves the model checkpoint.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    if is_best:
        # Construct best model path
        best_path = filepath.replace("checkpoint", "best_model")
        if "checkpoint" not in filepath:
            best_path = filepath + "_best"
        torch.save(state, best_path)


def load_checkpoint(filepath, model, device):
    """
    Loads model weights from a checkpoint file.
    """
    if not os.path.exists(filepath):
        logger.error(f"Checkpoint not found at {filepath}")
        return None

    logger.info(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath, map_location=device)

    # Handle state dict loading (support both full checkpoint dict and direct state dict)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model
