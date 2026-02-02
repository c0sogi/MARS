import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.swa_utils import update_bn as torch_update_bn
import numpy as np
from library.utils import AverageMeter, calculate_roc_auc


def train_one_epoch(model, loader, optimizer, device, epoch, mixup_fn=None):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        device: Device to run training on.
        epoch: Current epoch number.
        mixup_fn: Optional Mixup object/function.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    losses = AverageMeter("Loss")
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        # Apply Mixup if provided
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    print(f"Epoch {epoch} Training Loss: {losses.avg}")
    return losses.avg


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        device: Device to run evaluation on.

    Returns:
        tuple: (Average Loss, ROC AUC Score)
    """
    model.eval()
    losses = AverageMeter("Loss")
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

            # Apply sigmoid to get probabilities for AUC calculation
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        auc = calculate_roc_auc(all_targets, all_preds)
    else:
        auc = 0.5

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation AUC: {auc}")

    return losses.avg, auc


class SWAEngine:
    """
    Manages Stochastic Weight Averaging (SWA) training phase.
    """

    def __init__(self, model, optimizer, config, swa_start_epoch):
        """
        Args:
            model: The base model being trained.
            optimizer: The optimizer used for the base model.
            config: Configuration object containing SWA params.
            swa_start_epoch: Epoch to start SWA.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.swa_start_epoch = swa_start_epoch
        self.device = config.DEVICE

        # Initialize the Averaged Model
        self.swa_model = AveragedModel(model).to(self.device)

        # Initialize SWA Learning Rate Scheduler
        self.swa_scheduler = SWALR(
            optimizer,
            swa_lr=config.SWA_LR,
            anneal_epochs=config.SWA_ANNEAL_EPOCHS,
            anneal_strategy=config.SWA_ANNEAL_STRATEGY,
        )

    def is_swa_active(self, epoch):
        """Checks if SWA should be active for the current epoch."""
        return epoch >= self.swa_start_epoch

    def step(self, epoch):
        """
        Performs SWA updates. Should be called at the end of each epoch.

        Returns:
            bool: True if SWA step was performed, False otherwise.
        """
        if self.is_swa_active(epoch):
            # Update the averaged model with current base model parameters
            self.swa_model.update_parameters(self.model)
            # Step the SWA scheduler
            self.swa_scheduler.step()
            return True
        return False

    def update_bn(self, loader):
        """
        Updates BatchNorm statistics for the SWA model.
        Must be called at the end of training using the training loader.
        """
        print("Updating SWA BatchNorm statistics...")
        torch_update_bn(loader, self.swa_model, self.device)

    def get_averaged_model(self):
        """Returns the SWA model."""
        return self.swa_model
