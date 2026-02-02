import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from library.utils import mixup_data, mixup_criterion


def train_one_epoch(
    model, dataloader, optimizer, criterion, device, epoch, mixup_alpha=0.2
):
    """
    Trains the model for one epoch using Mixup and Deep Supervision.

    Args:
        model (nn.Module): The RepVGG model (must return (output, aux_output) in train mode).
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): PyTorch optimizer.
        criterion (Loss): Loss function (e.g., BCEWithLogitsLoss).
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number (for logging).
        mixup_alpha (float): Alpha parameter for Beta distribution in Mixup.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, labels, alpha=mixup_alpha, device=device
        )

        optimizer.zero_grad()

        # Forward pass
        # RepVGG in training mode (deploy=False) returns (main_out, aux_out)
        outputs, aux_outputs = model(images)

        # Compute Loss (Weighted Sum of Main and Aux)
        loss_main = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss_aux = mixup_criterion(criterion, aux_outputs, targets_a, targets_b, lam)

        # 0.4 weight for auxiliary loss as per strategy
        loss = loss_main + 0.4 * loss_aux

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Device to run evaluation on.

    Returns:
        float: ROC AUC score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Forward pass
            # RepVGG in eval mode returns only logits (no aux output)
            # If model is AveragedModel (SWA), it wraps the underlying model
            outputs = model(images)

            # Apply sigmoid to logits
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    if len(all_preds) == 0:
        return 0.0

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle cases where only one class is present in the batch/set
        auc = 0.5

    print(f"Validation AUC: {auc}")
    return auc


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA) logic.
    """

    def __init__(self, model, optimizer, swa_start_epoch, swa_lr=1e-4):
        """
        Args:
            model (nn.Module): The base model.
            optimizer (Optimizer): The optimizer used for training.
            swa_start_epoch (int): The epoch to start SWA.
            swa_lr (float): The learning rate to use during SWA phase.
        """
        self.swa_start_epoch = swa_start_epoch
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        self.active = False

    def step(self, epoch, model, standard_scheduler=None):
        """
        Updates SWA state at the end of an epoch.

        Args:
            epoch (int): Current epoch index (0-based).
            model (nn.Module): Current state of the model to average.
            standard_scheduler (Scheduler, optional): The standard scheduler to step
                                                      if SWA hasn't started yet.
        """
        if epoch >= self.swa_start_epoch:
            self.active = True
            self.swa_model.update_parameters(model)
            self.swa_scheduler.step()
        else:
            if standard_scheduler is not None:
                standard_scheduler.step()

    def finalize(self, train_loader, device):
        """
        Finalizes the SWA model by updating Batch Norm statistics.

        Args:
            train_loader (DataLoader): Loader to use for BN update.
            device (torch.device): Computation device.

        Returns:
            nn.Module: The finalized SWA model.
        """
        print("Updating SWA Batch Norm statistics...")
        update_bn(train_loader, self.swa_model, device=device)
        return self.swa_model
