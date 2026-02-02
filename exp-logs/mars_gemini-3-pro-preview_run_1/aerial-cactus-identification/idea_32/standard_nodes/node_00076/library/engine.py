import torch
import torch.nn as nn
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR
from library.utils import calculate_roc_auc, get_logger
from library.config import DEVICE

# Initialize logger
logger = get_logger("engine")


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch using Mixup and Auxiliary Loss.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training dataloader.
        optimizer (Optimizer): The optimizer.
        criterion (nn.Module): Loss function (BCEWithLogitsLoss).
        device (str): Device to train on.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        images = batch["image"].to(device)
        batch_size = images.size(0)

        # Handle Mixup parameters
        if "lam" in batch:
            lam = batch["lam"]
            target_a = batch["target_a"].to(device).unsqueeze(1)
            target_b = batch["target_b"].to(device).unsqueeze(1)
        else:
            # Fallback if Mixup is not applied
            lam = 1.0
            targets = batch["target"].to(device).unsqueeze(1)
            target_a = targets
            target_b = targets

        optimizer.zero_grad()

        # Forward pass
        # Models are designed to return (main, aux) in training mode
        outputs = model(images)

        if isinstance(outputs, tuple):
            main_out, aux_out = outputs
        else:
            main_out = outputs
            aux_out = None

        # Calculate Main Loss
        loss_main = lam * criterion(main_out, target_a) + (1 - lam) * criterion(
            main_out, target_b
        )

        # Calculate Aux Loss if available
        loss = loss_main
        if aux_out is not None:
            loss_aux = lam * criterion(aux_out, target_a) + (1 - lam) * criterion(
                aux_out, target_b
            )
            loss = loss + 0.4 * loss_aux

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation dataloader.
        criterion (nn.Module): Loss function.
        device (str): Device to evaluate on.

    Returns:
        tuple: (average_loss, roc_auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)
            batch_size = images.size(0)

            # Forward pass
            # Models are designed to return only main output in eval mode
            outputs = model(images)

            # Safety check for tuple return
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, targets)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to logits to get probabilities
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, auc


class SWAHandler:
    """
    Handles Stochastic Weight Averaging (SWA) logic.
    """

    def __init__(self, model, optimizer, swa_start_epoch, swa_lr, device):
        self.model = model
        self.optimizer = optimizer
        self.swa_start_epoch = swa_start_epoch
        self.device = device

        # Initialize SWA model and scheduler
        self.swa_model = AveragedModel(model).to(device)
        self.swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        self.active = False

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch. Updates SWA model if active.

        Returns:
            bool: True if SWA step was performed (caller should skip main scheduler step).
        """
        if epoch >= self.swa_start_epoch:
            self.active = True
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
            return True
        return False

    def update_bn(self, loader):
        """
        Updates BatchNorm statistics for the SWA model using the loader.
        Handles the dictionary-based batch format.
        """
        # Reset running statistics
        for module in self.swa_model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                module.momentum = None
                module.num_batches_tracked *= 0

        self.swa_model.train()
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                # Forward pass to update stats; output is ignored
                self.swa_model(images)

    def get_model(self):
        return self.swa_model


class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    """

    def __init__(self, patience=5, min_delta=0, mode="max"):
        """
        Args:
            patience (int): How many epochs to wait after last time validation score improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                        quantity monitored has stopped decreasing; in 'max' mode it will stop
                        when the quantity monitored has stopped increasing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.inf
        self.val_score_max = -np.inf

    def __call__(self, score):
        save_checkpoint = False

        if self.best_score is None:
            self.best_score = score
            save_checkpoint = True
        elif self.mode == "min":
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                save_checkpoint = True
            else:
                self.counter += 1
        elif self.mode == "max":
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                save_checkpoint = True
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return save_checkpoint
