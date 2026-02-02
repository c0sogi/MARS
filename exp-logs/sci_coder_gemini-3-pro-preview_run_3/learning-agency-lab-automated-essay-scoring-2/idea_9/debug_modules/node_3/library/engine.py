import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import get_logger, compute_qwk

logger = get_logger("engine")


class AWP:
    """
    Adversarial Weight Perturbation (AWP) class.
    Perturbs model weights in the direction of the gradient to maximize loss,
    acting as a regularizer to improve generalization.
    """

    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=Config.AWP_LR,
        adv_eps=Config.AWP_EPS,
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs, labels, attention_mask, criterion, scaler):
        """
        Executes the adversarial attack:
        1. Saves current weights.
        2. Perturbs weights based on current gradients.
        3. Performs a forward pass and calculates adversarial loss.
        4. Backpropagates the adversarial loss.
        5. Restores original weights.
        """
        with torch.amp.autocast("cuda", enabled=True):
            self._save()
            self._attack_step()

            # Forward pass with perturbed weights
            outputs = self.model(inputs, attention_mask)

            # Determine loss type and compute adversarial loss
            if outputs.shape[1] == 1:
                # Regression
                adv_loss = criterion(outputs.view(-1), labels.view(-1))
            else:
                # Ordinal Classification
                # Generate ordinal targets: (B, 5)
                levels = torch.arange(1, Config.NUM_ORDINAL_LABELS + 1).to(
                    labels.device
                )
                ordinal_labels = (labels.unsqueeze(1) > levels).float()
                adv_loss = criterion(outputs, ordinal_labels)

            # Scale and backward
            scaler.scale(adv_loss).backward()

            self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


def train_one_epoch(
    model, dataloader, optimizer, scheduler, device, epoch, criterion, scaler, awp=None
):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model.
        dataloader: Training dataloader.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: Torch device (cuda/cpu).
        epoch: Current epoch number.
        criterion: Loss function (MSELoss or BCEWithLogitsLoss).
        scaler: GradScaler for mixed precision.
        awp: Instance of AWP class (optional).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()

    dataset_size = 0
    running_loss = 0.0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        # Mixed Precision Context
        with torch.amp.autocast("cuda", enabled=True):
            outputs = model(input_ids, attention_mask)

            # Calculate Loss
            if outputs.shape[1] == 1:
                # Regression: Output is (B, 1), Labels are (B,)
                loss = criterion(outputs.view(-1), labels.view(-1))
            else:
                # Ordinal: Output is (B, 5), Labels are (B,)
                # Convert scalar scores to ordinal vectors
                # e.g., Score 3 -> [1, 1, 0, 0, 0]
                levels = torch.arange(1, Config.NUM_ORDINAL_LABELS + 1).to(device)
                ordinal_labels = (labels.unsqueeze(1) > levels).float()
                loss = criterion(outputs, ordinal_labels)

        # Normalize loss for gradient accumulation
        loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

        # Backward pass (accumulate gradients)
        scaler.scale(loss).backward()

        # Gradient Accumulation Step
        if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:

            # Adversarial Weight Perturbation
            if awp is not None and epoch >= Config.AWP_START_EPOCH:
                awp.attack_backward(
                    input_ids, labels, attention_mask, criterion, scaler
                )

            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Scheduler Step (per iteration)
            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * Config.GRADIENT_ACCUMULATION_STEPS) * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        dataloader: Validation dataloader.
        device: Torch device.
        criterion: Loss function.

    Returns:
        tuple: (average_loss, qwk_score, predictions, targets)
    """
    model.eval()

    running_loss = 0.0
    dataset_size = 0

    preds = []
    targets = []
    essay_ids = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if "essay_ids" in batch:
                essay_ids.extend(batch["essay_ids"])

            batch_size = input_ids.size(0)

            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(input_ids, attention_mask)

                if outputs.shape[1] == 1:
                    # Regression
                    loss = criterion(outputs.view(-1), labels.view(-1))
                    batch_preds = outputs.view(-1).cpu().numpy()
                else:
                    # Ordinal
                    levels = torch.arange(1, Config.NUM_ORDINAL_LABELS + 1).to(device)
                    ordinal_labels = (labels.unsqueeze(1) > levels).float()
                    loss = criterion(outputs, ordinal_labels)

                    # Convert Logits to Score
                    # Probabilities P(y > k)
                    probs = torch.sigmoid(outputs)
                    # Expected Score = 1 + Sum(P(y > k))
                    batch_preds = (probs.sum(dim=1) + 1).cpu().numpy()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            preds.extend(batch_preds)
            targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Aggregate chunk predictions by essay_id
    if essay_ids:
        df_preds = pd.DataFrame(
            {"essay_id": essay_ids, "pred": preds, "target": targets}
        )
        # Group by essay_id and average predictions
        df_agg = df_preds.groupby("essay_id", sort=False).mean()
        final_preds = df_agg["pred"].values
        final_targets = df_agg["target"].values
    else:
        final_preds = np.array(preds)
        final_targets = np.array(targets)

    # Compute Metric
    qwk = compute_qwk(final_targets, final_preds)

    return epoch_loss, qwk, final_preds, final_targets
