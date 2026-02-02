import math
import sys
import torch
import torch.nn as nn
from typing import Iterable, Optional, Callable, Dict, Any

from library.utils import AverageMeter
from library.config import Config


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)
) -> Iterable[torch.Tensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (torch.Tensor): Model predictions of shape (N, C).
        target (torch.Tensor): Ground truth labels of shape (N).
        topk (tuple): Tuple of k values to compute accuracy for.

    Returns:
        List[torch.Tensor]: List of accuracy values for each k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    loss_fn: Callable,
    max_norm: float = None,
    model_ema: Optional[Any] = None,
    mixup_fn: Optional[Callable] = None,
    accum_iter: int = 1,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model to train.
        optimizer: The optimizer.
        data_loader: The training data loader.
        device: The device to run training on.
        epoch: Current epoch number.
        loss_fn: The loss function.
        max_norm: Gradient clipping value (optional).
        model_ema: Model EMA instance (optional).
        mixup_fn: Mixup/Cutmix function (optional).
        accum_iter: Gradient accumulation steps.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    optimizer.zero_grad()

    for batch_idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply MixUp / CutMix if enabled
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # Forward pass
        outputs = model(samples)
        loss = loss_fn(outputs, targets)

        # Normalize loss for gradient accumulation
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter
        loss.backward()

        # Step optimizer and update EMA every accum_iter steps
        if (batch_idx + 1) % accum_iter == 0:
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            optimizer.zero_grad()

            if model_ema is not None:
                model_ema.update(model)

        loss_meter.update(loss_value, samples.size(0))

    return loss_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module, data_loader: Iterable, loss_fn: Callable, device: torch.device
) -> Dict[str, float]:
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model to evaluate.
        data_loader: The validation data loader.
        loss_fn: The loss function.
        device: The device to run evaluation on.

    Returns:
        Dict[str, float]: Dictionary containing 'loss' and 'accuracy'.
    """
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward pass
        outputs = model(samples)
        loss = loss_fn(outputs, targets)

        # Compute accuracy (Top-1)
        acc1 = accuracy(outputs, targets, topk=(1,))[0]

        loss_meter.update(loss.item(), samples.size(0))
        acc_meter.update(acc1.item(), samples.size(0))

    # Print metrics with full precision as requested
    print(f"Validation Loss: {loss_meter.avg}")
    print(f"Validation Accuracy: {acc_meter.avg}")

    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """

    def __init__(self, patience: int = 5, mode: str = "max", delta: float = 0.0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation metric improved.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                        quantity monitored has stopped decreasing; in 'max' mode it will
                        stop when the quantity monitored has stopped increasing.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "min":
            self.val_score_fn = lambda x: -x
        else:
            self.val_score_fn = lambda x: x

    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
            return True  # Improvement found (first run)

        score_to_check = self.val_score_fn(score)
        best_to_check = self.val_score_fn(self.best_score)

        if score_to_check < best_to_check + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # No improvement
        else:
            self.best_score = score
            self.counter = 0
            return True  # Improvement found
