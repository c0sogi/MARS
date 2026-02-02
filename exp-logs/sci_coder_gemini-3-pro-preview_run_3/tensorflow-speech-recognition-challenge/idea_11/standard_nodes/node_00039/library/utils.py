import os
import torch
import pandas as pd
import numpy as np
import shutil
from library.config import path_cfg, set_seed


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = "Metric", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def calculate_accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
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


class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    Supports both 'min' (for loss) and 'max' (for accuracy) modes.
    """

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pth", mode="min"
    ):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
            verbose (bool): If True, prints a message for each validation improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the
                quantity monitored has stopped decreasing; in 'max' mode it will stop when the
                quantity monitored has stopped increasing.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.mode = mode

        if mode == "min":
            self.val_score_best = np.inf
        else:
            self.val_score_best = -np.inf

    def __call__(self, score, model, optimizer=None, epoch=None):

        if self.mode == "min":
            improved = self.best_score is None or score < self.best_score - self.delta
        else:
            improved = self.best_score is None or score > self.best_score + self.delta

        if improved:
            self.best_score = score
            self.save_checkpoint(score, model, optimizer, epoch)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model, optimizer, epoch):
        """Saves model when validation metric improves."""
        if self.verbose:
            print(
                f"Validation metric improved ({self.val_score_best} --> {score}).  Saving model to {self.path}"
            )

        self.val_score_best = score

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "best_score": score,
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(state, self.path)


def save_metrics(metrics_dict, file_path):
    """
    Saves a dictionary of metrics to a CSV file. Appends if file exists.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    df = pd.DataFrame([metrics_dict])
    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, mode="w", header=True, index=False)


def load_checkpoint(model, path, optimizer=None, device="cpu"):
    """
    Loads model weights and optimizer state from a checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_score = checkpoint.get("best_score", None)

    return start_epoch, best_score


def count_parameters(model):
    """Returns the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
