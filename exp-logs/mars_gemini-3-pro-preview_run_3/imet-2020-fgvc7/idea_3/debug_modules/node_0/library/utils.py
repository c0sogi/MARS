import os
import random
import numpy as np
import torch
import shutil
from sklearn.metrics import f1_score


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_micro_f1(preds, targets, threshold=0.5):
    """
    Calculates the Micro-averaged F1 score.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities (N, C).
        targets (torch.Tensor or np.ndarray): Ground truth binary labels (N, C).
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        float: Micro F1 score.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions based on threshold
    binary_preds = (preds > threshold).astype(int)

    return f1_score(targets, binary_preds, average="micro")


def find_optimal_threshold(preds, targets, num_steps=100):
    """
    Finds the probability threshold that maximizes Micro F1 score.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities.
        targets (torch.Tensor or np.ndarray): Ground truth labels.
        num_steps (int): Number of steps to search between 0 and 1.

    Returns:
        tuple: (best_threshold, best_f1_score)
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    thresholds = np.linspace(0.01, 0.99, num_steps)
    best_threshold = 0.5
    best_f1 = 0.0

    for thresh in thresholds:
        binary_preds = (preds > thresh).astype(int)
        f1 = f1_score(targets, binary_preds, average="micro")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    return best_threshold, best_f1


def save_checkpoint(state, is_best, filepath):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model, optimizer, etc.
        is_best (bool): Whether this checkpoint is the best so far.
        filepath (str): Path to save the checkpoint.
    """
    torch.save(state, filepath)


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"
):
    """
    Loads a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (str or torch.device): Device to map the checkpoint to.

    Returns:
        tuple: (start_epoch, best_score)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("best_score", 0.0)


class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    """

    def __init__(
        self, patience=5, delta=0, mode="max", verbose=False, path="checkpoint.pth"
    ):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                        quantity monitored has stopped decreasing; in 'max' mode it will stop
                        when the quantity monitored has stopped increasing.
            verbose (bool): If True, prints a message for each validation improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf
        self.val_score_max = -np.Inf

    def __call__(self, score, model, optimizer=None, scheduler=None, epoch=None):
        if self.mode == "max":
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(score, model, optimizer, scheduler, epoch)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(score, model, optimizer, scheduler, epoch)
                self.counter = 0
        elif self.mode == "min":
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(score, model, optimizer, scheduler, epoch)
            elif score > self.best_score - self.delta:
                self.counter += 1
                if self.verbose:
                    print(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(score, model, optimizer, scheduler, epoch)
                self.counter = 0

    def save_checkpoint(self, score, model, optimizer, scheduler, epoch):
        """Saves model when validation score improves."""
        if self.verbose:
            if self.mode == "max":
                print(
                    f"Validation score improved ({self.val_score_max} --> {score}).  Saving model ..."
                )
            else:
                print(
                    f"Validation loss decreased ({self.val_score_min} --> {score}).  Saving model ..."
                )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_score": score,
        }
        if optimizer:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler:
            state["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(state, self.path)

        if self.mode == "max":
            self.val_score_max = score
        else:
            self.val_score_min = score
