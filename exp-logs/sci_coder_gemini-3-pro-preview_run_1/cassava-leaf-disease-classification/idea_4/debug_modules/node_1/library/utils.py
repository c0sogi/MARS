import os
import sys
import random
import logging
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Seeds all random number generators to ensure fully reproducible results.

    Args:
        seed (int): The random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_file: str):
    """
    Configures and returns a logger that writes to both a file and the console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_device():
    """
    Returns the PyTorch device (CUDA if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(output, target):
    """
    Computes the top-1 accuracy.

    Args:
        output (torch.Tensor): Model outputs (logits or probabilities) of shape (N, C).
        target (torch.Tensor): Ground truth labels of shape (N).

    Returns:
        float: Accuracy ratio (0.0 to 1.0).
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(1.0 / batch_size).item()


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
    """

    def __init__(self):
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


class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    Saves the best model checkpoint.
    """

    def __init__(
        self, patience=7, mode="max", delta=0, verbose=False, path="checkpoint.pth"
    ):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
            mode (str): One of 'min' (e.g. loss) or 'max' (e.g. accuracy).
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf
        self.val_score_max = -np.Inf

        if self.mode == "min":
            self.check_func = lambda current, best: current < best - self.delta
        else:
            self.check_func = lambda current, best: current > best + self.delta

    def __call__(self, score, model):
        """
        Args:
            score (float): The metric value to monitor (e.g. val_loss or val_acc).
            model (torch.nn.Module): The model to save.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif not self.check_func(score, self.best_score):
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation score improves."""
        if self.verbose:
            if self.mode == "min":
                print(
                    f"Validation score decreased ({self.val_score_min:.6f} --> {score:.6f}).  Saving model ..."
                )
                self.val_score_min = score
            else:
                print(
                    f"Validation score increased ({self.val_score_max:.6f} --> {score:.6f}).  Saving model ..."
                )
                self.val_score_max = score

        # Handle DataParallel or DDP wrapper
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        torch.save(state_dict, self.path)
