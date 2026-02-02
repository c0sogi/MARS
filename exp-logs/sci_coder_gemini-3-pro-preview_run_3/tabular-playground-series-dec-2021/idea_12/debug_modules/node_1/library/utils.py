import os
import random
import numpy as np
import torch
import copy


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to enforce reproducibility
    across Python, NumPy, and PyTorch (CPU and CUDA).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def calculate_accuracy(output, target):
    """
    Calculates the classification accuracy.

    Args:
        output (torch.Tensor): Logits or probabilities of shape [Batch, Num_Classes]
        target (torch.Tensor): Ground truth class indices of shape [Batch]

    Returns:
        float: The accuracy (correct predictions / total samples).
    """
    with torch.no_grad():
        # Get the index of the max log-probability
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)


class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    Stores the best model state using deepcopy to avoid mutation issues.
    """

    def __init__(self, patience=10, mode="max", delta=0.0):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
            mode (str): One of {'min', 'max'}.
                        'min' for metrics like loss (lower is better).
                        'max' for metrics like accuracy (higher is better).
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

        # Initialize validation score based on mode
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, metric, model):
        score = metric

        if self.mode == "min":
            improved = (
                score < (self.best_score - self.delta)
                if self.best_score is not None
                else True
            )
        else:
            improved = (
                score > (self.best_score + self.delta)
                if self.best_score is not None
                else True
            )

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif not improved:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model state in memory when validation metric improves."""
        self.val_score = score
        # Use deepcopy to ensure we store the exact state at this moment,
        # preventing subsequent optimizer steps from mutating the 'best' weights.
        self.best_state = copy.deepcopy(model.state_dict())

    def load_best_weights(self, model):
        """Loads the best weights stored in memory into the provided model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        else:
            print("Warning: No best state saved in EarlyStopping.")
