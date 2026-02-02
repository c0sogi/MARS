import os
import torch
import nltk


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and other metrics during training.
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


class LevenshteinMetric:
    """
    Computes the Mean Levenshtein Distance between predicted and target strings.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_distance = 0
        self.count = 0

    def update(self, predicted_strings, target_strings):
        """
        Updates the metric with a batch of predictions and targets.

        Args:
            predicted_strings (list of str): List of predicted InChI strings.
            target_strings (list of str): List of ground truth InChI strings.
        """
        assert len(predicted_strings) == len(target_strings)

        for pred, target in zip(predicted_strings, target_strings):
            distance = nltk.edit_distance(pred, target)
            self.total_distance += distance
            self.count += 1

    def compute(self):
        """
        Returns the mean Levenshtein distance.
        """
        if self.count == 0:
            return 0.0
        return self.total_distance / self.count


def save_checkpoint(state, is_best, filepath):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filepath (str): Path to save the checkpoint.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(directory, "best_model.pth")
        torch.save(state, best_path)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device=None):
    """
    Loads a model checkpoint.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into.
        device (torch.device, optional): Device to map the location to.

    Returns:
        start_epoch (int): The epoch to resume from.
        best_score (float): The best score recorded in the checkpoint.
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint file not found at {filepath}")
        return 0, float("inf")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_score = checkpoint.get("best_score", float("inf"))

    print(f"Loaded checkpoint '{filepath}' (epoch {checkpoint.get('epoch', 0)})")
    return start_epoch, best_score
