import torch
from library.config import seed_everything


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    Wraps the library.config.seed_everything function.

    Args:
        seed (int): The seed value to set.
    """
    seed_everything(seed)


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
        """
        Update the meter with a new value.

        Args:
            val (float): The current value (e.g., batch loss).
            n (int): The weight/count for this value (e.g., batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """
    Computes the accuracy for binary classification.

    Args:
        output (torch.Tensor): Logits or probabilities from the model.
        target (torch.Tensor): Ground truth labels (0 or 1).

    Returns:
        float: The accuracy percentage (0-100).
    """
    with torch.no_grad():
        batch_size = target.size(0)

        # Check if output is logits (apply sigmoid) or already probabilities
        # Assuming logits for BCEWithLogitsLoss context, but safe to check range or just threshold
        # For this specific task (BCEWithLogitsLoss), output is logits.
        preds = (torch.sigmoid(output) > 0.5).float()

        # Ensure shapes match
        preds = preds.view(-1)
        target = target.view(-1)

        correct = preds.eq(target).sum()
        acc = correct.float().mul_(100.0 / batch_size)

        return acc.item()
