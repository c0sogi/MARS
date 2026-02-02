import torch
from library.config import set_seed


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the set_seed function from library.config to ensure consistency.

    Args:
        seed (int): The seed value to use.
    """
    set_seed(seed)


def calculate_accuracy(output, target):
    """
    Computes the classification accuracy for multi-class problems.

    Args:
        output (torch.Tensor): Model predictions (logits or probabilities) of shape (Batch_Size, Num_Classes).
        target (torch.Tensor): Ground truth class indices of shape (Batch_Size).

    Returns:
        float: The accuracy score as a float between 0.0 and 1.0.
    """
    with torch.no_grad():
        # Get the class index with the maximum value (logit/probability)
        # output shape: [batch_size, num_classes] -> preds shape: [batch_size]
        preds = torch.argmax(output, dim=1)

        # Ensure target is the same shape/device
        if target.ndim > 1:
            target = target.view(-1)

        # Calculate number of correct predictions
        correct = (preds == target).sum().item()

        # Calculate accuracy
        accuracy = correct / target.size(0)

    return accuracy


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics like loss and accuracy during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all internal statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The current value (e.g., batch loss or accuracy).
            n (int): The weight of the value (usually batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
