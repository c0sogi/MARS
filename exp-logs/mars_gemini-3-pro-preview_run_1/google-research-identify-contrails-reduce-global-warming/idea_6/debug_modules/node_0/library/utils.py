import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across all libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) for submission.

    The metric checks that the pairs are sorted, positive, and the decoded pixel
    values are not duplicated. The pixels are numbered from top to bottom,
    then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

    Args:
        mask (np.ndarray): Binary mask of shape (Height, Width).

    Returns:
        str: Space-delimited list of pairs (start, length) or '-' if empty.
    """
    # Flatten column-major (Fortran-style) to match the "top to bottom, then left to right" requirement
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect transitions at the start and end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is the start of the first run
    # runs[1] is the end of the first run
    # The length is runs[1] - runs[0]
    # Update the end indices to be lengths
    runs[1::2] -= runs[::2]

    # Convert to string
    encoded = " ".join(str(x) for x in runs)

    return encoded if encoded else "-"


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking Loss during training.
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


class GlobalDiceMeter:
    """
    Tracks the global intersection and union statistics to compute the
    Global Dice Coefficient exactly as defined in the metric description.

    Formula: 2 * |X n Y| / (|X| + |Y|)
    where X is the set of predicted pixels and Y is the ground truth.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.intersection = 0.0
        self.union = 0.0

    def update(self, y_pred, y_true):
        """
        Updates the intersection and union counts.

        Args:
            y_pred (torch.Tensor or np.ndarray): Predicted probabilities or binary mask.
            y_true (torch.Tensor or np.ndarray): Ground truth binary mask.
        """
        # Handle Tensor inputs
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu()
            y_true = y_true.detach().cpu()

            # Binarize predictions based on threshold
            y_pred = (y_pred > Config.THRESHOLD).float()
            y_true = y_true.float()

            inter = (y_pred * y_true).sum()
            sum_pred = y_pred.sum()
            sum_true = y_true.sum()

            self.intersection += inter.item()
            self.union += (sum_pred + sum_true).item()

        # Handle Numpy inputs
        else:
            y_pred = (y_pred > Config.THRESHOLD).astype(float)
            y_true = y_true.astype(float)

            inter = (y_pred * y_true).sum()
            self.intersection += inter
            self.union += y_pred.sum() + y_true.sum()

    def get_score(self):
        """
        Calculates the Global Dice Coefficient.

        Returns:
            float: The Dice score.
        """
        smooth = 1e-6  # Epsilon to avoid division by zero
        return (2.0 * self.intersection + smooth) / (self.union + smooth)


def save_checkpoint(state, is_best, checkpoint_dir, best_model_name="best_model.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        best_model_name (str): Filename for the best model.
    """
    filename = os.path.join(checkpoint_dir, "checkpoint.pth")
    torch.save(state, filename)
    if is_best:
        best_path = os.path.join(checkpoint_dir, best_model_name)
        torch.save(state, best_path)
