import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Seeds all random number generators for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and accuracy during training.
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


def map_per_image(predicted_labels, true_label):
    """
    Calculates the Average Precision (AP) for a single image.
    For single-label classification, AP is 1/rank if the true label is in the
    top-k predictions, otherwise 0.
    """
    try:
        # Check if the true label is in the predictions
        # We assume predicted_labels is a list of labels (strings or ints)
        # and true_label is a single label (string or int)
        rank = predicted_labels.index(true_label)
        return 1.0 / (rank + 1)
    except ValueError:
        return 0.0


def map_at_5(predictions, targets):
    """
    Calculates the Mean Average Precision @ 5 (MAP@5).

    Args:
        predictions (list of list): A list where each element is a list of top-5 predicted labels.
        targets (list): A list of ground truth labels.

    Returns:
        float: The MAP@5 score.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Length of predictions ({len(predictions)}) must match targets ({len(targets)})"
        )

    scores = []
    for preds, target in zip(predictions, targets):
        # Ensure we only consider the top 5 predictions
        top_5_preds = preds[:5]
        scores.append(map_per_image(top_5_preds, target))

    return np.mean(scores)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """
    Saves the model checkpoint to the working directory.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): The name of the checkpoint file.
    """
    # Ensure working directory exists (though Config.setup() does this, it's safer to check)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(Config.WORKING_DIR, "model_best.pth.tar")
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    model, optimizer=None, filename="checkpoint.pth.tar", device=Config.DEVICE
):
    """
    Loads a checkpoint into the model and optional optimizer.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        filename (str): The filename of the checkpoint to load.
        device (torch.device): The device to map the checkpoint to.

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch, best_score, etc.)
        None: If file not found.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)

    if not os.path.isfile(filepath):
        # Try looking in the root of working dir just in case, or return None
        if os.path.isfile(filename):
            filepath = filename
        else:
            return None

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Load model state
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
