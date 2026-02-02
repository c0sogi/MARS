import os
import random
import shutil
import numpy as np
import torch
from library.config import CFG


def seed_everything(seed=CFG.seed):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to CFG.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Used for tracking loss and accuracy.
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


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (torch.Tensor): Model logits or probabilities [batch_size, num_classes].
        target (torch.Tensor): Ground truth labels [batch_size].
        topk (tuple): Tuple of k values to compute accuracy for (e.g., (1, 5)).

    Returns:
        list: A list of accuracy values (in percentage) for each k in topk.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get the top k indices
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        # Compare predictions with target
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # Sum the correct predictions for top-k
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(
    state, is_best, output_dir=CFG.output_dir, filename="checkpoint.pth"
):
    """
    Saves the model checkpoint to the output directory.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        output_dir (str): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Save the current checkpoint
    torch.save(state, filepath)

    # If it's the best model, create a copy named 'model_best.pth'
    if is_best:
        best_filepath = os.path.join(output_dir, "model_best.pth")
        shutil.copyfile(filepath, best_filepath)
