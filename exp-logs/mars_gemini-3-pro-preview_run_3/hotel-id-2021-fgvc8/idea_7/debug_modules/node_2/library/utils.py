import os
import shutil
import torch
import numpy as np
from library.config import Config, seed_everything


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name=None):
        self.name = name
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
        output (torch.Tensor): Model output logits or probabilities (N, C).
        target (torch.Tensor): Ground truth labels (N).
        topk (tuple): Tuple of k values to compute accuracy for.

    Returns:
        list: List of accuracy values (percentage 0-100) for each k.
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


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    Args:
        actual (list): The ground truth elements (should be a list of items).
        predicted (list): The predicted elements (ordered list of items).
        k (int): The maximum number of predicted elements.

    Returns:
        float: The average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    Args:
        actual (list of lists): The ground truth lists.
        predicted (list of lists): The predicted lists.
        k (int): The maximum number of predicted elements.

    Returns:
        float: The mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def save_checkpoint(state, is_best, output_dir=None, filename="checkpoint.pth"):
    """
    Saves model checkpoint to disk.

    Args:
        state (dict): Model state dictionary (params, optimizer, epoch, etc.).
        is_best (bool): Whether this checkpoint is the best so far.
        output_dir (str, optional): Directory to save the checkpoint. Defaults to Config.output_dir.
        filename (str): Filename for the checkpoint.
    """
    if output_dir is None:
        output_dir = Config.output_dir

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(output_dir, "best_model.pth"))
