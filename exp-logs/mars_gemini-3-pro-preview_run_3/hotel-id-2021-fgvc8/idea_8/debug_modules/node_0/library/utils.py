import os
import random
import shutil
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
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


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
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


def mean_average_precision(predictions, targets, k=5):
    """
    Computes Mean Average Precision @ K (MAP@K) for single-label classification.

    Args:
        predictions (torch.Tensor): Shape (N, num_classes) containing logits,
                                    or (N, K) containing top-k indices.
        targets (torch.Tensor): Shape (N,) containing ground truth class indices.
        k (int): The cutoff rank.

    Returns:
        float: The MAP@K score.
    """
    # Ensure tensors are on CPU for list operations
    if predictions.device != torch.device("cpu"):
        predictions = predictions.cpu()
    if targets.device != torch.device("cpu"):
        targets = targets.cpu()

    # If predictions are logits (floating point), extract top k indices
    if predictions.is_floating_point():
        _, topk_preds = predictions.topk(k, dim=1, largest=True, sorted=True)
    else:
        # If already indices, ensure we only look at the first k
        topk_preds = predictions[:, :k]

    batch_size = targets.size(0)
    map_sum = 0.0

    # Calculate AP for each sample in the batch
    for i in range(batch_size):
        target = targets[i].item()
        preds = topk_preds[i].tolist()

        score = 0.0
        if target in preds:
            # Rank is 0-indexed, so we add 1 for reciprocal rank
            rank = preds.index(target)
            score = 1.0 / (rank + 1)

        map_sum += score

    return map_sum / batch_size


def save_checkpoint(state, is_best, filepath, best_filepath):
    """
    Saves the model checkpoint to the specified filepath.
    If is_best is True, copies the file to best_filepath.

    Args:
        state (dict): The state dictionary to save (model, optimizer, epoch, etc.).
        is_best (bool): Whether this checkpoint represents the best metric so far.
        filepath (str): Path to save the standard checkpoint.
        best_filepath (str): Path to save the best model checkpoint.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, filepath, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads model weights and optional optimizer/scheduler state from a checkpoint.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        filepath (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (optional): Scheduler to load state into.
        device (str or torch.device): Device to map the checkpoint to.

    Returns:
        tuple: (start_epoch, best_score)
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint not found at {filepath}")
        return 0, 0.0

    print(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=device)

    # Handle DataParallel state_dict keys (remove 'module.' prefix)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_score = checkpoint.get("best_score", 0.0)

    print(f"Loaded checkpoint '{filepath}' (epoch {checkpoint.get('epoch', 0)})")
    return start_epoch, best_score
