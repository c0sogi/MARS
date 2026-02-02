import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # deterministic=True ensures reproducibility but might impact performance slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


def save_checkpoint(
    state, is_best, filename="checkpoint.pth.tar", output_dir=Config.WORKING_DIR
):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Boolean flag indicating if this is the best model so far.
        filename (str): Name of the checkpoint file.
        output_dir (str): Directory to save the checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(output_dir, "model_best.pth.tar")
        shutil.copyfile(filepath, best_path)


def calculate_map5(output, target):
    """
    Computes the Mean Average Precision @ 5 (MAP@5).

    Args:
        output (torch.Tensor or np.ndarray): Model output.
            Can be logits of shape (N, num_classes) or top-5 indices of shape (N, 5).
        target (torch.Tensor or np.ndarray): Ground truth labels of shape (N,).

    Returns:
        float: The MAP@5 score.
    """
    # Convert tensors to numpy
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu()
    else:
        output = torch.from_numpy(output)

    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    else:
        target = np.array(target)

    # If output is logits (N, C), convert to top 5 indices (N, 5)
    if output.dim() == 2 and output.shape[1] > 5:
        _, preds = output.topk(5, 1, True, True)
        preds = preds.numpy()
    else:
        preds = output.numpy()

    score = 0.0
    batch_size = len(target)

    for i in range(batch_size):
        p = preds[i]
        t = target[i]

        # MAP@5 for a single ground truth per image:
        # Score is 1/(k+1) if the correct label is at index k (0-indexed), else 0.
        # We stop checking after finding the match because there is only 1 correct label.
        for k in range(min(5, len(p))):
            if p[k] == t:
                score += 1.0 / (k + 1)
                break

    return score / batch_size
