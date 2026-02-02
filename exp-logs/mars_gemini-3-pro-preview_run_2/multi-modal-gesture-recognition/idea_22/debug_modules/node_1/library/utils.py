import os
import random
import numpy as np
import torch
import shutil
from nltk import edit_distance
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking losses and metrics during training.
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


def compute_levenshtein(predictions, targets):
    """
    Computes the Levenshtein error rate (normalized edit distance).

    Args:
        predictions (list of list of int): Predicted gesture IDs for each sequence.
        targets (list of list of int): Ground truth gesture IDs for each sequence.

    Returns:
        float: The Levenshtein error rate (Total Distance / Total Ground Truth Length).
    """
    total_distance = 0
    total_length = 0

    for pred, target in zip(predictions, targets):
        # Ensure inputs are lists
        p = list(pred)
        t = list(target)

        # Calculate edit distance for this sequence
        dist = edit_distance(p, t)

        total_distance += dist
        total_length += len(t)

    if total_length == 0:
        return 0.0

    return total_distance / total_length


def save_checkpoint(
    state, is_best, checkpoint_dir=Config.CHECKPOINT_DIR, filename="checkpoint.pth"
):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Filename for the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_path=None):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): The scheduler to load state into.
        checkpoint_path (str, optional): Path to the checkpoint file. Defaults to best_model.pth.

    Returns:
        tuple: (start_epoch, best_score)
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"[-] No checkpoint found at {checkpoint_path}")
        return 0, float("inf")

    print(f"[+] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint.get("epoch", 0), checkpoint.get("best_score", float("inf"))


def save_submission(predictions, ids, output_path=None):
    """
    Saves predictions to a CSV file in the required format.
    Format: SessionID,Label1,Label2,...

    Args:
        predictions (list of list of int): Predicted gesture IDs.
        ids (list of str): Sample IDs (e.g., 'Sample00300').
        output_path (str, optional): Path to save the submission file.
    """
    if output_path is None:
        output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for sample_id, pred_seq in zip(ids, predictions):
            # Convert list of ints to comma-separated string
            # Example: [2, 12, 3] -> "2,12,3"
            pred_str = ",".join(map(str, pred_seq))

            # Write line: SessionID,Label1,Label2,...
            line = f"{sample_id},{pred_str}\n"
            f.write(line)

    print(f"[+] Submission saved to {output_path}")
