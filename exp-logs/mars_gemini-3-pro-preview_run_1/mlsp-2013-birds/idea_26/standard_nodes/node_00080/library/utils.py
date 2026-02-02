import os
import random
import shutil
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Also configures CuDNN for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(state, is_best, output_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint. If is_best is True, copies the file to 'model_best.pth'.

    Args:
        state (dict): The state dictionary to save (model, optimizer, epoch, etc.).
        is_best (bool): Whether this checkpoint represents the best model so far.
        output_dir (str): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(output_dir, "model_best.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(
    model, checkpoint_path, optimizer=None, scheduler=None, device="cpu"
):
    """
    Loads a checkpoint into the model, and optionally into the optimizer and scheduler.

    Args:
        model (torch.nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (str or torch.device): Device to map the location to.

    Returns:
        dict: The loaded checkpoint dictionary, or None if file not found.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[-] Checkpoint not found at {checkpoint_path}")
        return None

    print(f"[+] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle DataParallel state_dict keys (remove 'module.' prefix if present)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the macro-averaged ROC AUC score.

    Args:
        y_true (np.array or torch.Tensor): Ground truth labels (N, NumClasses).
        y_pred (np.array or torch.Tensor): Predicted probabilities (N, NumClasses).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Handle NaNs in predictions by replacing them with 0
    if np.isnan(y_pred).any():
        y_pred = np.nan_to_num(y_pred, nan=0.0)

    try:
        # Calculate macro-average ROC AUC
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Fallback for cases where a class might be missing from the batch/split
        # Compute per-column and average valid columns
        n_classes = y_true.shape[1]
        scores = []
        for i in range(n_classes):
            try:
                # Only compute if class exists in y_true (has both 0 and 1)
                if len(np.unique(y_true[:, i])) > 1:
                    s = roc_auc_score(y_true[:, i], y_pred[:, i])
                    scores.append(s)
            except ValueError:
                pass

        if len(scores) > 0:
            score = np.mean(scores)
        else:
            score = 0.5  # Neutral score if calculation fails completely

    return score


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class Logger:
    """
    Simple logger that writes messages to both stdout and a log file.
    """

    def __init__(self, log_file):
        self.log_file = log_file
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Initialize/Overwrite the log file
        with open(self.log_file, "w") as f:
            pass

    def log(self, message):
        print(message)
        with open(self.log_file, "a") as f:
            f.write(str(message) + "\n")


def write_submission_csv(ids, probabilities, output_path):
    """
    Writes the submission CSV file in the required format.

    Format:
    Id,Probability
    rec_id*100+species_id, probability

    Args:
        ids (list or np.array): List of recording IDs (rec_id) for the test set.
        probabilities (np.array): (N, 19) array of predicted probabilities.
        output_path (str): Path to save the CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("Id,Probability\n")
        for i, rec_id in enumerate(ids):
            # Ensure rec_id is integer
            rec_id_int = int(rec_id)
            probs = probabilities[i]

            for species_idx, prob in enumerate(probs):
                # Construct composite ID
                row_id = rec_id_int * 100 + species_idx
                f.write(f"{row_id},{prob}\n")
