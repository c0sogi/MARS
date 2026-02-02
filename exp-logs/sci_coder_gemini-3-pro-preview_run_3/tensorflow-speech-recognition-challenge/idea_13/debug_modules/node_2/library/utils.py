import os
import random
import numpy as np
import torch
from library.config import LABEL_TO_IDX, IDX_TO_LABEL


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_accuracy(outputs, targets):
    """
    Calculates the multiclass accuracy.

    Args:
        outputs (torch.Tensor): Logits or probabilities from the model of shape (batch_size, num_classes).
        targets (torch.Tensor): Ground truth indices of shape (batch_size,).

    Returns:
        float: The accuracy score (0.0 to 1.0).
    """
    with torch.no_grad():
        _, preds = torch.max(outputs, 1)
        correct = (preds == targets).sum().item()
        accuracy = correct / targets.size(0)
    return accuracy


class LabelEncoder:
    """
    Utility class to encode and decode labels based on the project configuration.
    """

    def __init__(self):
        self.label_to_idx = LABEL_TO_IDX
        self.idx_to_label = IDX_TO_LABEL

    def encode(self, label):
        """
        Converts a string label to its integer index.
        """
        return self.label_to_idx[label]

    def decode(self, idx):
        """
        Converts an integer index back to its string label.
        """
        return self.idx_to_label[idx]

    def encode_batch(self, labels):
        """
        Encodes a list of labels.
        """
        return [self.encode(l) for l in labels]

    def decode_batch(self, idxs):
        """
        Decodes a list or tensor of indices.
        """
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.cpu().tolist()
        return [self.decode(i) for i in idxs]


def save_checkpoint(model, optimizer, epoch, val_acc, filepath):
    """
    Saves the model and optimizer state to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        val_acc (float): The validation accuracy at this checkpoint.
        filepath (str): Path to save the checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "val_acc": val_acc,
    }
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, device="cpu"):
    """
    Loads the model and optimizer state from a file.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str or torch.device): Device to map the checkpoint to.

    Returns:
        tuple: (epoch, val_acc) loaded from the checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    val_acc = checkpoint.get("val_acc", 0.0)

    return epoch, val_acc
