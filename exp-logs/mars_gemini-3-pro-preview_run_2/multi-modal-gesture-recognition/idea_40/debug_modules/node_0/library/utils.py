import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_pad_mask(lengths, max_len=None):
    """
    Creates a boolean mask indicating padding positions.
    True indicates the position is padding (should be ignored).
    False indicates the position is valid.

    Args:
        lengths (torch.Tensor): Tensor of shape (batch_size,) containing sequence lengths.
        max_len (int, optional): The maximum length of the sequences. If None, uses max(lengths).

    Returns:
        torch.Tensor: Boolean mask of shape (batch_size, max_len).
    """
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = lengths.max().item()

    # Create a range tensor [0, 1, ..., max_len-1]
    ids = (
        torch.arange(0, max_len, device=lengths.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    # Mask is True where index >= length (padding)
    mask = ids >= lengths.unsqueeze(1)

    return mask


def compute_bone_vectors(joints):
    """
    Computes 3D bone vectors from joint positions based on the connectivity defined in Config.

    Args:
        joints (torch.Tensor): Tensor of shape (batch_size, time_steps, num_joints, 3)
                               or (time_steps, num_joints, 3).
                               Assumes joints are ordered according to Config.SELECTED_JOINTS.

    Returns:
        torch.Tensor: Tensor of bone vectors with shape (..., num_bones, 3).
    """
    bone_pairs = Config.BONE_PAIRS
    vectors = []

    # Determine the dimension index for joints
    # If 4D: (B, T, J, 3) -> dim 2
    # If 3D: (T, J, 3) -> dim 1
    joint_dim = -2

    for parent_idx, child_idx in bone_pairs:
        # parent and child are indices into the selected joints dimension
        parent_pos = joints.select(joint_dim, parent_idx)
        child_pos = joints.select(joint_dim, child_idx)

        # Vector = Child - Parent
        vec = child_pos - parent_pos
        vectors.append(vec)

    # Stack along the joint dimension
    bone_vectors = torch.stack(vectors, dim=joint_dim)
    return bone_vectors


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
    """
    # Ensure checkpoint directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None):
    """
    Loads a checkpoint into the model (and optionally optimizer/scheduler).

    Args:
        model (torch.nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.

    Returns:
        int: The epoch to resume from (if found in checkpoint), else 0.
        float: The best metric score (if found), else None.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", None)

    return start_epoch, best_score
