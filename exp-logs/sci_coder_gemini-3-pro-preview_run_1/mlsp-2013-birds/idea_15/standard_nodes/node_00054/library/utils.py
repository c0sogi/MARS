import os
import random
import shutil
import numpy as np
import torch


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
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def save_checkpoint(
    state,
    is_best,
    filename="checkpoint.pth",
    best_filename="model_best.pth",
    save_dir="./working/idea_15",
):
    """
    Saves the training checkpoint.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
        best_filename (str): Name of the best model file.
        save_dir (str): Directory to save the checkpoints.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(save_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)
        # print(f"Saved new best model to {best_filepath}")


def load_checkpoint(model, filename, optimizer=None, device="cuda"):
    """
    Loads a checkpoint into the model and optional optimizer.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): Path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to ('cpu' or 'cuda').

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch, best_score, etc.)
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No checkpoint found at '{filename}'")

    print(f"Loading checkpoint from '{filename}'")
    checkpoint = torch.load(filename, map_location=device, weights_only=False)

    # Handle DataParallel wrapping if necessary
    state_dict = checkpoint["state_dict"]
    # Check if the model is wrapped (DataParallel or AveragedModel) by looking for 'module.' prefix in any key.
    # This avoids brittle checks on the first key which might be an SWA buffer like 'n_averaged'. Cite {debug_lesson_9}
    if any(k.startswith("module.") for k in state_dict.keys()):
        # If the checkpoint was saved from DataParallel but loading into single GPU/CPU
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                name = k[7:]  # remove 'module.'
                new_state_dict[name] = v
            # We skip keys like 'n_averaged' that don't have the prefix and aren't part of the base model
        state_dict = new_state_dict

    model.load_state_dict(state_dict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def verify_data_integrity(data, name="Data"):
    """
    Checks for NaNs or Infs in the provided data array.
    Useful for validating pseudo-labels or predictions.

    Args:
        data (np.ndarray or torch.Tensor): The data to check.
        name (str): Name of the data for error messaging.

    Raises:
        ValueError: If NaNs or Infs are found.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    if np.isnan(data).any():
        raise ValueError(f"Data integrity check failed: {name} contains NaNs.")

    if np.isinf(data).any():
        raise ValueError(f"Data integrity check failed: {name} contains Infs.")

    # print(f"Data integrity check passed for {name}.")
