import os
import random
import numpy as np
import torch
import pandas as pd


def set_seed(seed):
    """
    Sets the seed for random number generators to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in CuDNN backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_metadata(split):
    """
    Loads the metadata parquet file for a specific split.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    valid_splits = ["train", "val", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Expected one of {valid_splits}")

    path = os.path.join("./metadata", f"{split}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_parquet(path)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model and optimizer state to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch.
        loss (float): Current loss value.
        path (str): Destination path for the checkpoint.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "loss": loss,
        },
        path,
    )
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """
    Loads a checkpoint into the model and optimizer.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str or torch.device): Device to map the checkpoint to.

    Returns:
        tuple: (epoch, loss) from the checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    print(f"Loaded checkpoint from {path} (Epoch {epoch})")
    return epoch, loss


def manage_caching(filename, generate_fn, load_cached_data=True, **kwargs):
    """
    Manages caching of processed data to disk to avoid re-computation.

    Args:
        filename (str): Name of the cache file (must end in .parquet or .npy).
        generate_fn (callable): Function to generate data if cache is missed.
                                Must return a DataFrame (for parquet) or numpy array (for npy).
        load_cached_data (bool): If True, attempts to load from disk first.
        **kwargs: Arguments passed to generate_fn.

    Returns:
        The loaded or generated data.
    """
    cache_dir = "./working/idea_3/"
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, filename)

    # 1. Try to load
    if load_cached_data and os.path.exists(file_path):
        print(f"Loading cached data from {file_path}")
        try:
            if filename.endswith(".parquet"):
                return pd.read_parquet(file_path)
            elif filename.endswith(".npy"):
                return np.load(file_path, allow_pickle=False)
            else:
                raise ValueError("Unsupported cache file format. Use .parquet or .npy")
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    # 2. Compute/Process
    print(f"Generating data for {filename}...")
    data = generate_fn(**kwargs)

    # 3. Save
    if filename.endswith(".parquet"):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "generate_fn must return a pandas DataFrame for .parquet caching"
            )
        data.to_parquet(file_path, index=False)
    elif filename.endswith(".npy"):
        if not isinstance(data, np.ndarray):
            raise TypeError("generate_fn must return a numpy array for .npy caching")
        np.save(file_path, data)
    else:
        raise ValueError("Unsupported cache file format. Use .parquet or .npy")

    print(f"Data saved to {file_path}")
    return data


def print_metrics(metrics_dict, prefix="Val"):
    """
    Prints metrics with full precision.

    Args:
        metrics_dict (dict): Dictionary of metric names and values.
        prefix (str): Prefix string for the log message.
    """
    msg_parts = []
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            msg_parts.append(f"{k}: {v:.20f}")
        else:
            msg_parts.append(f"{k}: {v}")

    print(f"[{prefix}] " + " | ".join(msg_parts))
