import os
import random
import shutil
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name, log_file=None):
    """
    Creates and returns a logger that logs to both console and a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint. If is_best is True, copies it to best_model.pth.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"
):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (str): Device to map the location to (e.g., 'cpu', 'cuda').

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Handle DataParallel wrapping if necessary (remove 'module.' prefix)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # Load optimizer state
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def get_hierarchy_dicts(load_cached_data=True):
    """
    Generates mappings (dictionaries) from category_id (species) to genus_id and family_id.
    Leverages Config.get_hierarchy_mappings for parsing and caching.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache first.

    Returns:
        tuple: (species_to_genus, species_to_family)
            - species_to_genus (dict): Mapping {category_id: genus_id}
            - species_to_family (dict): Mapping {category_id: family_id}
    """
    # Use the method provided in Config to handle parsing and caching
    # This ensures we use the parquet cache defined in Config.HIERARCHY_CACHE_PATH
    df = Config.get_hierarchy_mappings(load_cached_data=load_cached_data)

    # Convert DataFrame columns to dictionaries
    species_to_genus = dict(zip(df["category_id"], df["genus_id"]))
    species_to_family = dict(zip(df["category_id"], df["family_id"]))

    # Mappings for sparse category_id <-> contiguous species_label
    species_to_label = dict(zip(df["category_id"], df["species_label"]))
    label_to_species = dict(zip(df["species_label"], df["category_id"]))

    return species_to_genus, species_to_family, species_to_label, label_to_species
