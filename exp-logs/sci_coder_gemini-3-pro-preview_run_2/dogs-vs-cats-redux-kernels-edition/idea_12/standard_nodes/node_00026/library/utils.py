import os
import sys
import random
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "train"):
    """
    Creates and returns a logger instance with standard formatting.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def generate_model_soup(checkpoint_paths: list, save_path: str):
    """
    Computes the uniform average of weights from the provided checkpoint paths
    (Model Soup) and saves the resulting model state dictionary.

    Args:
        checkpoint_paths (list): List of file paths to the .pth checkpoints.
        save_path (str): Path where the soup checkpoint will be saved.

    Returns:
        dict: The averaged state dictionary.
    """
    logger = get_logger("utils")

    if not checkpoint_paths:
        logger.error("No checkpoint paths provided for model soup.")
        return None

    # Ensure the directory for the save path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_models = len(checkpoint_paths)
    logger.info(f"Generating model soup from {n_models} checkpoints...")

    # Load the first model to initialize the soup
    # Map to CPU to save GPU memory during averaging
    try:
        first_ckpt = torch.load(checkpoint_paths[0], map_location="cpu")

        # Handle cases where the checkpoint might be a dict containing 'model_state_dict'
        if isinstance(first_ckpt, dict) and "model_state_dict" in first_ckpt:
            soup_state_dict = first_ckpt["model_state_dict"]
        else:
            soup_state_dict = first_ckpt

        # Clone and convert to float to ensure precision during addition
        soup_state_dict = {k: v.clone().float() for k, v in soup_state_dict.items()}

    except Exception as e:
        logger.error(f"Failed to load first checkpoint {checkpoint_paths[0]}: {e}")
        raise e

    # Iterate over the remaining checkpoints
    for i in range(1, n_models):
        path = checkpoint_paths[i]
        logger.info(f"Processing checkpoint for soup: {path}")
        try:
            ckpt = torch.load(path, map_location="cpu")

            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt

            for key in soup_state_dict:
                if key in state_dict:
                    # Add current model's weights
                    soup_state_dict[key] += state_dict[key].float()
                else:
                    logger.warning(f"Key {key} missing in checkpoint {path}")

        except Exception as e:
            logger.error(f"Failed to load/process checkpoint {path}: {e}")
            raise e

    # Average the weights
    logger.info("Averaging weights...")
    for key in soup_state_dict:
        soup_state_dict[key] /= n_models

    # Save the souped model
    logger.info(f"Saving model soup to {save_path}")
    torch.save(soup_state_dict, save_path)

    return soup_state_dict
