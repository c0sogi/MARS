import os
import torch
from library.config import Config, set_seed


def seed_everything(seed: int = Config.SEED):
    """
    Fixes random seeds for reproducibility across numpy, random, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    set_seed(seed)


def clean_cache(working_dir: str = Config.WORKING_DIR):
    """
    Automatically detects and deletes existing .npy, .pt, .parquet, or cache files
    at the start of execution to ensure the model trains on fresh, correctly processed data.

    Args:
        working_dir (str): The directory to clean. Defaults to Config.WORKING_DIR.
    """
    # If the directory matches the configured working directory, utilize the
    # centralized setup method which handles creation and cleaning.
    if working_dir == Config.WORKING_DIR:
        Config.setup_directories(clean_cache=True)
    else:
        # Fallback implementation for arbitrary directories
        if not os.path.exists(working_dir):
            os.makedirs(working_dir, exist_ok=True)
            return

        print(f"Cleaning cache in {working_dir}...")
        extensions = [".npy", ".pt", ".parquet", ".pth"]

        # Iterate and remove files with matching extensions
        for f in os.listdir(working_dir):
            if any(f.endswith(ext) for ext in extensions):
                try:
                    file_path = os.path.join(working_dir, f)
                    os.remove(file_path)
                    print(f"Deleted: {f}")
                except OSError as e:
                    print(f"Error deleting {f}: {e}")


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
