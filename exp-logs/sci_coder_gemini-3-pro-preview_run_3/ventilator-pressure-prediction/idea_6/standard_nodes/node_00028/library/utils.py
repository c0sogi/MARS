import os
import glob
from library.config import Config, seed_everything


def clear_cache(directory: str = None):
    """
    Detects and removes stale cache files (.npy) and model checkpoints (.pt, .pth)
    from the specified directory to ensure the pipeline uses fresh data.

    Args:
        directory (str, optional): The directory to clean. Defaults to Config.WORKING_DIR.
    """
    target_dir = directory if directory is not None else Config.WORKING_DIR

    # Ensure directory exists before attempting to clean
    if not os.path.exists(target_dir):
        # If the directory doesn't exist, there's nothing to clear.
        return

    # Patterns for files to delete: Data cache and Model checkpoints
    patterns = ["*.npy", "*.pt", "*.pth"]
    files_to_delete = []

    for pattern in patterns:
        files_to_delete.extend(glob.glob(os.path.join(target_dir, pattern)))

    if files_to_delete:
        print(
            f"System Integrity: Detected {len(files_to_delete)} stale files in '{target_dir}'. Clearing cache..."
        )
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")
        print("Cache cleared successfully.")
    else:
        print(f"System Integrity: No stale cache files found in '{target_dir}'.")
