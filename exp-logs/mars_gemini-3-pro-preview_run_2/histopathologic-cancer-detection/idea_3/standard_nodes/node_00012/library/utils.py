import os
import random
import numpy as np
import torch
import cv2
import pandas as pd
from library.config import Config


def seed_everything(seed: int = 42):
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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def calculate_dataset_stats(
    metadata_path: str,
    input_dir: str = Config.INPUT_DIR,
    sample_size: int = None,
    load_cached_data: bool = True,
):
    """
    Calculates the channel-wise mean and standard deviation of the dataset.
    Implements caching to avoid re-computation.

    Args:
        metadata_path (str): Path to the metadata CSV file (e.g., train.csv).
        input_dir (str): Directory containing the image files.
        sample_size (int, optional): Number of images to sample for faster calculation.
                                     If None, uses the full dataset.
        load_cached_data (bool): If True, attempts to load stats from cache.

    Returns:
        tuple: (mean, std) where each is a numpy array of shape (3,).
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_file = os.path.join(Config.CACHE_DIR, "dataset_stats.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            print(f"Loading dataset stats from cache: {cache_file}")
            stats = np.load(cache_file)
            mean, std = stats[0], stats[1]
            return mean, std
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing dataset stats from scratch...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Sample if requested
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(drop=True)
        print(f"Using a sample of {sample_size} images.")

    # Accumulators
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for idx, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        # Read image using OpenCV (BGR)
        img = cv2.imread(full_path)
        if img is None:
            continue

        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float64) / 255.0

        # Reshape to (N, 3)
        pixels = img.reshape(-1, 3)

        # Update accumulators
        channel_sum += pixels.sum(axis=0)
        channel_sq_sum += (pixels**2).sum(axis=0)
        pixel_count += pixels.shape[0]

    if pixel_count == 0:
        raise ValueError("No valid images found to compute statistics.")

    # Calculate Mean and Std
    mean = channel_sum / pixel_count
    # Std = sqrt(E[x^2] - (E[x])^2)
    std = np.sqrt((channel_sq_sum / pixel_count) - (mean**2))

    print(f"Computed Mean: {mean}")
    print(f"Computed Std: {std}")

    # 3. Save to cache
    stats = np.array([mean, std])
    np.save(cache_file, stats)
    print(f"Saved dataset stats to cache: {cache_file}")

    return mean, std
