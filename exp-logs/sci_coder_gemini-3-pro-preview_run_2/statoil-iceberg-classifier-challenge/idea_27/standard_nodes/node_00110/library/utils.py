import os
import random
import json
import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(
    input_dir: str = "./input",
    metadata_dir: str = "./metadata",
    cache_dir: str = "./working/idea_27",
    load_cached_data: bool = True,
):
    """
    Loads and processes the dataset. Implements caching to speed up subsequent runs.

    Constructs 3-channel images (Band 1, Band 2, Mean) as per the SWDI-Net specification.

    Args:
        input_dir (str): Directory containing raw JSON files.
        metadata_dir (str): Directory containing metadata CSVs.
        cache_dir (str): Directory to store/load cached .npz files.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary containing:
            - 'X_train': Training images (N, 75, 75, 3)
            - 'y_train': Training labels (N,)
            - 'inc_angle_train': Training incidence angles (N,)
            - 'ids_train': Training IDs (N,)
            - 'X_test': Test images (M, 75, 75, 3)
            - 'inc_angle_test': Test incidence angles (M,)
            - 'ids_test': Test IDs (M,)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "processed_data.npz")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            # Convert np.savez dictionary to standard dict
            return {key: data[key] for key in data.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    # 2. Process data from scratch
    print("Processing data from scratch...")

    train_json_path = os.path.join(input_dir, "train.json")
    test_json_path = os.path.join(input_dir, "test.json")

    if not os.path.exists(train_json_path) or not os.path.exists(test_json_path):
        raise FileNotFoundError(f"Data files not found in {input_dir}")

    # Load JSON data
    df_train = pd.read_json(train_json_path)
    df_test = pd.read_json(test_json_path)

    def process_images(df):
        """
        Reshapes bands and constructs the 3-channel image (HH, HV, Avg).
        """
        # Stack lists into numpy arrays and reshape
        # Band 1: HH, Band 2: HV
        b1 = np.stack([np.array(b) for b in df["band_1"]]).reshape(-1, 75, 75)
        b2 = np.stack([np.array(b) for b in df["band_2"]]).reshape(-1, 75, 75)

        # Construct 3rd channel: Arithmetic Mean (Band_1 + Band_2) / 2
        b3 = (b1 + b2) / 2.0

        # Stack channels: (N, 75, 75, 3)
        images = np.stack([b1, b2, b3], axis=3)
        return images.astype(np.float32)

    # Process Training Data
    print("Constructing training images...")
    X_train = process_images(df_train)
    y_train = df_train["is_iceberg"].values.astype(np.float32)
    ids_train = df_train["id"].values

    # Handle incidence angles: Convert 'na' to NaN
    inc_angle_train = pd.to_numeric(
        df_train["inc_angle"], errors="coerce"
    ).values.astype(np.float32)

    # Process Test Data
    print("Constructing test images...")
    X_test = process_images(df_test)
    ids_test = df_test["id"].values
    inc_angle_test = pd.to_numeric(df_test["inc_angle"], errors="coerce").values.astype(
        np.float32
    )

    # 3. Save to cache
    data_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "inc_angle_train": inc_angle_train,
        "ids_train": ids_train,
        "X_test": X_test,
        "inc_angle_test": inc_angle_test,
        "ids_test": ids_test,
    }

    print(f"Saving processed data to {cache_path}...")
    np.savez_compressed(cache_path, **data_dict)

    return data_dict
