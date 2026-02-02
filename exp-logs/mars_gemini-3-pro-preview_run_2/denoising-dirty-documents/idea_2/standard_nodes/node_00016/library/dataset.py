import os
import torch
from torch.utils.data import DataLoader
from library.model import DenoisingDataset, load_and_cache_data
from library.utils import set_seed


def get_dataloaders(
    metadata_dir="./metadata",
    cache_dir="./working/idea_2/cache",
    input_dir="./input",
    batch_size=16,
    patch_size=128,
    patches_per_image=4,
    num_workers=2,
    load_cached=True,
    seed=42,
):
    """
    Prepares the dataloaders and data lists for training and inference.

    This function handles the loading of metadata, caching of preprocessed images,
    and creation of the PyTorch DataLoader for the training set.

    Args:
        metadata_dir (str): Directory containing train.csv, val.csv, test.csv.
        cache_dir (str): Directory to store cached .npy files.
        input_dir (str): Directory containing raw images.
        batch_size (int): Batch size for training.
        patch_size (int): Size of patches for training.
        patches_per_image (int): Number of patches to extract per image per epoch.
        num_workers (int): Number of worker threads for DataLoader.
        load_cached (bool): Whether to attempt loading from cache.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, val_data, test_data)
            - train_loader: DataLoader for the training set (yields batches of patches).
            - val_data: List of dictionaries containing full validation images.
            - test_data: List of dictionaries containing full test images.
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Define metadata paths
    train_meta = os.path.join(metadata_dir, "train.csv")
    val_meta = os.path.join(metadata_dir, "val.csv")
    test_meta = os.path.join(metadata_dir, "test.csv")

    # Load data
    # The load_and_cache_data function from library.model handles the caching logic:
    # It checks if .npy files exist in cache_dir. If so (and load_cached=True), it loads them.
    # Otherwise, it reads from input_dir, normalizes, and saves to cache_dir.
    print(f"Loading training data from {train_meta}...")
    train_data = load_and_cache_data(
        train_meta, cache_dir, input_dir, load_cached=load_cached
    )

    print(f"Loading validation data from {val_meta}...")
    val_data = load_and_cache_data(
        val_meta, cache_dir, input_dir, load_cached=load_cached
    )

    print(f"Loading test data from {test_meta}...")
    test_data = load_and_cache_data(
        test_meta, cache_dir, input_dir, load_cached=load_cached
    )

    # Instantiate Dataset for Training
    # We use augment=True to enable random cropping and geometric transformations
    train_dataset = DenoisingDataset(
        data=train_data,
        patch_size=patch_size,
        augment=True,
        patches_per_image=patches_per_image,
    )

    # Create DataLoader for Training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )

    # Note: val_data and test_data are returned as lists of dictionaries.
    # The provided library.model.train_model and generate_submission functions
    # are designed to iterate over these lists directly to perform full-image
    # tiled inference, rather than using a DataLoader which would typically
    # require resizing or collation that destroys full-resolution details.

    return train_loader, val_data, test_data
