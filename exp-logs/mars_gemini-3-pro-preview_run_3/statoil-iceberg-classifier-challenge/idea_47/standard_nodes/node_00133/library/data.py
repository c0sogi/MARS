import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

# Import pre-implemented components from the provided configuration library
# to ensure consistency and avoid code duplication.
from library.config import process_data, IcebergDataset, BATCH_SIZE, METADATA_DIR


def get_dataloaders(batch_size=BATCH_SIZE, load_cached_data=True):
    """
    Prepares and returns DataLoaders for training, validation, and testing.

    This function leverages the caching mechanism implemented in library.config.process_data.
    It recovers the fixed train/validation split defined in the metadata CSVs.

    Args:
        batch_size (int): The batch size for the DataLoaders.
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays
                                 from the cache directory.

    Returns:
        tuple: (train_loader, val_loader, test_loader, ids_test)
    """

    # 1. Load Data
    # process_data handles:
    # - Loading raw JSON
    # - Constructing 3-channel images (HH, HV, Avg)
    # - Imputing missing incidence angles
    # - Caching results to ./working/idea_47/
    # Note: X_full contains concatenated data from train.csv and val.csv
    data = process_data(load_cached_data=load_cached_data)
    X_full, y_full, angle_full, ids_full, X_test, angle_test, ids_test = data

    # 2. Recover Train/Val Split
    # The process_data function concatenates train metadata then val metadata.
    # We read the train metadata file to find the split point.
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")

    # Ensure metadata exists (it is pre-generated)
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata file not found at {train_meta_path}")

    df_train_meta = pd.read_csv(train_meta_path)
    n_train = len(df_train_meta)

    # Slice the arrays to separate training and validation sets
    X_train = X_full[:n_train]
    y_train = y_full[:n_train]
    angle_train = angle_full[:n_train]

    X_val = X_full[n_train:]
    y_val = y_full[n_train:]
    angle_val = angle_full[n_train:]

    # 3. Define Transforms
    # Apply random flips only to the training set
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # 4. Create Datasets
    # IcebergDataset is imported from library.config
    train_ds = IcebergDataset(X_train, y_train, angle_train, transform=train_transform)
    val_ds = IcebergDataset(X_val, y_val, angle_val, transform=None)
    test_ds = IcebergDataset(X_test, None, angle_test, transform=None)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader, ids_test
