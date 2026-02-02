import torch
from torch.utils.data import Dataset
from library.data_processing import process_data


class ManufacturingDataset(Dataset):
    """
    A custom Dataset for the manufacturing control data.
    Wraps processed categorical and continuous features and optional targets.
    """

    def __init__(self, cat_features, cont_features, targets=None):
        """
        Args:
            cat_features (array-like): Categorical features indices.
            cont_features (array-like): Normalized continuous features.
            targets (array-like, optional): Target labels.
        """
        # Convert to tensors. Using as_tensor to avoid copy if possible,
        # but ensuring correct dtype.
        self.cat_features = torch.as_tensor(cat_features, dtype=torch.long)
        self.cont_features = torch.as_tensor(cont_features, dtype=torch.float32)

        if targets is not None:
            self.targets = torch.as_tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.cat_features)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (categorical_features, continuous_features, target) if target exists,
                   else (categorical_features, continuous_features).
        """
        if self.targets is not None:
            return self.cat_features[idx], self.cont_features[idx], self.targets[idx]
        else:
            return self.cat_features[idx], self.cont_features[idx]


def get_datasets(
    load_cached_data=True,
    base_dir="./metadata",
    cache_dir="./working/idea_33",
    debug=False,
):
    """
    Loads processed data and returns ManufacturingDataset instances for train, val, and test.

    Args:
        load_cached_data (bool): Whether to try loading from cache.
        base_dir (str): Directory containing metadata CSVs.
        cache_dir (str): Directory for caching processed numpy arrays.
        debug (bool): If True, limits the dataset size for debugging purposes.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, vocab_sizes, test_ids)
    """
    # Load data using the provided library function
    data = process_data(
        load_cached_data=load_cached_data, base_dir=base_dir, cache_dir=cache_dir
    )

    (
        X_cat_train,
        X_cont_train,
        y_train,
        X_cat_val,
        X_cont_val,
        y_val,
        X_cat_test,
        X_cont_test,
        test_ids,
        vocab_sizes,
    ) = data

    # Debugging: slice data if requested
    if debug:
        subset_size = 1000
        print(f"Debug mode enabled. Limiting datasets to {subset_size} samples.")
        X_cat_train = X_cat_train[:subset_size]
        X_cont_train = X_cont_train[:subset_size]
        y_train = y_train[:subset_size]

        X_cat_val = X_cat_val[:subset_size]
        X_cont_val = X_cont_val[:subset_size]
        y_val = y_val[:subset_size]

        X_cat_test = X_cat_test[:subset_size]
        X_cont_test = X_cont_test[:subset_size]
        test_ids = test_ids[:subset_size]

    # Instantiate Datasets
    train_dataset = ManufacturingDataset(X_cat_train, X_cont_train, y_train)
    val_dataset = ManufacturingDataset(X_cat_val, X_cont_val, y_val)
    test_dataset = ManufacturingDataset(X_cat_test, X_cont_test)

    return train_dataset, val_dataset, test_dataset, vocab_sizes, test_ids
