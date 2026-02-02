import torch
from torch.utils.data import DataLoader
from library.mlp_architecture import PizzaDataset

# Re-export PizzaDataset to make it accessible from this module
__all__ = ["PizzaDataset", "create_dataloader"]


def create_dataloader(
    features_dict,
    targets=None,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    max_samples=None,
):
    """
    Factory function to create a PyTorch DataLoader for the PizzaDataset.

    Args:
        features_dict (dict): Dictionary containing feature arrays (embeddings, dense features, etc.).
                              Expected keys: 'title_emb', 'body_emb', 'history_emb',
                              'history_mask', 'history_centroid', 'dense_features'.
        targets (array-like, optional): Target labels corresponding to the features.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data (typically True for training, False for validation/test).
        num_workers (int): Number of subprocesses to use for data loading.
        max_samples (int, optional): If provided, truncates the dataset to the first `max_samples`.
                                     Useful for debugging or quick testing.

    Returns:
        DataLoader: A configured PyTorch DataLoader instance.
    """
    # Handle debugging/subsetting if max_samples is specified
    if max_samples is not None:
        # Slice each feature array in the dictionary
        features_dict = {k: v[:max_samples] for k, v in features_dict.items()}

        # Slice targets if they exist
        if targets is not None:
            targets = targets[:max_samples]

    # Instantiate the Dataset class from the library
    dataset = PizzaDataset(features_dict, targets=targets)

    # Create and return the DataLoader
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return loader
