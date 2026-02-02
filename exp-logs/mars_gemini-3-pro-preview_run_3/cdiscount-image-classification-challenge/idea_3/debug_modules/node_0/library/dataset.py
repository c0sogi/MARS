import os
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import MAX_IMAGES_PER_PRODUCT


class CachedFeatureDataset(Dataset):
    """
    PyTorch Dataset wrapper for pre-computed features.
    Handles variable-length image sequences via padding and masking.
    Converts cached float16 features to float32 for training.
    """

    def __init__(self, features, indices, labels=None, max_len=MAX_IMAGES_PER_PRODUCT):
        """
        Args:
            features (np.ndarray): Flattened feature array [Total_Images, Feature_Dim].
            indices (np.ndarray): Index array [Num_Products, 2] containing (start_index, count).
            labels (np.ndarray, optional): Array of class indices [Num_Products].
            max_len (int): Maximum number of images per product to serve.
        """
        self.features = features
        self.indices = indices
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start, count = self.indices[idx]

        # Retrieve the bag of features for this product
        # Convert float16 (storage format) to float32 (training format)
        bag = self.features[start : start + count].astype(np.float32)

        L, D = bag.shape

        # Prepare padding and mask
        if L >= self.max_len:
            # Truncate if we have more images than max_len
            bag = bag[: self.max_len]
            mask = np.ones(self.max_len, dtype=np.float32)
        else:
            # Pad with zeros if we have fewer images
            padding = np.zeros((self.max_len - L, D), dtype=np.float32)
            bag = np.concatenate([bag, padding], axis=0)

            # Mask: 1 for real images, 0 for padding
            mask = np.concatenate(
                [
                    np.ones(L, dtype=np.float32),
                    np.zeros(self.max_len - L, dtype=np.float32),
                ],
                axis=0,
            )

        bag_tensor = torch.from_numpy(bag)
        mask_tensor = torch.from_numpy(mask)

        if self.labels is not None:
            # Return label as LongTensor for CrossEntropyLoss
            label = self.labels[idx]
            return bag_tensor, mask_tensor, torch.tensor(label, dtype=torch.long)

        return bag_tensor, mask_tensor


def get_class_weights(labels, num_classes, cache_path=None, load_cached_data=True):
    """
    Computes or loads inverse frequency class weights to handle class imbalance.

    Args:
        labels (np.ndarray): Array of mapped class indices (0 to num_classes-1).
        num_classes (int): Total number of classes.
        cache_path (str, optional): Path to save/load the computed weights.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.Tensor: Float tensor of shape [num_classes].
    """
    # 1. Attempt to load from cache
    if load_cached_data and cache_path is not None:
        if os.path.exists(cache_path):
            try:
                weights_np = np.load(cache_path)
                return torch.from_numpy(weights_np).float()
            except Exception:
                # If load fails, proceed to recompute
                pass

    # 2. Compute weights from scratch
    # Ensure labels are integers for bincount
    labels = labels.astype(np.int64)

    # Count occurrences of each class index
    counts = np.bincount(labels, minlength=num_classes)

    # Calculate inverse frequency weights
    # Add 1.0 to denominator to handle potential missing classes in the split safely
    weights_np = 1.0 / (counts + 1.0)

    # Normalize weights so that the mean weight is 1.0
    # This keeps the scale of the loss similar to the unweighted case
    weights_np = weights_np / weights_np.mean()

    # Convert to float32
    weights_np = weights_np.astype(np.float32)

    # 3. Save to cache
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, weights_np)

    return torch.from_numpy(weights_np)
