import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Returns mixed inputs, pairs of targets, and lambda for Mixup regularization.

    Args:
        x (torch.Tensor): Input batch of images.
        y (torch.Tensor): Input batch of labels.
        alpha (float): Mixup interpolation coefficient parameter.
        device (str): Device to perform the operation on.

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_a (torch.Tensor): Labels for the first image in the pair.
        y_b (torch.Tensor): Labels for the second image in the pair.
        lam (float): The interpolation factor used.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.

    Args:
        criterion: The loss function (e.g., BCEWithLogitsLoss).
        pred: Model predictions.
        y_a: Labels for the first image.
        y_b: Labels for the second image.
        lam: Interpolation factor.

    Returns:
        loss: Weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_transforms(phase="train"):
    """
    Returns the data transformation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.

    Returns:
        transforms.Compose: The composition of transforms.
    """
    # Normalization to [-1, 1]
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]

    if phase == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ]
        )


def load_and_cache_data(
    metadata_path,
    input_dir,
    cache_prefix,
    load_cached_data=True,
    debug_subset_size=None,
):
    """
    Loads data from metadata/disk, caches it as .npy files, and returns numpy arrays.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        input_dir (str): Root directory containing the images.
        cache_prefix (str): Prefix for the cached filenames (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.
        debug_subset_size (int, optional): If set, only load this many samples.

    Returns:
        tuple: (images, ids, labels) as numpy arrays.
    """
    cache_dir = "./working/idea_4/"
    os.makedirs(cache_dir, exist_ok=True)

    imgs_path = os.path.join(cache_dir, f"{cache_prefix}_imgs.npy")
    ids_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")
    lbls_path = os.path.join(cache_dir, f"{cache_prefix}_lbls.npy")

    # Attempt to load from cache
    if (
        load_cached_data
        and os.path.exists(imgs_path)
        and os.path.exists(ids_path)
        and os.path.exists(lbls_path)
    ):
        # Check if we need to respect debug_subset_size on cached data
        imgs = np.load(imgs_path)
        ids = np.load(ids_path)
        lbls = np.load(lbls_path)

        if debug_subset_size is not None and len(imgs) > debug_subset_size:
            return (
                imgs[:debug_subset_size],
                ids[:debug_subset_size],
                lbls[:debug_subset_size],
            )
        return imgs, ids, lbls

    # Load from source
    df = pd.read_csv(metadata_path)

    if debug_subset_size is not None:
        df = df.iloc[:debug_subset_size]

    img_list = []
    id_list = []
    lbl_list = []

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_list.append(img)
        id_list.append(row["id"])
        lbl_list.append(row["has_cactus"])

    # Convert to numpy arrays
    imgs = np.array(img_list, dtype=np.uint8)
    ids = np.array(id_list)
    lbls = np.array(lbl_list, dtype=np.float32)

    # Save to cache (only if we processed the full requested set, or if debugging we save the debug set)
    # To be safe with the caching logic, we generally overwrite cache if we re-computed.
    np.save(imgs_path, imgs)
    np.save(ids_path, ids)
    np.save(lbls_path, lbls)

    return imgs, ids, lbls


class CactusDataset(Dataset):
    def __init__(self, images, labels, image_ids, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray): Array of labels (N,).
            image_ids (np.ndarray): Array of image IDs (N,).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img_id = self.image_ids[idx]

        if self.transform:
            img = self.transform(img)

        # Return image tensor, label tensor (float for BCE), and ID
        return img, torch.tensor(label, dtype=torch.float32), img_id
