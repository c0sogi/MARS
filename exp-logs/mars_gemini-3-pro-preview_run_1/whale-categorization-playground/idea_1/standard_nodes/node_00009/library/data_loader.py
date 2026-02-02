import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from library.utils import set_seed


class LabelEncoder:
    """
    Encodes string labels to integers and decodes them back.
    Supports caching of the class list to ensure consistency.
    """

    def __init__(self):
        self.classes_ = None
        self.class_to_idx = None

    def fit(self, labels, cache_dir="./working/idea_1", load_cached_data=False):
        """
        Fits the encoder on a list of labels.

        Args:
            labels (list or pd.Series): List of string labels.
            cache_dir (str): Directory to store the cached classes.
            load_cached_data (bool): Whether to try loading from cache.
        """
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "classes.npy")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading label classes from {cache_file}")
            self.classes_ = np.load(cache_file, allow_pickle=True)
        else:
            print("Computing unique label classes...")
            # Get unique classes and sort them for deterministic behavior
            unique_labels = np.unique(labels)
            self.classes_ = unique_labels

            # Save to cache
            np.save(cache_file, self.classes_)
            print(f"Saved label classes to {cache_file}")

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}

    def transform(self, labels):
        """Converts string labels to integers."""
        if self.class_to_idx is None:
            raise ValueError("LabelEncoder is not fitted yet.")

        # Handle single string or list-like
        if isinstance(labels, str):
            return self.class_to_idx[labels]
        return np.array([self.class_to_idx[l] for l in labels])

    def inverse_transform(self, indices):
        """Converts integers back to string labels."""
        if self.classes_ is None:
            raise ValueError("LabelEncoder is not fitted yet.")

        # Handle single int or tensor/array
        if isinstance(indices, (int, np.integer)):
            return self.classes_[indices]

        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()

        return self.classes_[indices]

    def num_classes(self):
        return len(self.classes_) if self.classes_ is not None else 0


class WhaleDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, label_encoder=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            root_dir (str): Root directory containing image files (e.g., ./input).
            transform (callable, optional): Optional transform to be applied on a sample.
            label_encoder (LabelEncoder, optional): Encoder to convert labels to indices.
            is_test (bool): If True, returns (image, filename). If False, returns (image, label).
        """
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.label_encoder = label_encoder
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata 'file_path' is relative to input dir (e.g., 'train/img.jpg')
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image
        # Use IMREAD_COLOR to ensure 3 channels (BGR) even if grayscale
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if image is None:
            # Fallback for missing images (should not happen based on metadata check)
            # Create a black image
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # For test, return image and filename (to map predictions)
            return image, row["Image"]
        else:
            # For train/val, return image and label index
            label_str = row["Id"]
            label_idx = self.label_encoder.transform(label_str)
            return image, label_idx


def get_dataloaders(
    data_dir="./input",
    metadata_dir="./metadata",
    batch_size=32,
    num_workers=4,
    load_cached_data=False,
    cache_dir="./working/idea_1",
    image_size=224,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        data_dir (str): Path to input data directory.
        metadata_dir (str): Path to metadata directory.
        batch_size (int): Batch size.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to use cached label encoding.
        cache_dir (str): Directory for caching.
        image_size (int): Target image size (square).

    Returns:
        train_loader, val_loader, test_loader, label_encoder
    """
    set_seed(42)

    # 1. Load Metadata
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # 2. Setup Label Encoder
    label_encoder = LabelEncoder()
    # Fit on training data IDs
    label_encoder.fit(
        df_train["Id"], cache_dir=cache_dir, load_cached_data=load_cached_data
    )

    print(f"Number of classes: {label_encoder.num_classes()}")

    # 3. Define Transforms
    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Validation/Test transforms are usually the same as train (minus augmentation if we had any)
    val_test_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )

    # 4. Create Datasets
    train_dataset = WhaleDataset(
        df_train,
        root_dir=data_dir,
        transform=train_transform,
        label_encoder=label_encoder,
        is_test=False,
    )

    val_dataset = WhaleDataset(
        df_val,
        root_dir=data_dir,
        transform=val_test_transform,
        label_encoder=label_encoder,
        is_test=False,
    )

    test_dataset = WhaleDataset(
        df_test,
        root_dir=data_dir,
        transform=val_test_transform,
        label_encoder=None,
        is_test=True,
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, label_encoder
