import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import parse_inchi_attributes, compute_attribute_stats
from library.tokenizer import Tokenizer


class InChiDataset(Dataset):
    """
    PyTorch Dataset for InChI prediction.
    Handles image loading, preprocessing, and label encoding (text + attributes).
    """

    def __init__(self, df, tokenizer, transform=None, attr_stats=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path' and 'InChI'/'image_id'.
            tokenizer (Tokenizer): Instance of the Tokenizer class.
            transform (A.Compose): Albumentations transforms.
            attr_stats (np.ndarray): Mean and Std of attributes for normalization. Shape (2, 7).
            is_test (bool): Whether this is the test set (returns image_id instead of labels).
        """
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.attr_stats = attr_stats
        self.is_test = is_test

        # Pre-calculate paths to avoid overhead in __getitem__
        # Config.INPUT_DIR is "./input", file_path in df is relative e.g., "train/..."
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].values
        ]

        if not self.is_test:
            self.inchi_labels = df["InChI"].values
        else:
            self.image_ids = df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        path = self.file_paths[idx]
        image = cv2.imread(path)
        if image is None:
            # Fallback for missing images (though verification script showed none)
            # Create a black image of correct size
            image = np.zeros(
                (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Basic transform if none provided
            basic_tfm = A.Compose(
                [
                    A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
            augmented = basic_tfm(image=image)
            image = augmented["image"]

        # 3. Handle Labels
        if self.is_test:
            return {"image": image, "image_id": self.image_ids[idx]}
        else:
            inchi_text = self.inchi_labels[idx]

            # Encode Text
            encoded_text = self.tokenizer.encode(inchi_text)

            # Extract and Normalize Attributes
            # raw_attrs shape: (7,) -> [C, H, O, N, S, Halogen, Length]
            raw_attrs = parse_inchi_attributes(inchi_text)

            if self.attr_stats is not None:
                mean = self.attr_stats[0]
                std = self.attr_stats[1]
                norm_attrs = (raw_attrs - mean) / std
            else:
                norm_attrs = raw_attrs

            return {
                "image": image,
                "text_seq": torch.tensor(encoded_text, dtype=torch.long),
                "attributes": torch.tensor(norm_attrs, dtype=torch.float32),
                "original_text": inchi_text,
            }


def get_transforms(image_size):
    """
    Returns albumentations transforms for training and validation/test.
    """
    # Standard ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transforms = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            # Augmentations can be added here (Flip, Rotate, etc.)
            # For chemical structures, rotations must be handled carefully,
            # but slight affine transforms or noise are okay.
            # Keeping it simple for stability first.
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return train_transforms, val_transforms


def get_dataloaders(debug=False):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.

    Args:
        debug (bool): If True, subsets data for faster debugging.

    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    # 1. Load Metadata
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    if debug:
        print("Debug mode: Subsetting data...")
        train_df = train_df.iloc[:1000]
        val_df = val_df.iloc[:500]
        test_df = test_df.iloc[:100]

    # 2. Prepare Tokenizer
    # Fit on training texts (cached)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(
        texts=train_df["InChI"].astype(str).tolist(), load_cached_data=True
    )

    # 3. Prepare Attribute Statistics
    # Compute on training set (cached)
    attr_stats = compute_attribute_stats(train_df=train_df, load_cached_data=True)

    # 4. Prepare Transforms
    train_tfm, val_tfm = get_transforms(Config.IMAGE_SIZE)

    # 5. Create Datasets
    train_dataset = InChiDataset(
        df=train_df,
        tokenizer=tokenizer,
        transform=train_tfm,
        attr_stats=attr_stats,
        is_test=False,
    )

    val_dataset = InChiDataset(
        df=val_df,
        tokenizer=tokenizer,
        transform=val_tfm,
        attr_stats=attr_stats,
        is_test=False,
    )

    test_dataset = InChiDataset(
        df=test_df,
        tokenizer=tokenizer,
        transform=val_tfm,
        attr_stats=attr_stats,  # Not needed for test inputs, but good for consistency if we wanted to predict
        is_test=True,
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, tokenizer
