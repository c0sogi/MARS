import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import (
    extract_attributes,
    compute_attribute_stats,
    normalize_attributes,
)
from library.tokenizer import get_tokenizer


def get_transforms(phase: str):
    """
    Returns the image transformations for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                # Augmentations to improve robustness
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def preprocess_metadata(
    path: str, mode: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads metadata and pre-calculates attributes for training/validation sets.
    Implements caching using Parquet files.

    Args:
        path (str): Path to the raw metadata CSV.
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    # Construct cache path
    filename = os.path.basename(path).replace(".csv", "_processed.parquet")
    cache_path = os.path.join(Config.WORKING_DIR, filename)

    # Ensure directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            print(f"[{mode}] Loaded processed metadata from {cache_path}")
            return df
        except Exception as e:
            print(f"[{mode}] Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"[{mode}] Processing metadata from {path}...")
    df = pd.read_csv(path)

    # Extract attributes only for train/val where ground truth is available
    if mode in ["train", "val"]:
        print(f"[{mode}] Extracting attributes for {len(df)} samples...")

        # We assume extract_attributes returns a numpy array of shape (NUM_ATTRIBUTES,)
        # We will expand this into separate columns for the dataframe

        # Apply extraction
        # Using a loop or apply might be slow for huge datasets, but it's deterministic.
        # For 1.5M rows, this might take a few minutes.
        # We use a list comprehension for better performance than df.apply
        inchi_list = df["InChI"].astype(str).tolist()
        attrs_list = [extract_attributes(s) for s in inchi_list]

        # Convert to a matrix
        attrs_matrix = np.vstack(attrs_list)

        # Add to dataframe
        for i in range(Config.NUM_ATTRIBUTES):
            df[f"attr_{i}"] = attrs_matrix[:, i]

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        print(f"[{mode}] Saved processed metadata to {cache_path}")
    except Exception as e:
        print(f"[{mode}] Warning: Failed to save cache: {e}")

    return df


class ChemicalDataset(Dataset):
    def __init__(
        self, df, tokenizer, transform, attr_mean=None, attr_std=None, mode="train"
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            tokenizer (InChITokenizer): Tokenizer instance.
            transform (A.Compose): Albumentations transforms.
            attr_mean (np.ndarray): Global mean of attributes (for normalization).
            attr_std (np.ndarray): Global std of attributes.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.mode = mode

        # Convert stats to torch tensors if provided
        if attr_mean is not None:
            self.attr_mean = torch.tensor(attr_mean, dtype=torch.float32)
            self.attr_std = torch.tensor(attr_std, dtype=torch.float32)
        else:
            self.attr_mean = None
            self.attr_std = None

        # Pre-fetch file paths to avoid overhead
        self.file_paths = df["file_path"].values
        self.image_ids = df["image_id"].values

        # Pre-fetch attributes if available
        self.has_attributes = False
        if mode in ["train", "val"]:
            # Check if attribute columns exist
            attr_cols = [f"attr_{i}" for i in range(Config.NUM_ATTRIBUTES)]
            if all(col in df.columns for col in attr_cols):
                self.attributes = df[attr_cols].values.astype(np.float32)
                self.has_attributes = True
                self.inchi_strs = df["InChI"].values
            else:
                # Should not happen if preprocess_metadata is used
                raise ValueError(
                    "Attributes not found in dataframe. Use preprocess_metadata."
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (though verification passed)
            # Create a black image
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # 3. Handle Targets
        if self.mode in ["train", "val"]:
            # Attributes
            raw_attrs = torch.tensor(self.attributes[idx], dtype=torch.float32)

            # Normalize attributes
            if self.attr_mean is not None and self.attr_std is not None:
                norm_attrs = normalize_attributes(
                    raw_attrs, self.attr_mean, self.attr_std
                )
            else:
                norm_attrs = raw_attrs

            # Sequence
            inchi_text = self.inchi_strs[idx]
            seq_list = self.tokenizer.text_to_sequence(
                inchi_text, max_len=Config.MAX_LEN, padding=True
            )
            seq_tensor = torch.tensor(seq_list, dtype=torch.long)

            # Calculate actual length (excluding padding, but including SOS/EOS)
            # SOS is at start, EOS is somewhere.
            # If padded, EOS is at index before padding starts.
            # text_to_sequence adds EOS.
            # We can find EOS index.
            try:
                eos_pos = seq_list.index(self.tokenizer.stoi[Config.EOS_TOKEN])
                seq_len = eos_pos + 1  # Include EOS
            except ValueError:
                seq_len = len(seq_list)

            return {
                "image": image,
                "attributes": norm_attrs,
                "seq": seq_tensor,
                "seq_len": torch.tensor(seq_len, dtype=torch.long),
            }

        else:
            # Test mode
            return {"image": image, "image_id": self.image_ids[idx]}


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Factory function to prepare DataLoaders.

    Args:
        load_cached_data (bool): Use cached metadata/stats.
        debug (bool): If True, use a small subset of data.

    Returns:
        tuple: (train_loader, val_loader, test_loader, tokenizer)
    """
    # 1. Setup Tokenizer
    tokenizer = get_tokenizer(load_cached_data=load_cached_data)

    # 2. Compute/Load Attribute Statistics
    # We pass None to df to let utils load the raw metadata if needed
    attr_mean, attr_std = compute_attribute_stats(
        df=None, load_cached_data=load_cached_data
    )

    # 3. Load and Preprocess Metadata
    train_df = preprocess_metadata(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_df = preprocess_metadata(Config.VAL_METADATA_PATH, "val", load_cached_data)
    test_df = preprocess_metadata(Config.TEST_METADATA_PATH, "test", load_cached_data)

    # 4. Debugging Subset
    if debug:
        print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} rows per split.")
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 5. Create Datasets
    train_dataset = ChemicalDataset(
        train_df, tokenizer, get_transforms("train"), attr_mean, attr_std, mode="train"
    )
    val_dataset = ChemicalDataset(
        val_df, tokenizer, get_transforms("val"), attr_mean, attr_std, mode="val"
    )
    test_dataset = ChemicalDataset(
        test_df,
        tokenizer,
        get_transforms("test"),
        attr_mean=None,
        attr_std=None,
        mode="test",
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
