import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import Tokenizer, AttributeNormalizer, parse_inchi_attributes


def get_transforms(config: Config, mode: str = "train"):
    """
    Returns the image transformations for the given mode using Albumentations.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                # Standard normalization using ImageNet stats
                A.Normalize(mean=config.MEAN, std=config.STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                A.Normalize(mean=config.MEAN, std=config.STD),
                ToTensorV2(),
            ]
        )


def get_cached_attributes(
    config: Config, df: pd.DataFrame, split: str, load_cached_data: bool = True
):
    """
    Computes or loads cached attribute vectors for the dataset.

    Args:
        config: Configuration object.
        df: Dataframe containing InChI strings.
        split: 'train' or 'val'.
        load_cached_data: Whether to try loading from cache.

    Returns:
        np.ndarray: Raw attribute vectors (N, num_attributes).
    """
    cache_path = config.TRAIN_ATTR_CACHE if split == "train" else config.VAL_ATTR_CACHE

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        return np.load(cache_path)

    # 2. Compute from scratch if cache miss or force reload
    attributes = []
    # Ensure we are working with a list of strings
    inchi_list = df["InChI"].astype(str).tolist()

    for inchi in inchi_list:
        attr = parse_inchi_attributes(inchi, config.TRACKED_ATOMS)
        attributes.append(attr)

    attributes = np.array(attributes, dtype=np.float32)

    # 3. Save to cache for future runs
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, attributes)

    return attributes


class ChemicalDataset(Dataset):
    """
    PyTorch Dataset for chemical structure images and InChI labels.
    """

    def __init__(
        self,
        config: Config,
        df: pd.DataFrame,
        tokenizer: Tokenizer,
        attributes: np.ndarray = None,
        transform=None,
        mode: str = "train",
    ):
        self.config = config
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.attributes = attributes
        self.transform = transform
        self.mode = mode
        self.root_dir = config.INPUT_DIR

    def __len__(self):
        # Support debugging on a small subset
        if self.config.DEBUG:
            return min(len(self.df), self.config.DEBUG_SAMPLE_SIZE)
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.root_dir, row["file_path"])
        image_id = row["image_id"]

        # Load Image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing images (should ideally not happen after verification)
            # Create a black image of the correct size
            image = np.zeros(
                (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        result = {"image": image, "image_id": image_id}

        # Handle Labels (Train/Val modes only)
        if self.mode in ["train", "val"]:
            inchi_text = row["InChI"]

            # Tokenize text
            token_ids = self.tokenizer.text_to_sequence(inchi_text)
            token_ids = torch.tensor(token_ids, dtype=torch.long)

            result["token_ids"] = token_ids
            result["seq_len"] = len(token_ids)
            result["original_text"] = inchi_text

            # Add Attributes if available
            if self.attributes is not None:
                attr_vec = self.attributes[idx]
                result["attributes"] = torch.tensor(attr_vec, dtype=torch.float32)

        return result


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequence padding.
    """
    batch_out = {}

    # Stack images (they are all resized to the same dim)
    images = torch.stack([item["image"] for item in batch])
    batch_out["image"] = images

    batch_out["image_id"] = [item["image_id"] for item in batch]

    # Handle training data (has token_ids)
    if "token_ids" in batch[0]:
        token_ids = [item["token_ids"] for item in batch]

        # Determine padding value. In Tokenizer.__init__, PAD_TOKEN is added first, so index is 0.
        pad_idx = 0

        # Pad sequences to max length in this batch
        padded_ids = torch.nn.utils.rnn.pad_sequence(
            token_ids, batch_first=True, padding_value=pad_idx
        )
        batch_out["token_ids"] = padded_ids

        # Store lengths for potential packing or masking
        batch_out["seq_len"] = torch.tensor(
            [item["seq_len"] for item in batch], dtype=torch.long
        )

        batch_out["original_text"] = [item["original_text"] for item in batch]

        # Stack attributes
        if "attributes" in batch[0]:
            attributes = torch.stack([item["attributes"] for item in batch])
            batch_out["attributes"] = attributes

    return batch_out


def prepare_datasets(config: Config, load_cached_data: bool = True):
    """
    Orchestrates the loading of dataframes, fitting of tokenizer/normalizer,
    and creation of Dataset objects.

    Args:
        config: Configuration object.
        load_cached_data: Whether to use cached artifacts (tokenizer, attributes).

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, tokenizer)
    """
    # 1. Load Metadata DataFrames
    df_train = pd.read_csv(config.TRAIN_METADATA)
    df_val = pd.read_csv(config.VAL_METADATA)
    df_test = pd.read_csv(config.TEST_METADATA)

    # 2. Setup Tokenizer
    tokenizer = Tokenizer(config)
    if os.path.exists(config.TOKENIZER_PATH) and load_cached_data:
        tokenizer.load(config.TOKENIZER_PATH)
    else:
        # Fit on training text only to avoid leakage
        tokenizer.fit_on_texts(df_train["InChI"].astype(str).values)
        tokenizer.save(config.TOKENIZER_PATH)

    # 3. Setup Attributes & Normalization
    # Get raw attributes (computed or loaded from cache)
    raw_train_attrs = get_cached_attributes(config, df_train, "train", load_cached_data)
    raw_val_attrs = get_cached_attributes(config, df_val, "val", load_cached_data)

    # Initialize Normalizer
    normalizer = AttributeNormalizer(config)

    # Fit Normalizer on training data
    if load_cached_data and os.path.exists(config.ATTR_STATS_CACHE):
        normalizer.load(config.ATTR_STATS_CACHE)
    else:
        normalizer.fit(raw_train_attrs)
        normalizer.save(config.ATTR_STATS_CACHE)

    # Apply Z-score normalization
    norm_train_attrs = normalizer.transform(raw_train_attrs)
    norm_val_attrs = normalizer.transform(raw_val_attrs)

    # 4. Create Dataset Objects
    train_transform = get_transforms(config, "train")
    val_transform = get_transforms(config, "val")

    train_dataset = ChemicalDataset(
        config,
        df_train,
        tokenizer,
        attributes=norm_train_attrs,
        transform=train_transform,
        mode="train",
    )

    val_dataset = ChemicalDataset(
        config,
        df_val,
        tokenizer,
        attributes=norm_val_attrs,
        transform=val_transform,
        mode="val",
    )

    test_dataset = ChemicalDataset(
        config,
        df_test,
        tokenizer,
        attributes=None,  # Test set has no ground truth attributes
        transform=val_transform,
        mode="test",
    )

    return train_dataset, val_dataset, test_dataset, tokenizer
