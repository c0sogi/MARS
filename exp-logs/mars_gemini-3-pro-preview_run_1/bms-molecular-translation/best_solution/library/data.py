import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import parse_inchi_attributes


# Ensure reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


class Tokenizer:
    """
    Handles conversion between InChI strings and integer sequences.
    """

    def __init__(self):
        self.vocab = Config.VOCAB
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}

    def text_to_sequence(self, text, max_len=Config.MAX_SEQ_LEN):
        """
        Converts a string to a sequence of indices with SOS, EOS, and PAD tokens.
        """
        seq = [self.char_to_idx[Config.SPECIAL_TOKENS[Config.SOS_IDX]]]
        for char in text:
            if char in self.char_to_idx:
                seq.append(self.char_to_idx[char])
        seq.append(self.char_to_idx[Config.SPECIAL_TOKENS[Config.EOS_IDX]])

        # Padding
        if len(seq) < max_len:
            seq += [self.char_to_idx[Config.SPECIAL_TOKENS[Config.PAD_IDX]]] * (
                max_len - len(seq)
            )
        else:
            # Truncate and ensure EOS is at the end
            seq = seq[: max_len - 1] + [
                self.char_to_idx[Config.SPECIAL_TOKENS[Config.EOS_IDX]]
            ]

        return np.array(seq, dtype=np.int64)

    def sequence_to_text(self, seq):
        """
        Converts a sequence of indices back to a string.
        """
        chars = []
        for idx in seq:
            idx = int(idx)
            if idx == Config.PAD_IDX:
                continue
            if idx == Config.SOS_IDX:
                continue
            if idx == Config.EOS_IDX:
                break
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
        return "".join(chars)


def load_and_process_metadata(csv_path, cache_name, load_cached_data=True):
    """
    Loads metadata CSV, computes auxiliary attributes, and caches the result.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_name (str): Name for the cached parquet file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed DataFrame with attribute columns.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached metadata from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing metadata from {csv_path}")
    df = pd.read_csv(csv_path)

    # Compute attributes if InChI column exists
    if "InChI" in df.columns:
        # Apply parsing function to all InChI strings
        # This returns a Series of numpy arrays
        attrs_series = df["InChI"].apply(parse_inchi_attributes)

        # Stack into a matrix (N_samples, ATTRIBUTE_DIM)
        attr_matrix = np.stack(attrs_series.values)

        # Assign to separate columns in DataFrame for efficient Parquet storage
        for i in range(Config.ATTRIBUTE_DIM):
            df[f"attr_{i}"] = attr_matrix[:, i]

    # Save to cache
    df.to_parquet(cache_path, index=False)
    print(f"Saved processed metadata to {cache_path}")

    return df


class InChiDataset(Dataset):
    """
    PyTorch Dataset for InChI prediction.
    Loads images, applies transforms, and returns tokenized labels + attributes.
    """

    def __init__(self, df, tokenizer, transform=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (creates black image)
            image = np.zeros(
                (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode in ["train", "val"]:
            inchi_text = row["InChI"]

            # Tokenize text
            seq = self.tokenizer.text_to_sequence(inchi_text)

            # Retrieve pre-computed attributes
            attrs = [row[f"attr_{i}"] for i in range(Config.ATTRIBUTE_DIM)]
            attrs = np.array(attrs, dtype=np.float32)

            return image, torch.tensor(seq), torch.tensor(attrs)

        else:  # test mode
            image_id = row["image_id"]
            # For test, we only need the image and ID for submission
            return image, image_id


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for train/val/test.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(load_cached_data=True, debug_size=None):
    """
    Creates and returns DataLoaders for training, validation, and testing.

    Args:
        load_cached_data (bool): Whether to use cached processed metadata.
        debug_size (int, optional): If set, limits the dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader, tokenizer)
    """
    tokenizer = Tokenizer()

    # Load and process metadata
    train_df = load_and_process_metadata(
        Config.TRAIN_METADATA_PATH, "train_processed", load_cached_data
    )
    val_df = load_and_process_metadata(
        Config.VAL_METADATA_PATH, "val_processed", load_cached_data
    )
    test_df = load_and_process_metadata(
        Config.TEST_METADATA_PATH, "test_processed", load_cached_data
    )

    # Apply debug subsetting if requested
    if debug_size is not None:
        train_df = train_df.iloc[:debug_size]
        val_df = val_df.iloc[:debug_size]
        test_df = test_df.iloc[:debug_size]
        print(f"Debug mode: Reduced dataset sizes to {debug_size}")

    # Instantiate Datasets
    train_dataset = InChiDataset(
        train_df, tokenizer, transform=get_transforms("train"), mode="train"
    )
    val_dataset = InChiDataset(
        val_df, tokenizer, transform=get_transforms("val"), mode="val"
    )
    test_dataset = InChiDataset(
        test_df, tokenizer, transform=get_transforms("test"), mode="test"
    )

    # Create DataLoaders
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
