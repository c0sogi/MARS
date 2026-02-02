import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.tokenizer import InChITokenizer


class InChIDataset(Dataset):
    """
    Dataset class for InChI chemical structure recognition.
    Handles image loading, preprocessing, and label tokenization.
    """

    def __init__(self, df, root_dir, tokenizer, transform=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata (image_id, file_path, [InChI]).
            root_dir (str): Root directory for images.
            tokenizer (InChITokenizer): Tokenizer instance.
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.df = df
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.file_paths = df["file_path"].values

        # Check if labels exist (Train/Val set) or not (Test set)
        if "InChI" in df.columns:
            self.labels = df["InChI"].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        file_path = self.file_paths[idx]
        full_path = os.path.join(self.root_dir, file_path)

        # Load image
        # Config.INPUT_CHANNELS is 1, so we load in grayscale
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing images (create a black image)
            image = np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)

        # Albumentations expects (H, W, C) for normalization even if grayscale
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Manual fallback: Resize -> Normalize -> ToTensor
            image = cv2.resize(image, (Config.IMAGE_SIZE[1], Config.IMAGE_SIZE[0]))
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)

        result = {"image": image}

        if self.labels is not None:
            text = self.labels[idx]
            # Tokenize text to tensor indices
            encoded = self.tokenizer.encode(text)
            result["label"] = encoded
            result["label_len"] = len(encoded)
            result["original_text"] = text
        else:
            # For test set, return image_id for submission file creation
            result["image_id"] = self.df.iloc[idx]["image_id"]

        return result


def get_transforms(phase="train"):
    """
    Returns albumentations transforms for train/val/test phases.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    # Base transforms
    transforms = [
        A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
        # Normalize for 1 channel input
        A.Normalize(mean=(0.485,), std=(0.229,), max_pixel_value=255.0),
        ToTensorV2(),
    ]

    # Add augmentations for training if needed (currently keeping baseline simple)
    if phase == "train":
        pass

    return A.Compose(transforms)


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences in batch.
    Pads the label sequences to the maximum length in the batch.
    """
    images = []
    labels = []
    lengths = []
    original_texts = []
    image_ids = []

    has_labels = "label" in batch[0]

    for item in batch:
        images.append(item["image"])
        if has_labels:
            labels.append(item["label"])
            lengths.append(item["label_len"])
            original_texts.append(item["original_text"])
        else:
            image_ids.append(item["image_id"])

    # Stack images into (Batch, C, H, W)
    images = torch.stack(images)

    batch_dict = {"images": images}

    if has_labels:
        # Pad labels dynamically to max length in this batch
        pad_idx = Config.PAD_IDX
        # pad_sequence expects list of tensors (L_i) -> (B, Max_L)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=pad_idx
        )

        batch_dict["labels"] = labels
        batch_dict["lengths"] = torch.tensor(lengths)
        batch_dict["original_texts"] = original_texts
    else:
        batch_dict["image_ids"] = image_ids

    return batch_dict


def get_dataloaders(
    train_metadata_path=Config.TRAIN_METADATA_PATH,
    val_metadata_path=Config.VAL_METADATA_PATH,
    test_metadata_path=Config.TEST_METADATA_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        train_metadata_path (str): Path to train metadata CSV.
        val_metadata_path (str): Path to val metadata CSV.
        test_metadata_path (str): Path to test metadata CSV.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        debug (bool): If True, subsamples the dataset for debugging.
        debug_sample_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    tokenizer = InChITokenizer()

    train_loader = None
    val_loader = None
    test_loader = None

    # --- Train Loader ---
    if os.path.exists(train_metadata_path):
        df_train = pd.read_csv(train_metadata_path)
        if debug:
            df_train = df_train.sample(
                n=min(len(df_train), debug_sample_size), random_state=Config.SEED
            ).reset_index(drop=True)

        train_dataset = InChIDataset(
            df=df_train,
            root_dir=Config.TRAIN_IMG_DIR,
            tokenizer=tokenizer,
            transform=get_transforms("train"),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # --- Val Loader ---
    if os.path.exists(val_metadata_path):
        df_val = pd.read_csv(val_metadata_path)
        if debug:
            df_val = df_val.sample(
                n=min(len(df_val), debug_sample_size), random_state=Config.SEED
            ).reset_index(drop=True)

        val_dataset = InChIDataset(
            df=df_val,
            root_dir=Config.TRAIN_IMG_DIR,
            tokenizer=tokenizer,
            transform=get_transforms("val"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # --- Test Loader ---
    if os.path.exists(test_metadata_path):
        df_test = pd.read_csv(test_metadata_path)
        if debug:
            df_test = df_test.sample(
                n=min(len(df_test), debug_sample_size), random_state=Config.SEED
            ).reset_index(drop=True)

        test_dataset = InChIDataset(
            df=df_test,
            root_dir=Config.TEST_IMG_DIR,
            tokenizer=tokenizer,
            transform=get_transforms("test"),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
