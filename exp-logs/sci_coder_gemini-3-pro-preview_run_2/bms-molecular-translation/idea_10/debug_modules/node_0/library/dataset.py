import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.tokenizer import InChiTokenizer


class ChemicalImageDataset(Dataset):
    """
    Dataset for loading chemical images and InChI labels.
    Performs anisotropic resizing (fixed height, variable width) to support
    sequence modeling of variable length molecules.
    """

    def __init__(self, config: Config, tokenizer: InChiTokenizer, mode: str = "train"):
        """
        Args:
            config (Config): Configuration object containing paths and hyperparameters.
            tokenizer (InChiTokenizer): Tokenizer for encoding InChI strings.
            mode (str): 'train', 'val', or 'test'. Determines which metadata file to load.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.mode = mode

        # Determine which metadata file to load
        if mode == "train":
            self.metadata_path = config.train_metadata_path
        elif mode == "val":
            self.metadata_path = config.val_metadata_path
        elif mode == "test":
            self.metadata_path = config.test_metadata_path
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'."
            )

        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

        # Handle Debug Mode
        if config.debug:
            # Subset the dataset to a small number of samples for debugging
            subset_size = min(len(self.df), config.batch_size * 10)
            self.df = self.df.head(subset_size)
            if mode == "train":  # Only print for train to avoid clutter
                print(
                    f"Debug mode enabled: Subsetting {mode} dataset to {len(self.df)} samples."
                )

        # Check if labels are available in this dataset split
        self.has_labels = "InChI" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve metadata for the current sample
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        rel_path = row["file_path"]

        # Construct full image path
        full_path = os.path.join(self.config.input_dir, rel_path)

        # Load Image
        # We use cv2.IMREAD_GRAYSCALE because the model expects 1-channel input
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Graceful fallback for missing images (should not happen with validated metadata)
            # Create a white placeholder image
            image = np.full(
                (self.config.image_height, self.config.image_height),
                255,
                dtype=np.uint8,
            )
            print(f"Warning: Failed to load image at {full_path}. Using placeholder.")

        # Anisotropic Resizing
        # Resize image to a fixed height while maintaining aspect ratio.
        # This allows the width to vary, preserving horizontal resolution for long molecules.
        h, w = image.shape
        target_h = self.config.image_height

        if h > 0:
            scale = target_h / h
            target_w = int(w * scale)
            # Ensure width is at least 1 pixel
            target_w = max(1, target_w)
            image = cv2.resize(
                image, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
        else:
            # Fallback for degenerate images
            target_w = target_h
            image = cv2.resize(image, (target_w, target_h))

        # Normalization
        # Convert to float32 and normalize to [0, 1] range.
        # Background is white (255 -> 1.0), text is black (0 -> 0.0).
        image = image.astype(np.float32) / 255.0

        # Convert to Tensor and add Channel Dimension
        # Shape becomes (1, H, W)
        image_tensor = torch.from_numpy(image).unsqueeze(0)

        # Encode Label
        label_tensor = torch.tensor([], dtype=torch.long)
        if self.has_labels:
            inchi_text = row["InChI"]
            label_tensor = self.tokenizer.encode(inchi_text)

        return {
            "image": image_tensor,
            "label": label_tensor,
            "image_id": image_id,
            "width": target_w,  # Return width for batch padding in collate_fn
        }


class ChemicalCollate:
    """
    Custom collate callable to handle batches of variable-width images.
    Pads images to the maximum width in the batch (aligned to model stride).
    Pads labels to the maximum sequence length in the batch.
    """

    def __init__(self, config: Config, pad_id: int):
        self.stride = config.horizontal_stride
        self.pad_id = pad_id

    def __call__(self, batch):
        # Unpack batch data
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]
        image_ids = [item["image_id"] for item in batch]
        widths = [item["width"] for item in batch]

        # ---------------------------------------------------------
        # 1. Pad Images
        # ---------------------------------------------------------
        max_width = max(widths)

        # Calculate padded width: must be a multiple of the horizontal stride
        # This ensures feature map dimensions are valid after downsampling.
        padded_width = int(np.ceil(max_width / self.stride) * self.stride)

        batch_size = len(images)
        c, h, _ = images[0].shape

        # Initialize padded batch with 1.0 (white background)
        padded_images = torch.ones(
            (batch_size, c, h, padded_width), dtype=torch.float32
        )

        # Copy each image into the padded tensor (left-aligned)
        for i, img in enumerate(images):
            w = img.shape[2]
            padded_images[i, :, :, :w] = img

        # ---------------------------------------------------------
        # 2. Pad Labels
        # ---------------------------------------------------------
        # Check if we have labels (train/val mode)
        if len(labels) > 0 and labels[0].numel() > 0:
            lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
            max_len = max(lengths)

            # Initialize padded labels with PAD_ID
            padded_labels = torch.full(
                (batch_size, max_len), self.pad_id, dtype=torch.long
            )

            # Copy each label sequence
            for i, label in enumerate(labels):
                end = len(label)
                padded_labels[i, :end] = label
        else:
            # Test mode (no labels)
            padded_labels = None
            lengths = None

        return {
            "images": padded_images,
            "labels": padded_labels,
            "lengths": lengths,
            "image_ids": image_ids,
        }
