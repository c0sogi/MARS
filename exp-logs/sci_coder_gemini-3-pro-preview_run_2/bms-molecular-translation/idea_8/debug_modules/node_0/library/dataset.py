import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.tokenizer import InChiTokenizer


class InChiDataset(Dataset):
    """
    PyTorch Dataset for loading InChI images and labels.
    """

    def __init__(self, csv_path, mode="train", transform=None, sample_size=None):
        """
        Args:
            csv_path (str): Path to the metadata CSV file (train.csv, val.csv, or test.csv).
            mode (str): Operation mode - 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            sample_size (int, optional): Limit the dataset size for debugging purposes.
        """
        self.mode = mode
        self.transform = transform
        self.input_dir = Config.INPUT_DIR
        self.img_height = Config.IMG_HEIGHT

        # Load metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Debugging: subset the data if requested
        if sample_size is not None:
            self.df = self.df.head(sample_size).copy()

        # Initialize tokenizer
        self.tokenizer = InChiTokenizer()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        rel_path = row["file_path"]

        # Construct full image path
        full_path = os.path.join(self.input_dir, rel_path)

        # 1. Load Image (Grayscale)
        # InChI images are binary/grayscale, color is irrelevant.
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for corrupt/missing images: create a blank white image
            # This prevents the dataloader from crashing during training
            image = np.ones((self.img_height, self.img_height), dtype=np.uint8) * 255

        # 2. Preprocessing
        # Resize to fixed height, maintaining aspect ratio
        h, w = image.shape
        new_h = self.img_height
        new_w = int(w * (new_h / h))

        # Ensure width is at least 1 pixel
        new_w = max(1, new_w)

        image = cv2.resize(image, (new_w, new_h))

        # Normalize to [0, 1] range
        image = image.astype(np.float32) / 255.0

        # Convert to tensor: (H, W) -> (1, H, W)
        image_tensor = torch.from_numpy(image).unsqueeze(0)

        result = {"image": image_tensor, "image_id": image_id, "width": new_w}

        # 3. Process Label (if available)
        if self.mode in ["train", "val"]:
            inchi_text = row["InChI"]
            # Tokenize text to integer sequence
            target_sequence = self.tokenizer.text_to_sequence(inchi_text)

            result["target"] = target_sequence
            result["target_text"] = inchi_text

        return result


class CollateFn:
    """
    Custom collate function to handle variable-width images and variable-length sequences.
    """

    def __call__(self, batch):
        batch_size = len(batch)

        # ---------------------------------------------------------------------
        # 1. Collate Images (Dynamic Padding)
        # ---------------------------------------------------------------------
        # Determine the maximum width in this batch
        widths = [item["width"] for item in batch]
        max_width = max(widths)
        img_h = batch[0]["image"].size(1)  # Should be Config.IMG_HEIGHT

        # Create a batch tensor initialized with 1.0 (white background padding)
        # Shape: (Batch Size, Channels, Height, Max Width)
        padded_images = torch.ones(batch_size, 1, img_h, max_width, dtype=torch.float32)

        for i, item in enumerate(batch):
            w = item["width"]
            # Copy the image into the padded tensor
            padded_images[i, :, :, :w] = item["image"]

        # Input lengths (width in pixels) needed for some models/losses
        input_lengths = torch.tensor(widths, dtype=torch.long)

        result = {
            "images": padded_images,
            "input_lengths": input_lengths,
            "image_ids": [item["image_id"] for item in batch],
        }

        # ---------------------------------------------------------------------
        # 2. Collate Targets (if present)
        # ---------------------------------------------------------------------
        if "target" in batch[0]:
            targets = [item["target"] for item in batch]
            lengths = [len(t) for t in targets]
            max_target_len = max(lengths)

            # Create padded targets tensor initialized with 0 (blank token)
            # Shape: (Batch Size, Max Target Length)
            padded_targets = torch.zeros(batch_size, max_target_len, dtype=torch.long)

            for i, t in enumerate(targets):
                seq_len = len(t)
                if seq_len > 0:
                    padded_targets[i, :seq_len] = t

            target_lengths = torch.tensor(lengths, dtype=torch.long)

            result["targets"] = padded_targets
            result["target_lengths"] = target_lengths
            result["target_texts"] = [item["target_text"] for item in batch]

        return result
