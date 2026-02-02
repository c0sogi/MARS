import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from library.config import Config


class PetDataset(Dataset):
    """
    PyTorch Dataset for Pet Pawpularity Prediction.
    Handles loading images, applying model-specific preprocessing via AutoImageProcessor,
    and extracting tabular metadata.
    """

    def __init__(
        self, metadata_path, model_name, augment=False, input_root=Config.INPUT_DIR
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            model_name (str): HuggingFace model name to load the specific AutoImageProcessor.
            augment (bool): If True, returns stacked tensors of original and horizontally flipped images.
            input_root (str): Root directory containing the image files.
        """
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.augment = augment
        self.input_root = input_root

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Handle Debug mode: subset data for rapid iteration
        if Config.DEBUG:
            self.df = self.df.head(100).reset_index(drop=True)

        # Initialize Image Processor
        # AutoImageProcessor ensures we use the exact normalization/resize logic as the pre-trained model
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        except Exception as e:
            # Fallback or re-raise with context
            raise RuntimeError(
                f"Failed to load AutoImageProcessor for {model_name}: {e}"
            )

        # Pre-check columns
        self.has_target = Config.TARGET_COL in self.df.columns
        self.meta_cols = Config.META_FEATURES

        # Verify metadata columns exist
        missing_cols = [c for c in self.meta_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing metadata columns in {metadata_path}: {missing_cols}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # file_path in CSV is relative (e.g., "train/id.jpg")
        img_path = os.path.join(self.input_root, row[Config.PATH_COL])

        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # In case of corruption or missing file (though verification script should catch this)
            # Return a blank image to prevent crashing during training
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # 2. Process Image (with optional augmentation)
        pixel_values = self._process_image(image)

        # 3. Extract Metadata
        # Convert binary columns to float tensor
        meta_features = torch.tensor(
            row[self.meta_cols].values.astype("float32"), dtype=torch.float32
        )

        # 4. Construct Sample
        sample = {
            "id": row[Config.ID_COL],
            "pixel_values": pixel_values,
            "meta_features": meta_features,
        }

        if self.has_target:
            target = torch.tensor(row[Config.TARGET_COL], dtype=torch.float32)
            sample["target"] = target

        return sample

    def _process_image(self, image):
        """
        Applies processor to image. Handles augmentation logic.

        Returns:
            torch.Tensor: Shape (C, H, W) if augment=False, or (2, C, H, W) if augment=True.
        """
        if self.augment:
            # Create flipped version
            image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Process both
            # return_tensors="pt" returns shape (1, C, H, W), we take [0] to get (C, H, W)
            inputs_orig = self.processor(images=image, return_tensors="pt")[
                "pixel_values"
            ][0]
            inputs_flip = self.processor(images=image_flip, return_tensors="pt")[
                "pixel_values"
            ][0]

            # Stack them: (2, C, H, W)
            return torch.stack([inputs_orig, inputs_flip])
        else:
            # Process single image
            inputs = self.processor(images=image, return_tensors="pt")["pixel_values"][
                0
            ]
            return inputs


def get_dataloader(
    metadata_path,
    model_name,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    augment=False,
    num_workers=Config.NUM_WORKERS,
):
    """
    Factory function to create a DataLoader.

    Args:
        metadata_path (str): Path to the CSV file.
        model_name (str): Name of the backbone model (e.g. Config.MODEL_SIGLIP).
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        augment (bool): Whether to apply flip augmentation (returns stacked tensors).
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    dataset = PetDataset(
        metadata_path=metadata_path,
        model_name=model_name,
        augment=augment,
        input_root=Config.INPUT_DIR,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
    )

    return loader
