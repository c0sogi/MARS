import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class LeafDataset(Dataset):
    """
    Dataset class for Leaf Species Classification.
    Loads binary leaf images, generates 4 canonical rotated views, and loads tabular features.
    """

    def __init__(
        self, csv_path, root_dir, class_to_idx=None, transform=None, max_samples=None
    ):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            root_dir (str): Root directory for relative file paths (usually ./input).
            class_to_idx (dict, optional): Mapping from species name to integer label.
            transform (callable, optional): Optional transform (not used for rotation logic).
            max_samples (int, optional): Limit dataset size for debugging.
        """
        self.df = pd.read_csv(csv_path)
        if max_samples is not None:
            self.df = self.df.iloc[:max_samples]

        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

        # Identify tabular feature columns
        self.margin_cols = [c for c in self.df.columns if c.startswith("margin")]
        self.shape_cols = [c for c in self.df.columns if c.startswith("shape")]
        self.texture_cols = [c for c in self.df.columns if c.startswith("texture")]
        self.feature_cols = self.margin_cols + self.shape_cols + self.texture_cols

        # Normalization constants (ImageNet)
        self.mean = torch.tensor(Config.IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(Config.IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Image Processing ---
        # Construct full path. row['file_path'] is relative, e.g., "images/123.jpg"
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image (Grayscale since binary)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback (should not occur with valid metadata)
            img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8)

        # Resize
        img = cv2.resize(
            img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE), interpolation=cv2.INTER_AREA
        )

        # Convert to RGB (3 channels) by replicating channels
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # To Tensor [0, 1]
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # Shape: (3, H, W)

        # Generate 4 Rotated Views and Normalize
        views = []
        for k in range(4):  # 0, 1, 2, 3 correspond to 0, 90, 180, 270 degrees
            if k == 0:
                view = img_tensor
            else:
                # Rotate spatial dimensions (H, W) which are dims 1 and 2
                view = torch.rot90(img_tensor, k, dims=[1, 2])

            # Normalize
            view = (view - self.mean) / self.std
            views.append(view)

        # Stack views -> Shape: (4, 3, H, W)
        stacked_views = torch.stack(views)

        # --- Tabular Processing ---
        tabular_features = row[self.feature_cols].values.astype(np.float32)
        tabular_tensor = torch.from_numpy(tabular_features)

        # --- Label & ID ---
        image_id = row["id"]
        label = -1
        if self.class_to_idx is not None and "species" in row:
            label = self.class_to_idx.get(row["species"], -1)

        return stacked_views, tabular_tensor, label, image_id


def get_dataloaders(
    train_csv=Config.TRAIN_CSV,
    val_csv=Config.VAL_CSV,
    test_csv=Config.TEST_CSV,
    root_dir=Config.INPUT_DIR,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    max_samples=Config.MAX_SAMPLES,
):
    """
    Constructs DataLoaders for train, validation, and test sets.

    Returns:
        train_loader, val_loader, test_loader, classes
    """
    # Determine classes from training data to ensure consistent mapping
    train_df = pd.read_csv(train_csv)
    classes = sorted(train_df["species"].unique().tolist())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    # Instantiate Datasets
    train_dataset = LeafDataset(
        csv_path=train_csv,
        root_dir=root_dir,
        class_to_idx=class_to_idx,
        max_samples=max_samples,
    )

    val_dataset = LeafDataset(
        csv_path=val_csv,
        root_dir=root_dir,
        class_to_idx=class_to_idx,
        max_samples=max_samples,
    )

    test_dataset = LeafDataset(
        csv_path=test_csv,
        root_dir=root_dir,
        class_to_idx=None,  # No labels for test set
        max_samples=max_samples,
    )

    # Instantiate DataLoaders
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

    return train_loader, val_loader, test_loader, classes
