import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from library.config import Config


class LeafDataset(Dataset):
    """
    Dataset class for Leaf Classification that provides multi-view image data
    and tabular features for each sample.

    Returns in __getitem__:
        images: Tensor of shape (4, 3, H, W) representing 4 rotated views (0, 90, 180, 270).
        features: Tensor of shape (192,) representing raw tabular features.
        label: LongTensor scalar (class index) if train/val, else -1.
        image_id: Integer ID of the image.
    """

    def __init__(self, df, label_encoder=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.is_test = is_test
        self.label_encoder = label_encoder

        # Identify feature columns based on prefixes defined in the dataset description
        self.feature_cols = [
            c
            for c in self.df.columns
            if c.startswith("margin")
            or c.startswith("shape")
            or c.startswith("texture")
        ]

        # Pre-load paths and ids to avoid dataframe lookups in __getitem__
        self.image_paths = self.df["file_path"].values
        self.ids = self.df["id"].values

        if not self.is_test:
            self.species = self.df["species"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load and Preprocess Image
        rel_path = self.image_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read image (Binary images are usually single channel, but we convert to RGB for backbones)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        # Handle edge case if image read fails (though metadata check ensures existence)
        if img is None:
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)

        # Resize to target size
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

        # Convert to RGB (3 channels) by replicating the grayscale channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0

        # 2. Generate 4 Rotated Views
        # We generate views at 0, 90, 180, 270 degrees to enforce rotational invariance via averaging later
        views = []
        mean = np.array(Config.IMAGENET_MEAN, dtype=np.float32)
        std = np.array(Config.IMAGENET_STD, dtype=np.float32)

        for angle in Config.ROTATION_ANGLES:
            if angle == 0:
                view = img.copy()
            elif angle == 90:
                view = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                view = cv2.rotate(img, cv2.ROTATE_180)
            elif angle == 270:
                view = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                view = img.copy()

            # Apply ImageNet Normalization
            view = (view - mean) / std

            # Convert HWC to CHW format
            view = view.transpose(2, 0, 1)
            views.append(view)

        # Stack views into a single tensor: (4, 3, H, W)
        images_tensor = torch.tensor(np.stack(views), dtype=torch.float32)

        # 3. Extract Tabular Features
        features = self.df.iloc[idx][self.feature_cols].values.astype(np.float32)
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # 4. Get ID and Label
        image_id = self.ids[idx]

        if self.is_test:
            return images_tensor, features_tensor, -1, image_id
        else:
            label_str = self.species[idx]
            label = self.label_encoder.transform([label_str])[0]
            return (
                images_tensor,
                features_tensor,
                torch.tensor(label, dtype=torch.long),
                image_id,
            )


def load_data(debug=False):
    """
    Loads metadata CSVs, fits the LabelEncoder, and initializes Datasets.

    Args:
        debug (bool): If True, limits the dataset size for debugging.

    Returns:
        train_dataset (LeafDataset)
        val_dataset (LeafDataset)
        test_dataset (LeafDataset)
        le (LabelEncoder): Fitted label encoder.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Handle Debugging / Sample Size
    limit = (
        Config.DEBUG_SAMPLE_SIZE
        if Config.DEBUG_SAMPLE_SIZE is not None
        else (50 if debug else None)
    )
    if limit:
        train_df = train_df.iloc[:limit]
        val_df = val_df.iloc[:limit]
        test_df = test_df.iloc[:limit]

    # Fit Label Encoder
    # We combine train and val species to ensure the encoder knows all possible classes
    all_species = pd.concat([train_df["species"], val_df["species"]]).unique()
    all_species = np.sort(all_species)  # Sort for deterministic index mapping

    le = LabelEncoder()
    le.fit(all_species)

    # Cache the classes for consistency during inference/submission
    Config.setup()  # Ensure directories exist
    np.save(Config.get_cache_path(Config.CACHE_CLASSES), le.classes_)

    # Initialize Datasets
    train_dataset = LeafDataset(train_df, label_encoder=le, is_test=False)
    val_dataset = LeafDataset(val_df, label_encoder=le, is_test=False)
    test_dataset = LeafDataset(test_df, label_encoder=None, is_test=True)

    return train_dataset, val_dataset, test_dataset, le
