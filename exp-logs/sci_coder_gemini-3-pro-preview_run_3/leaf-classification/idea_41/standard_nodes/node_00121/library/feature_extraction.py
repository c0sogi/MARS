import os
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from library.config import Config
from library.utils import setup_logger, seed_everything

# Initialize logger
logger = setup_logger("feature_extraction")


class RotatedLeafDataset(Dataset):
    """
    Dataset that loads leaf images, resizes them, and generates 12 equidistant rotations.
    Returns a tensor of shape (12, 3, H, W).
    """

    def __init__(self, df, input_dir, img_size=224, angles=None):
        self.df = df
        self.input_dir = input_dir
        self.img_size = img_size
        self.angles = angles if angles is not None else [0]

        # Standard ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct full path. Metadata file_path is relative to input_dir (e.g. "images/123.jpg")
        img_path = os.path.join(self.input_dir, row["file_path"])

        try:
            # Load image and convert to RGB
            img = Image.open(img_path).convert("RGB")

            # Resize
            img = img.resize((self.img_size, self.img_size), Image.Resampling.BICUBIC)

            rotated_tensors = []
            for angle in self.angles:
                # Rotate image. Fill background with white (255, 255, 255) as these are leaves on white bg.
                # TF.rotate handles PIL images directly.
                r_img = TF.rotate(img, angle, fill=(255, 255, 255))

                # Transform to tensor and normalize
                t_img = self.to_tensor(r_img)
                t_img = self.normalize(t_img)
                rotated_tensors.append(t_img)

            # Stack to (12, 3, H, W)
            return torch.stack(rotated_tensors)

        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            # Return a zero tensor in case of failure to avoid crashing (though strictly shouldn't happen)
            return torch.zeros((len(self.angles), 3, self.img_size, self.img_size))


class DualStreamExtractor:
    """
    Extracts features using DINOv2 and ConvNeXt-Large.
    """

    def __init__(self, device):
        self.device = device
        self.dino_model = None
        self.convnext_model = None
        self._load_models()

    def _load_models(self):
        logger.info(
            f"Loading models: {Config.MODEL_DINO} and {Config.MODEL_CONVNEXT}..."
        )

        # DINOv2 (Global Geometry)
        self.dino_model = timm.create_model(
            Config.MODEL_DINO,
            pretrained=True,
            num_classes=0,  # Get features
            img_size=Config.IMG_SIZE,
        )
        self.dino_model.to(self.device)
        self.dino_model.eval()

        # ConvNeXt (Local Texture)
        self.convnext_model = timm.create_model(
            Config.MODEL_CONVNEXT, pretrained=True, num_classes=0  # Get features
        )
        self.convnext_model.to(self.device)
        self.convnext_model.eval()

        logger.info("Models loaded successfully.")

    @torch.no_grad()
    def extract(self, dataloader):
        dino_features_list = []
        convnext_features_list = []

        total_batches = len(dataloader)
        logger.info(f"Starting extraction on {total_batches} batches...")

        for batch_idx, images in enumerate(dataloader):
            # images shape: (B, 12, 3, H, W)
            B, N_VIEWS, C, H, W = images.shape

            # Flatten to (B * 12, 3, H, W) for batch inference
            flat_images = images.view(-1, C, H, W).to(self.device)

            # Inference DINO
            dino_out = self.dino_model(flat_images)  # (B*12, Dim_DINO)

            # Inference ConvNeXt
            convnext_out = self.convnext_model(flat_images)  # (B*12, Dim_CN)

            # Reshape back to (B, 12, Dim)
            dino_out = dino_out.view(B, N_VIEWS, -1).cpu().numpy()
            convnext_out = convnext_out.view(B, N_VIEWS, -1).cpu().numpy()

            dino_features_list.append(dino_out)
            convnext_features_list.append(convnext_out)

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed batch {batch_idx + 1}/{total_batches}")

        # Concatenate all batches
        dino_features = np.concatenate(dino_features_list, axis=0)
        convnext_features = np.concatenate(convnext_features_list, axis=0)

        return dino_features, convnext_features


def get_tabular_data(df):
    """
    Extracts tabular features, IDs, and labels (if available) from the dataframe.
    """
    # Identify tabular feature columns
    feature_cols = [
        c
        for c in df.columns
        if any(c.startswith(prefix) for prefix in Config.TABULAR_COLS_PREFIXES)
    ]
    # Sort for consistency
    feature_cols = sorted(feature_cols)

    ids = df["id"].values
    tabular_features = df[feature_cols].values.astype(np.float32)

    labels = None
    if "species" in df.columns:
        labels = df["species"].values

    return ids, tabular_features, labels


def run_feature_extraction(load_cached_data: bool = True):
    """
    Main function to run the feature extraction pipeline.

    Args:
        load_cached_data (bool): If True, attempts to load features from disk.
                                 If False or files missing, re-runs extraction.

    Returns:
        dict: Dictionary containing loaded/extracted data arrays.
    """
    seed_everything(Config.SEED)

    # Define cache file paths
    cache_files = {
        "train_dino": Config.CACHE_TRAIN_DINO,
        "train_convnext": Config.CACHE_TRAIN_CONVNEXT,
        "train_ids": Config.CACHE_TRAIN_IDS,
        "train_labels": Config.CACHE_TRAIN_LABELS,
        "train_tab": Config.CACHE_TRAIN_TABULAR,
        "test_dino": Config.CACHE_TEST_DINO,
        "test_convnext": Config.CACHE_TEST_CONVNEXT,
        "test_ids": Config.CACHE_TEST_IDS,
        "test_tab": Config.CACHE_TEST_TABULAR,
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(path) for path in cache_files.values())

    if load_cached_data and all_cached:
        logger.info("Loading cached features from disk...")
        data = {}
        for key, path in cache_files.items():
            data[key] = np.load(path, allow_pickle=True)
        logger.info("Data loaded successfully.")
        return data

    logger.info("Cache not found or forced reload. Starting feature extraction...")

    # 1. Load Metadata
    # Combine Train and Val for the "Train" set (for Cross-Validation)
    df_train_part = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_part = pd.read_csv(Config.VAL_METADATA_PATH)
    df_train = pd.concat([df_train_part, df_val_part], ignore_index=True)

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    logger.info(f"Training samples (Train+Val): {len(df_train)}")
    logger.info(f"Test samples: {len(df_test)}")

    # 2. Setup Device and Extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    extractor = DualStreamExtractor(device)

    # 3. Process Training Data
    logger.info("Processing Training Data...")
    train_dataset = RotatedLeafDataset(
        df_train,
        Config.INPUT_DIR,
        img_size=Config.IMG_SIZE,
        angles=Config.ROTATION_ANGLES,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    train_dino, train_convnext = extractor.extract(train_loader)
    train_ids, train_tab, train_labels = get_tabular_data(df_train)

    # 4. Process Test Data
    logger.info("Processing Test Data...")
    test_dataset = RotatedLeafDataset(
        df_test,
        Config.INPUT_DIR,
        img_size=Config.IMG_SIZE,
        angles=Config.ROTATION_ANGLES,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_dino, test_convnext = extractor.extract(test_loader)
    test_ids, test_tab, _ = get_tabular_data(df_test)

    # 5. Save to Cache
    logger.info("Saving features to disk...")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    np.save(Config.CACHE_TRAIN_DINO, train_dino)
    np.save(Config.CACHE_TRAIN_CONVNEXT, train_convnext)
    np.save(Config.CACHE_TRAIN_IDS, train_ids)
    np.save(Config.CACHE_TRAIN_LABELS, train_labels)
    np.save(Config.CACHE_TRAIN_TABULAR, train_tab)

    np.save(Config.CACHE_TEST_DINO, test_dino)
    np.save(Config.CACHE_TEST_CONVNEXT, test_convnext)
    np.save(Config.CACHE_TEST_IDS, test_ids)
    np.save(Config.CACHE_TEST_TABULAR, test_tab)

    logger.info("Feature extraction complete and saved.")

    return {
        "train_dino": train_dino,
        "train_convnext": train_convnext,
        "train_ids": train_ids,
        "train_labels": train_labels,
        "train_tab": train_tab,
        "test_dino": test_dino,
        "test_convnext": test_convnext,
        "test_ids": test_ids,
        "test_tab": test_tab,
    }
