import os
import cv2
import torch
import timm
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from tqdm import tqdm

from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    CACHE_DIR,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_ROTATIONS,
    ROTATION_ANGLES,
    MODEL_DINO_NAME,
    MODEL_CONVNEXT_NAME,
    BATCH_SIZE,
    NUM_WORKERS,
)
from library.utils import setup_logger, get_device, save_array, load_array

# Initialize logger
logger = setup_logger("feature_extraction.log")


class LeafRotationDataset(Dataset):
    """
    Dataset class that loads leaf images and generates 12 equidistant rotated views.
    """

    def __init__(self, metadata_df, input_dir, transform=None):
        self.metadata = metadata_df
        self.input_dir = input_dir
        self.transform = transform
        self.angles = ROTATION_ANGLES

        # Base transform: Resize and Normalize
        # Rotation happens on PIL image before ToTensor
        self.to_tensor_norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load image using OpenCV
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL for easy rotation with fill support
        img_pil = Image.fromarray(img)

        # Resize to target size (224x224)
        # We resize before rotation. Since images are leaves on white background,
        # we can resize directly.
        img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)

        views = []
        for angle in self.angles:
            # Rotate image
            # fill=(255, 255, 255) ensures the background remains white
            img_rotated = F.rotate(
                img_pil,
                angle,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=(255, 255, 255),
            )

            # Convert to tensor and normalize
            img_tensor = self.to_tensor_norm(img_rotated)
            views.append(img_tensor)

        # Stack views: (12, 3, 224, 224)
        views_tensor = torch.stack(views)

        return views_tensor, img_id


def get_models(device):
    """
    Loads DINOv2 and ConvNeXt models with classification heads removed.
    """
    logger.info(f"Loading DINOv2 model: {MODEL_DINO_NAME}")
    dino_model = timm.create_model(MODEL_DINO_NAME, pretrained=True, num_classes=0)
    dino_model.to(device)
    dino_model.eval()

    logger.info(f"Loading ConvNeXt model: {MODEL_CONVNEXT_NAME}")
    conv_model = timm.create_model(MODEL_CONVNEXT_NAME, pretrained=True, num_classes=0)
    conv_model.to(device)
    conv_model.eval()

    return dino_model, conv_model


def extract_features(dataset, device, desc="Extracting"):
    """
    Runs inference on the dataset using both models.
    Returns dictionaries of features and IDs.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    dino_model, conv_model = get_models(device)

    all_dino_feats = []
    all_conv_feats = []
    all_ids = []

    # Disable gradients for inference
    with torch.no_grad():
        for batch_imgs, batch_ids in tqdm(dataloader, desc=desc):
            # batch_imgs shape: (B, 12, 3, H, W)
            B, N_VIEWS, C, H, W = batch_imgs.shape

            # Flatten to (B * 12, 3, H, W) for batch inference
            flat_imgs = batch_imgs.view(-1, C, H, W).to(device)

            # --- DINOv2 Extraction ---
            # Output: (B*12, 1024)
            dino_out = dino_model(flat_imgs)
            dino_out = dino_out.view(B, N_VIEWS, -1).cpu().numpy()
            all_dino_feats.append(dino_out)

            # --- ConvNeXt Extraction ---
            # Output: (B*12, 1536)
            conv_out = conv_model(flat_imgs)
            conv_out = conv_out.view(B, N_VIEWS, -1).cpu().numpy()
            all_conv_feats.append(conv_out)

            all_ids.append(batch_ids.numpy())

    # Concatenate all batches
    dino_features = np.concatenate(all_dino_feats, axis=0)
    conv_features = np.concatenate(all_conv_feats, axis=0)
    ids = np.concatenate(all_ids, axis=0)

    return {
        "dino": dino_features,
        "conv": conv_features,
        "ids": ids,
    }


def run_extraction(load_cached_data: bool = True, debug_sample_size: int = None):
    """
    Main driver function to extract features for Train, Val, and Test sets.
    Handles caching to avoid re-computation.
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    # Define file names for cache
    splits = ["train", "val", "test"]
    cache_files = {}
    for split in splits:
        cache_files[split] = {
            "dino": f"{split}_dino.npy",
            "conv": f"{split}_conv.npy",
            "ids": f"{split}_ids.npy",
        }

    # Check if all cache files exist
    all_cached = True
    if load_cached_data:
        for split in splits:
            for key, fname in cache_files[split].items():
                if load_array(fname) is None:
                    all_cached = False
                    break
    else:
        all_cached = False

    # Return cached data if available
    if all_cached and load_cached_data:
        logger.info("Loading features from cache...")
        results = {}
        for split in splits:
            results[split] = {
                "dino": load_array(cache_files[split]["dino"]),
                "conv": load_array(cache_files[split]["conv"]),
                "ids": load_array(cache_files[split]["ids"]),
            }
        return results["train"], results["val"], results["test"]

    # --- Process Data from Scratch ---
    logger.info("Starting feature extraction pipeline...")

    results = {}

    for split in splits:
        logger.info(f"Processing {split} set...")

        # Load metadata
        meta_path = os.path.join(METADATA_DIR, f"{split}.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df = pd.read_csv(meta_path)

        # Debugging limit
        if debug_sample_size is not None:
            logger.warning(
                f"Debugging mode: Limiting {split} to {debug_sample_size} samples."
            )
            df = df.head(debug_sample_size)

        # Create Dataset
        dataset = LeafRotationDataset(df, INPUT_DIR)

        # Extract
        feats = extract_features(dataset, device, desc=f"Extracting {split}")

        # Save to cache
        save_array(feats["dino"], cache_files[split]["dino"])
        save_array(feats["conv"], cache_files[split]["conv"])
        save_array(feats["ids"], cache_files[split]["ids"])

        results[split] = feats

    logger.info("Feature extraction completed and cached.")
    return results["train"], results["val"], results["test"]
