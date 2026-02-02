import os
import cv2
import torch
import timm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from library.config import Config
from library.utils import get_logger, save_data_to_cache, load_data_from_cache

# Initialize logger
logger = get_logger(name="feature_extractor")


class LeafImageDataset(Dataset):
    """
    Dataset class that loads images and generates 12 rotated views per image
    to support Manifold Densification.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform
        self.rotation_angles = Config.ROTATION_ANGLES
        self.input_dir = Config.INPUT_DIR
        self.img_size = Config.IMG_SIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = int(row["id"])
        rel_path = row["file_path"]
        full_path = os.path.join(self.input_dir, rel_path)

        # Read image using OpenCV
        # Images are binary black leaves on white backgrounds
        img = cv2.imread(full_path)

        if img is None:
            # Fallback for missing images (though metadata check ensures existence)
            # Create a blank white image
            img = np.full((self.img_size, self.img_size, 3), 255, dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Generate 12 views
        views = []
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        for angle in self.rotation_angles:
            # Calculate rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Rotate with white border to match background
            rotated_img = cv2.warpAffine(
                img,
                M,
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

            # Convert to PIL for consistent torchvision transforms
            pil_img = Image.fromarray(rotated_img)

            if self.transform:
                views.append(self.transform(pil_img))
            else:
                views.append(transforms.ToTensor()(pil_img))

        # Stack views: (12, 3, H, W)
        views_tensor = torch.stack(views)

        return image_id, views_tensor


class FeatureExtractor:
    """
    Dual-Stream Feature Extractor implementing DINOv2 and ConvNeXt inference
    on 12-view manifolds.
    """

    def __init__(self):
        self.device = torch.device(
            Config.DEVICE if torch.cuda.is_available() else "cpu"
        )
        self.img_size = Config.IMG_SIZE

        # Standard ImageNet normalization for pre-trained models
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.model_dino = None
        self.model_conv = None

    def _load_models(self):
        """Loads DINOv2 and ConvNeXt models onto the GPU."""
        if self.model_dino is not None and self.model_conv is not None:
            return

        logger.info(f"Loading Global Geometry Stream: {Config.MODEL_DINO}")
        self.model_dino = timm.create_model(
            Config.MODEL_DINO,
            pretrained=True,
            num_classes=0,  # Return raw embeddings
            img_size=self.img_size,
        )
        self.model_dino.to(self.device)
        self.model_dino.eval()

        logger.info(f"Loading Local Texture Stream: {Config.MODEL_CONVNEXT}")
        self.model_conv = timm.create_model(
            Config.MODEL_CONVNEXT,
            pretrained=True,
            num_classes=0,
            global_pool="avg",  # Ensure vector output
        )
        self.model_conv.to(self.device)
        self.model_conv.eval()

    def extract_features(
        self, df: pd.DataFrame, dataset_name: str, load_cached_data: bool = True
    ):
        """
        Extracts features for the provided dataset.

        Args:
            df (pd.DataFrame): DataFrame containing 'id' and 'file_path'.
            dataset_name (str): Identifier for the dataset (e.g., 'train', 'test', 'val') used for caching.
            load_cached_data (bool): If True, attempts to load from disk cache first.

        Returns:
            tuple: (ids, dino_features, conv_features)
                - ids: np.ndarray (N,)
                - dino_features: np.ndarray (N, 12, 1024)
                - conv_features: np.ndarray (N, 12, 1536)
        """
        # 1. Handle Debug Mode Slicing
        if Config.DEBUG:
            logger.info(
                f"DEBUG mode active: Limiting {dataset_name} to {Config.DEBUG_SAMPLE_SIZE} samples."
            )
            df = df.head(Config.DEBUG_SAMPLE_SIZE).copy()
            dataset_name = f"{dataset_name}_debug"

        # 2. Define Cache Paths
        cache_ids_path = os.path.join(Config.WORKING_DIR, f"{dataset_name}_ids.npy")
        cache_dino_path = os.path.join(Config.WORKING_DIR, f"{dataset_name}_dino.npy")
        cache_conv_path = os.path.join(Config.WORKING_DIR, f"{dataset_name}_conv.npy")

        # 3. Try Loading from Cache
        if load_cached_data:
            ids = load_data_from_cache(cache_ids_path)
            dino_feats = load_data_from_cache(cache_dino_path)
            conv_feats = load_data_from_cache(cache_conv_path)

            if ids is not None and dino_feats is not None and conv_feats is not None:
                logger.info(f"Successfully loaded {dataset_name} features from cache.")
                return ids, dino_feats, conv_feats
            else:
                logger.info(
                    f"Cache miss for {dataset_name}. Starting feature extraction..."
                )

        # 4. Initialize Models and Data
        self._load_models()

        dataset = LeafImageDataset(df, transform=self.transform)

        # Effective batch size on GPU = BATCH_SIZE * 12 views
        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        all_ids = []
        all_dino_feats = []
        all_conv_feats = []

        # 5. Inference Loop
        with torch.no_grad():
            for batch_ids, batch_imgs in tqdm(
                dataloader, desc=f"Extracting {dataset_name}"
            ):
                # batch_imgs shape: (Batch, 12, 3, 224, 224)
                B, V, C, H, W = batch_imgs.shape

                # Flatten views into the batch dimension for inference
                # Shape: (Batch * 12, 3, 224, 224)
                flat_imgs = batch_imgs.view(B * V, C, H, W).to(self.device)

                # DINOv2 Inference
                # Output: (Batch * 12, 1024)
                dino_out = self.model_dino(flat_imgs)

                # ConvNeXt Inference
                # Output: (Batch * 12, 1536)
                conv_out = self.model_conv(flat_imgs)

                # Reshape back to (Batch, 12, Feature_Dim)
                dino_out = dino_out.view(B, V, -1).cpu().numpy()
                conv_out = conv_out.view(B, V, -1).cpu().numpy()

                all_ids.append(batch_ids.numpy())
                all_dino_feats.append(dino_out)
                all_conv_feats.append(conv_out)

        # 6. Aggregate Results
        final_ids = np.concatenate(all_ids)
        final_dino = np.concatenate(all_dino_feats)
        final_conv = np.concatenate(all_conv_feats)

        # 7. Save to Cache
        save_data_to_cache(final_ids, cache_ids_path)
        save_data_to_cache(final_dino, cache_dino_path)
        save_data_to_cache(final_conv, cache_conv_path)

        logger.info(f"Feature extraction complete. Saved to {Config.WORKING_DIR}")

        # Clean up to free GPU memory
        del self.model_dino
        del self.model_conv
        self.model_dino = None
        self.model_conv = None
        torch.cuda.empty_cache()

        return final_ids, final_dino, final_conv
