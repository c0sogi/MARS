import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import read_dicom_robust, get_logger

logger = get_logger(__name__)


class DataCacher:
    """
    Handles loading, processing, and caching of MRI data.
    Implements Circuit-Breaker logic and Fidelity-Aligned ROI selection.
    """

    @staticmethod
    def _get_sorted_files(dir_path):
        """Returns sorted list of file paths in a directory."""
        if not os.path.exists(dir_path):
            return []
        files = [
            f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
        ]
        # Sort numerically (e.g., Image-1.dcm, Image-10.dcm)
        try:
            files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
        except:
            files.sort()
        return [os.path.join(dir_path, f) for f in files]

    @staticmethod
    def process_data(metadata_df, cache_key, load_cached_data=True):
        """
        Loads data into RAM, computing ROI centers and caching the result.

        Args:
            metadata_df (pd.DataFrame): Metadata containing paths.
            cache_key (str): Identifier for the cache file (e.g. 'train', 'test').
            load_cached_data (bool): Whether to attempt loading from disk.

        Returns:
            dict: {
                'images': {braTS21ID: {modality: np.ndarray(D, H, W)}},
                'roi_centers': {braTS21ID: int}
            }
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(Config.CACHE_DIR, f"{cache_key}.npy")

        # 1. Attempt Load
        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading cached data from {cache_path}...")
            try:
                # allow_pickle=True is required for loading object arrays (dicts)
                data = np.load(cache_path, allow_pickle=True).item()
                logger.info("Cache loaded successfully.")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Reprocessing...")

        # 2. Process from Scratch
        logger.info(f"Processing {len(metadata_df)} subjects for {cache_key}...")

        images_dict = {}
        roi_centers = {}

        # Debug limit
        if Config.DEBUG_DATA_LIMIT:
            metadata_df = metadata_df.head(Config.DEBUG_DATA_LIMIT)
            logger.info(f"Debug Mode: Limiting to {Config.DEBUG_DATA_LIMIT} samples.")

        for _, row in metadata_df.iterrows():
            subject_id = row["BraTS21ID"]
            subject_data = {}

            # Load all modalities
            for mod in Config.MODALITIES:
                path_col = f"path_{mod}"
                rel_path = row[path_col]
                full_dir = os.path.join(Config.INPUT_DIR, rel_path)

                file_paths = DataCacher._get_sorted_files(full_dir)

                mod_slices = []
                for fp in file_paths:
                    img = read_dicom_robust(fp, target_size=Config.IMG_SIZE)
                    mod_slices.append(img)

                if not mod_slices:
                    # Handle empty/missing directories by creating a dummy slice
                    mod_slices = [
                        np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
                    ]

                # Stack to (D, H, W)
                subject_data[mod] = np.array(mod_slices, dtype=np.float32)

            # Fidelity-Aligned ROI Selection (FLAIR Anchor)
            flair_vol = subject_data["FLAIR"]
            depth = flair_vol.shape[0]

            # Bounds: 15% - 85%
            start_idx = int(depth * Config.ROI_START_PCT)
            end_idx = int(depth * Config.ROI_END_PCT)

            if end_idx <= start_idx:
                # Fallback for very shallow volumes
                roi_center = depth // 2
            else:
                # Calculate intensity sums (Integral)
                valid_slices = flair_vol[start_idx:end_idx]
                intensities = np.sum(valid_slices, axis=(1, 2))
                best_local = np.argmax(intensities)
                roi_center = start_idx + best_local

            roi_centers[subject_id] = roi_center
            images_dict[subject_id] = subject_data

        # Circuit Breaker Logic
        if len(images_dict) == 0:
            raise RuntimeError("No data processed! Circuit breaker triggered.")

        # 3. Save to Cache
        combined_data = {"images": images_dict, "roi_centers": roi_centers}

        logger.info(f"Saving cache to {cache_path}...")
        np.save(cache_path, combined_data)
        logger.info("Cache saved.")

        return combined_data


class MRIDataset(Dataset):
    """
    PyTorch Dataset for Glioblastoma MGMT detection.
    Implements Stochastic Multi-Scale Stacking and Geometric Augmentation.
    """

    def __init__(self, data_cache, metadata_df, transform=None, stride_mode="random"):
        """
        Args:
            data_cache (dict): Output from DataCacher.
            metadata_df (pd.DataFrame): Metadata with IDs and Labels.
            transform (A.Compose): Albumentations transforms.
            stride_mode (str or int): 'random' for training, or specific int (2 or 5) for eval.
        """
        self.images = data_cache["images"]
        self.roi_centers = data_cache["roi_centers"]
        self.metadata = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.stride_mode = stride_mode

        # Pre-filter metadata to only include IDs present in cache
        valid_ids = set(self.images.keys())
        self.metadata = self.metadata[
            self.metadata["BraTS21ID"].isin(valid_ids)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        subject_id = row["BraTS21ID"]

        # Get Label (if exists, else 0.5)
        label = row["MGMT_value"] if "MGMT_value" in row else 0.5
        label = torch.tensor(label, dtype=torch.float32)

        # Get Data
        subj_images = self.images[subject_id]
        center_idx = self.roi_centers[subject_id]

        # Determine Stride (Stochastic Multi-Scale)
        if self.stride_mode == "random":
            stride = np.random.choice(Config.STRIDE_OPTIONS)
        else:
            stride = int(self.stride_mode)

        # Select Slices: [Center-Stride, Center, Center+Stride]
        offsets = [-stride, 0, stride]
        indices = [center_idx + o for o in offsets]

        channels = []

        for mod in Config.MODALITIES:
            vol = subj_images[mod]
            depth = vol.shape[0]

            mod_slices = []
            for i in indices:
                # Edge Clamping for spatial neighbors
                clamped_i = max(0, min(depth - 1, i))
                slice_img = vol[clamped_i]
                mod_slices.append(slice_img)

            channels.extend(mod_slices)

        # Stack -> (H, W, 12)
        image_stack = np.stack(channels, axis=-1)

        # Augmentation
        if self.transform:
            augmented = self.transform(image=image_stack)
            image_stack = augmented["image"]

        # Normalization: Independent Per-Channel Min-Max Scaling [0, 1]
        # Handle both Tensor (from ToTensorV2) and numpy array
        if isinstance(image_stack, torch.Tensor):
            image_tensor = image_stack.float()
            # Shape: (C, H, W)
            for c in range(image_tensor.shape[0]):
                c_min = image_tensor[c].min()
                c_max = image_tensor[c].max()
                if c_max > c_min:
                    image_tensor[c] = (image_tensor[c] - c_min) / (c_max - c_min)
                else:
                    image_tensor[c] = 0.0
        else:
            # Shape: (H, W, C)
            image_stack = image_stack.astype(np.float32)
            for c in range(image_stack.shape[2]):
                c_min = image_stack[:, :, c].min()
                c_max = image_stack[:, :, c].max()
                if c_max > c_min:
                    image_stack[:, :, c] = (image_stack[:, :, c] - c_min) / (
                        c_max - c_min
                    )
                else:
                    image_stack[:, :, c] = 0.0
            image_tensor = torch.from_numpy(image_stack.transpose(2, 0, 1))

        return image_tensor, label


def get_transforms(phase):
    """
    Returns Albumentations transforms for train/val/test.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Rotation with Reflection Padding to avoid artifacts
                A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])
