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
        Loads data, computes ROI, extracts specific slices, and caches the final tensors.
        Optimized to reduce I/O by only reading necessary slices for non-anchor modalities.
        Cite solution_lesson_node_00098: Decouple I/O via Pre-Computed Tensor Caching.

        Args:
            metadata_df (pd.DataFrame): Metadata containing paths.
            cache_key (str): Identifier for the cache file.
            load_cached_data (bool): Whether to attempt loading from disk.

        Returns:
            dict: {
                'images': {braTS21ID: np.ndarray(12, H, W)},
                'roi_centers': {braTS21ID: int}
            }
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(Config.CACHE_DIR, f"{cache_key}.npy")

        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading cached data from {cache_path}...")
            try:
                data = np.load(cache_path, allow_pickle=True).item()
                logger.info("Cache loaded successfully.")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Reprocessing...")

        logger.info(f"Processing {len(metadata_df)} subjects for {cache_key}...")

        images_dict = {}
        roi_centers = {}

        if Config.DEBUG_DATA_LIMIT:
            metadata_df = metadata_df.head(Config.DEBUG_DATA_LIMIT)

        for _, row in metadata_df.iterrows():
            subject_id = row["BraTS21ID"]

            # 1. Process FLAIR first to determine ROI (Cite solution_lesson_node_00093)
            flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
            flair_files = DataCacher._get_sorted_files(flair_path)

            flair_slices = []
            for fp in flair_files:
                flair_slices.append(read_dicom_robust(fp, target_size=Config.IMG_SIZE))

            if not flair_slices:
                # Fallback for empty FLAIR
                flair_vol = np.zeros(
                    (1, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
                )
                depth = 1
            else:
                flair_vol = np.array(flair_slices, dtype=np.float32)
                depth = flair_vol.shape[0]

            # ROI Selection
            start_idx = int(depth * Config.ROI_START_PCT)
            end_idx = int(depth * Config.ROI_END_PCT)

            if end_idx <= start_idx:
                roi_center = depth // 2
            else:
                valid_slices = flair_vol[start_idx:end_idx]
                intensities = np.sum(valid_slices, axis=(1, 2))
                best_local = np.argmax(intensities)
                roi_center = start_idx + best_local

            roi_centers[subject_id] = roi_center

            # 2. Extract Slices for All Modalities (Cite solution_lesson_node_00110: Fixed Stride)
            # Indices: [Center-Stride, Center, Center+Stride]
            stride = Config.STRIDE
            indices = [roi_center - stride, roi_center, roi_center + stride]

            channels = []

            for mod in Config.MODALITIES:
                if mod == "FLAIR":
                    # We already have FLAIR volume in memory
                    vol_depth = flair_vol.shape[0]
                    for idx in indices:
                        clamped_i = max(0, min(vol_depth - 1, idx))
                        channels.append(flair_vol[clamped_i])
                else:
                    # Sparse Read: Only read specific files
                    mod_path = os.path.join(Config.INPUT_DIR, row[f"path_{mod}"])
                    mod_files = DataCacher._get_sorted_files(mod_path)
                    vol_depth = len(mod_files)

                    if vol_depth == 0:
                        # Missing modality
                        for _ in range(3):
                            channels.append(
                                np.zeros(
                                    (Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
                                )
                            )
                    else:
                        for idx in indices:
                            clamped_i = max(0, min(vol_depth - 1, idx))
                            fp = mod_files[clamped_i]
                            img = read_dicom_robust(fp, target_size=Config.IMG_SIZE)
                            channels.append(img)

            # Stack -> (12, H, W)
            image_stack = np.stack(channels, axis=-1)
            images_dict[subject_id] = image_stack

        if len(images_dict) == 0:
            raise RuntimeError("No data processed! Circuit breaker triggered.")

        combined_data = {"images": images_dict, "roi_centers": roi_centers}
        logger.info(f"Saving cache to {cache_path}...")
        np.save(cache_path, combined_data)
        logger.info("Cache saved.")

        return combined_data


class MRIDataset(Dataset):
    """
    PyTorch Dataset for Glioblastoma MGMT detection.
    Uses pre-computed tensors for maximum efficiency.
    """

    def __init__(self, data_cache, metadata_df, transform=None):
        """
        Args:
            data_cache (dict): Output from DataCacher.
            metadata_df (pd.DataFrame): Metadata with IDs and Labels.
            transform (A.Compose): Albumentations transforms.
        """
        self.images = data_cache["images"]
        self.metadata = metadata_df.reset_index(drop=True)
        self.transform = transform

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

        # Get Pre-computed Tensor (H, W, 12)
        image_stack = self.images[subject_id]

        # Augmentation
        if self.transform:
            augmented = self.transform(image=image_stack)
            image_stack = augmented["image"]

        # Normalization: Independent Per-Channel Min-Max Scaling [0, 1]
        # Cite solution_lesson_node_00058: Independent Modality Normalization
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
