import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    IMAGE_SIZE,
    SLICE_DEPTH,
    NUM_SLABS,
    SLAB_STRIDE,
    IN_CHANNELS,
    SEED,
)
from library.utils import load_dicom_as_array, normalize_minmax, get_sorted_file_list


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline for the specific phase.
    Implements the geometric distortions specified in WITS-II.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
                        ),
                        A.GridDistortion(p=0.5),
                    ],
                    p=0.3,
                ),
                # Ensure output is tensor
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )


class SlabDataset(Dataset):
    """
    Implements the WITS-II Data Pipeline:
    - Independent Instance Learning (3 slabs per subject treated as independent)
    - Thick Slab Construction (9 channels: 3 slices x 3 modalities)
    - Independent Heuristic Alignment (Median-based centering per modality)
    """

    def __init__(
        self, metadata_df, transform=None, load_cached_data=True, split_name="train"
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata containing paths and labels.
            transform (albumentations.Compose): Augmentation pipeline.
            load_cached_data (bool): Whether to load/save data from disk cache.
            split_name (str): Name of the split (train/val/test) for cache naming.
        """
        self.metadata_df = metadata_df
        self.transform = transform
        self.split_name = split_name

        # Cache paths
        self.cache_dir = WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_images_path = os.path.join(
            self.cache_dir, f"cache_{split_name}_images.npy"
        )
        self.cache_labels_path = os.path.join(
            self.cache_dir, f"cache_{split_name}_labels.npy"
        )
        self.cache_ids_path = os.path.join(
            self.cache_dir, f"cache_{split_name}_ids.npy"
        )

        # Load or Process Data
        loaded = False
        if load_cached_data and self._check_cache_exists():
            print(f"Loading cached {split_name} data from {self.cache_dir}...")
            cached_images = np.load(self.cache_images_path)

            # Validate dimensions against configuration (Cite debug_lesson_5)
            if cached_images.shape[-1] == IN_CHANNELS:
                self.images = cached_images
                self.labels = np.load(self.cache_labels_path)
                self.ids = np.load(self.cache_ids_path)
                loaded = True
            else:
                print(
                    f"Cache mismatch: Expected {IN_CHANNELS} channels, got {cached_images.shape[-1]}. Reprocessing..."
                )

        if not loaded:
            print(f"Processing {split_name} data from scratch (WITS-II Pipeline)...")
            self.images, self.labels, self.ids = self._process_dataset()

            # Save to cache
            print(f"Saving {split_name} data to cache...")
            np.save(self.cache_images_path, self.images)
            np.save(self.cache_labels_path, self.labels)
            np.save(self.cache_ids_path, self.ids)

        print(
            f"Dataset {split_name} ready. Shape: {self.images.shape}, Samples: {len(self.labels)}"
        )

    def _check_cache_exists(self):
        return (
            os.path.exists(self.cache_images_path)
            and os.path.exists(self.cache_labels_path)
            and os.path.exists(self.cache_ids_path)
        )

    def _process_dataset(self):
        """
        Iterates through metadata, extracts slabs, and stacks channels.
        Returns numpy arrays for images, labels, and IDs.
        """
        processed_images = []
        processed_labels = []
        processed_ids = []

        # Modalities to process in order (Channels 0-2, 3-5, 6-8)
        modalities = ["flair", "t1wce", "t2w"]

        for idx, row in self.metadata_df.iterrows():
            subject_id = row["BraTS21ID"]
            # Handle missing label for test set
            label = row["MGMT_value"] if "MGMT_value" in row else 0.5

            # 1. Identify File Lists and Median Indices per Modality
            modality_data = {}
            valid_subject = True

            for mod in modalities:
                rel_path = row[f"{mod}_path"]
                full_path = os.path.join(INPUT_DIR, rel_path)
                files = get_sorted_file_list(full_path)

                if not files:
                    # If any modality is missing, we might skip or handle gracefully.
                    # For this task, we assume data quality is decent or we pad.
                    # Here we skip to avoid noise.
                    valid_subject = False
                    break

                modality_data[mod] = {
                    "files": files,
                    "path": full_path,
                    "median_idx": len(files) // 2,
                }

            if not valid_subject:
                continue

            # 2. Extract Slabs (Single Median Slice - Cite Lesson 00015)
            # We only take the exact median slice (offset 0)
            offsets = [0]

            for offset in offsets:
                # This list will hold 3 channels (1 slice * 3 modalities)
                slab_channels = []

                for mod in modalities:
                    info = modality_data[mod]
                    center_idx = info["median_idx"] + offset
                    files = info["files"]
                    base_path = info["path"]

                    # We need 1 slice: center
                    # Cite Lesson 00009: Avoid naive channel stacking of depth
                    slice_indices = [center_idx]

                    for s_idx in slice_indices:
                        # Boundary check
                        if 0 <= s_idx < len(files):
                            file_path = os.path.join(base_path, files[s_idx])
                            img = load_dicom_as_array(file_path)
                            img = normalize_minmax(img)
                        else:
                            # Pad with zeros if out of bounds
                            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

                        # Resize if necessary (though DICOMs are usually 512 or 256)
                        if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
                            try:
                                img = cv2.resize(
                                    img,
                                    (IMAGE_SIZE, IMAGE_SIZE),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                            except Exception:
                                img = np.zeros(
                                    (IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
                                )

                        slab_channels.append(img)

                # Stack to (H, W, 3)
                # slab_channels is list of (224, 224) arrays
                slab_tensor = np.stack(slab_channels, axis=-1)

                processed_images.append(slab_tensor)
                processed_labels.append(label)
                processed_ids.append(subject_id)

        return (
            np.array(processed_images, dtype=np.float32),
            np.array(processed_labels, dtype=np.float32),
            np.array(processed_ids, dtype=np.int64),
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]  # (H, W, 9)
        label = self.labels[idx]

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # (9, H, W) via ToTensorV2
        else:
            # Fallback to tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1))

        return image, torch.tensor(label, dtype=torch.float32)
