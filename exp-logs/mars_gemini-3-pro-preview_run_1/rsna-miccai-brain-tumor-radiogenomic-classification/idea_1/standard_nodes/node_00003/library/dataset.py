import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import load_mip_slice


def get_transforms(split="train"):
    """
    Returns the Albumentations transformation pipeline for a given split.
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class MGMTDataset(Dataset):
    def __init__(
        self, metadata_path, split="train", transform=None, load_cached_data=True
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            split (str): 'train', 'val', or 'test'. Used for naming cache files.
            transform (albumentations.Compose): Transformations to apply.
            load_cached_data (bool): Whether to try loading from cache.
        """
        self.split = split
        self.transform = transform
        self.metadata = pd.read_csv(metadata_path)

        # Define cache paths
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.images_cache_path = os.path.join(self.cache_dir, f"{split}_images.npy")
        self.targets_cache_path = os.path.join(self.cache_dir, f"{split}_targets.npy")
        self.ids_cache_path = os.path.join(self.cache_dir, f"{split}_ids.npy")

        self.images = None
        self.targets = None
        self.ids = None

        # Logic: Try load cache -> If fail or forced, process -> Save cache
        loaded = False
        if load_cached_data:
            if os.path.exists(self.images_cache_path) and os.path.exists(
                self.ids_cache_path
            ):
                # Check targets existence only if not test (test might not have targets,
                # but our metadata structure implies test_metadata doesn't have MGMT_value column usually,
                # however we handle it gracefully)
                try:
                    print(f"Loading cached data for {split} from {self.cache_dir}...")
                    images_temp = np.load(self.images_cache_path)
                    ids_temp = np.load(self.ids_cache_path)

                    if len(images_temp) != len(self.metadata):
                        print(
                            f"Cache mismatch! Found {len(images_temp)} images but metadata has {len(self.metadata)}. Reprocessing..."
                        )
                        loaded = False
                    else:
                        self.images = images_temp
                        self.ids = ids_temp

                        if os.path.exists(self.targets_cache_path):
                            self.targets = np.load(self.targets_cache_path)
                        else:
                            self.targets = np.zeros(len(self.images))  # Placeholder

                        loaded = True
                except Exception as e:
                    print(f"Failed to load cache: {e}. Reprocessing...")
                    loaded = False

        if not loaded:
            print(f"Processing data for {split} from scratch...")
            self._process_and_cache()

    def _normalize_slice(self, img):
        """
        Normalizes a 2D image slice to 0-255 uint8.
        """
        if img is None or img.size == 0:
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)

        img = img.astype(np.float32)
        min_val = img.min()
        max_val = img.max()

        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)

        img = (img * 255).astype(np.uint8)

        # Resize here to ensure consistency before stacking
        # Although transform does resizing, doing it here saves memory in cache
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        return img

    def _process_and_cache(self):
        images_list = []
        targets_list = []
        ids_list = []

        # Modalities to stack: FLAIR, T1wCE, T2w
        # Corresponds to Channels 0, 1, 2

        total = len(self.metadata)

        for idx, row in self.metadata.iterrows():
            # Get BraTS21ID
            subject_id = row["BraTS21ID"]
            ids_list.append(subject_id)

            # Get Target if available
            if "MGMT_value" in row:
                targets_list.append(row["MGMT_value"])
            else:
                targets_list.append(0.5)  # Placeholder for test

            # Construct paths
            # Metadata contains relative paths e.g., 'train/00000/FLAIR'
            flair_path = os.path.join(Config.INPUT_DIR, row["flair_path"])
            t1wce_path = os.path.join(Config.INPUT_DIR, row["t1wce_path"])
            t2w_path = os.path.join(Config.INPUT_DIR, row["t2w_path"])

            # Load Slices using MIP (Cite solution_lesson_node_00002)
            img_flair = load_mip_slice(flair_path)
            img_t1wce = load_mip_slice(t1wce_path)
            img_t2w = load_mip_slice(t2w_path)

            # Normalize and Resize
            img_flair = self._normalize_slice(img_flair)
            img_t1wce = self._normalize_slice(img_t1wce)
            img_t2w = self._normalize_slice(img_t2w)

            # Stack to (H, W, 3) -> (224, 224, 3)
            # Channel order: FLAIR, T1wCE, T2w
            stacked_img = np.stack([img_flair, img_t1wce, img_t2w], axis=-1)
            images_list.append(stacked_img)

        # Convert to numpy arrays
        self.images = np.array(images_list, dtype=np.uint8)
        self.ids = np.array(ids_list, dtype=np.int64)
        self.targets = np.array(targets_list, dtype=np.float32)

        # Save to cache
        print(f"Saving cache to {self.cache_dir}...")
        np.save(self.images_cache_path, self.images)
        np.save(self.ids_cache_path, self.ids)
        if "MGMT_value" in self.metadata.columns:
            np.save(self.targets_cache_path, self.targets)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (H, W, 3) uint8
        target = self.targets[idx]
        subject_id = self.ids[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (shouldn't happen usually)
            image = ToTensorV2()(image=image)["image"]

        # Return dictionary or tuple? Standard PyTorch is tuple (data, target)
        # But we might need ID for submission.
        # Let's return (image, target, subject_id) to be safe for inference loops

        return image, torch.tensor(target, dtype=torch.float32), subject_id
