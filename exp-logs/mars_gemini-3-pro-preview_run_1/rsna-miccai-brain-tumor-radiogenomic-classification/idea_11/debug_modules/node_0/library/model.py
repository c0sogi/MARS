import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import timm
import cv2
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class GLiClassModel(nn.Module):
    """
    EfficientNet-B0 based model for 2.5D MRI Classification.
    Accepts 3-channel input (FLAIR, T1wCE, T2w) and outputs a single logit.
    """

    def __init__(
        self,
        backbone=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        in_chans=Config.IN_CHANNELS,
    ):
        super(GLiClassModel, self).__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans
        )

    def forward(self, x):
        return self.backbone(x)


class GLiClassDataset(Dataset):
    """
    Dataset class implementing the Independent-Instance 2.5D Volumetric Ensemble (I2VE) strategy.
    Handles Brain-Centric ROI detection, deterministic slice extraction, and caching.
    """

    def __init__(
        self, metadata_path, split="train", transform=None, load_cached_data=True
    ):
        self.split = split
        self.transform = transform
        self.df = pd.read_csv(metadata_path)

        # Cache paths
        self.cache_images_path = os.path.join(
            Config.CACHE_DIR, f"cache_{split}_images.npy"
        )
        self.cache_targets_path = os.path.join(
            Config.CACHE_DIR, f"cache_{split}_targets.npy"
        )
        self.cache_ids_path = os.path.join(Config.CACHE_DIR, f"cache_{split}_ids.npy")

        # Load or Generate Data
        if load_cached_data and os.path.exists(self.cache_images_path):
            print(f"Loading {split} data from cache...")
            self.images = np.load(self.cache_images_path)
            self.targets = np.load(self.cache_targets_path)
            self.ids = np.load(self.cache_ids_path)
        else:
            print(f"Processing {split} data from scratch...")
            self.images, self.targets, self.ids = self._process_data()

            # Save to cache
            np.save(self.cache_images_path, self.images)
            np.save(self.cache_targets_path, self.targets)
            np.save(self.cache_ids_path, self.ids)
            print(f"Saved {split} data to cache at {Config.CACHE_DIR}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        subject_id = self.ids[idx]

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Default conversion to tensor if no transform provided
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        return image, torch.tensor(target, dtype=torch.float32), subject_id

    def _read_dicom(self, path):
        """Reads a DICOM file and returns a numpy array."""
        try:
            ds = pydicom.dcmread(path)
            return ds.pixel_array
        except Exception:
            # Fallback to OpenCV if pydicom fails or file is standard image
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not read file: {path}")
            return img

    def _get_brain_roi(self, t2w_path):
        """
        Scans T2w images to find the Z-range containing brain tissue.
        Returns (start_index, end_index).
        """
        files = sorted(
            [f for f in os.listdir(t2w_path) if f.endswith(".dcm")],
            key=lambda x: int(x.split("-")[-1].split(".")[0]),
        )

        if not files:
            return 0, 0

        # Scan every 5th slice to save time
        step = 5
        non_empty_indices = []

        for i in range(0, len(files), step):
            f_path = os.path.join(t2w_path, files[i])
            try:
                img = self._read_dicom(f_path)
                if img.max() > 0:
                    non_empty_indices.append(i)
            except:
                continue

        if not non_empty_indices:
            return 0, len(files) - 1

        # Add buffer
        start = max(0, non_empty_indices[0] - step)
        end = min(len(files) - 1, non_empty_indices[-1] + step)

        return start, end

    def _normalize(self, img):
        """Min-Max normalization to [0, 1]."""
        img = img.astype(np.float32)
        min_val = img.min()
        max_val = img.max()
        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)
        return img

    def _process_data(self):
        all_images = []
        all_targets = []
        all_ids = []

        for _, row in self.df.iterrows():
            sid = row["BraTS21ID"]
            target = row["MGMT_value"] if "MGMT_value" in row else -1.0

            # Paths
            paths = {
                "FLAIR": os.path.join(Config.INPUT_DIR, row["flair_path"]),
                "T1wCE": os.path.join(Config.INPUT_DIR, row["t1wce_path"]),
                "T2w": os.path.join(Config.INPUT_DIR, row["t2w_path"]),
            }

            # 1. Determine ROI using T2w (usually best contrast for geometry)
            if not os.path.exists(paths["T2w"]):
                continue  # Skip if path invalid

            roi_start, roi_end = self._get_brain_roi(paths["T2w"])
            roi_len = roi_end - roi_start

            if roi_len <= 0:
                continue

            # 2. Extract Slices at specific depths
            for depth in Config.SLICE_DEPTHS:
                # Calculate relative index
                rel_idx = int(roi_start + roi_len * depth)

                channels = []
                valid_sample = True

                for mod in Config.SELECTED_MODALITIES:
                    mod_dir = paths[mod]
                    if not os.path.exists(mod_dir):
                        valid_sample = False
                        break

                    # Get files sorted by instance number
                    files = sorted(
                        [f for f in os.listdir(mod_dir) if f.endswith(".dcm")],
                        key=lambda x: int(x.split("-")[-1].split(".")[0]),
                    )

                    if not files:
                        valid_sample = False
                        break

                    # Map T2w-based relative depth to this modality's file list
                    # We assume roughly co-registered depth, so 50% in T2w ~ 50% in FLAIR
                    mod_idx = int(
                        len(files) * (rel_idx / (len(os.listdir(paths["T2w"])) or 1))
                    )
                    mod_idx = min(max(0, mod_idx), len(files) - 1)

                    img_path = os.path.join(mod_dir, files[mod_idx])
                    try:
                        img = self._read_dicom(img_path)
                        img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
                        img = self._normalize(img)
                        channels.append(img)
                    except:
                        valid_sample = False
                        break

                if valid_sample and len(channels) == 3:
                    # Stack channels: (H, W, 3)
                    stacked_img = np.stack(channels, axis=-1)
                    all_images.append(stacked_img)
                    all_targets.append(target)
                    all_ids.append(sid)

        return (
            np.array(all_images, dtype=np.float32),
            np.array(all_targets, dtype=np.float32),
            np.array(all_ids, dtype=np.int64),
        )


def get_transforms(split="train"):
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])
