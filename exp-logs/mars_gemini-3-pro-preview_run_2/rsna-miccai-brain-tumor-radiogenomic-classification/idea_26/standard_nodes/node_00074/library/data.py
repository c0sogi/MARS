import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import read_dicom_raw, get_roi_map


class BraTSDataset(Dataset):
    """
    Dataset class for Asymmetric Grouped EfficientNet.
    Constructs a 24-channel input tensor using Dual-Stride Interleaved Stacking.
    """

    def __init__(self, metadata_df, roi_map, phase="train", transform=None):
        self.df = metadata_df
        self.roi_map = roi_map
        self.phase = phase
        self.transform = transform

        # Define default augmentations for training if not provided
        if self.phase == "train" and self.transform is None:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    # Rotation limited to +/- 15 degrees with zero padding
                    A.Rotate(
                        limit=Config.ROTATION_DEGREES,
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                ]
            )

    def __len__(self):
        return len(self.df)

    def normalize(self, img):
        """
        Applies Independent Per-Slice Min-Max Scaling [0, 1].
        """
        min_val = img.min()
        max_val = img.max()
        if max_val - min_val > 0:
            return (img - min_val) / (max_val - min_val)
        return img - min_val  # Returns zeros if image is constant

    def load_modality_slices(self, subject_dir, anchor_idx, num_slices_flair):
        """
        Loads the Local and Context slices for a specific modality.
        Maps the FLAIR anchor index to the current modality using relative depth.
        """
        # 1. List and sort files numerically
        try:
            files = sorted(
                [f for f in os.listdir(subject_dir) if f.endswith(".dcm")],
                key=lambda x: int(x.split("-")[1].split(".")[0]) if "-" in x else x,
            )
        except (FileNotFoundError, IndexError):
            files = []

        num_slices = len(files)

        # Handle missing data with blank slices
        if num_slices == 0:
            return [
                np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
                for _ in range(6)
            ]

        # 2. Map Anchor: Use relative depth from FLAIR to find corresponding slice in this modality
        if num_slices_flair > 0:
            relative_depth = anchor_idx / num_slices_flair
            local_anchor = int(relative_depth * num_slices)
        else:
            local_anchor = 0  # Fallback

        # Clamp anchor to valid range
        local_anchor = max(0, min(local_anchor, num_slices - 1))

        # 3. Define Indices for Single-Stride Stacking
        # We need 3 slices total: [Anchor-S, Anchor, Anchor+S]
        indices = []

        # Use the single defined stride (Cite solution_lesson_node_00037)
        stride = Config.STRIDES[0]  # 5
        for offset in [-stride, 0, stride]:
            indices.append(local_anchor + offset)

        loaded_slices = []
        for idx in indices:
            # Edge Clamping: Replicate boundary slices for out-of-bounds indices
            idx_clamped = max(0, min(idx, num_slices - 1))
            f_path = os.path.join(subject_dir, files[idx_clamped])

            # Read Raw
            img = read_dicom_raw(f_path)

            # Resize (Area interpolation for noise suppression)
            img = cv2.resize(
                img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
            )
            img = img.astype(np.float32)

            # Normalize
            img = self.normalize(img)
            loaded_slices.append(img)

        return loaded_slices

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        brats_id = row["BraTS21ID"]

        # 1. Get FLAIR Anchor from ROI Map
        anchor_idx = self.roi_map.get(brats_id, 0)

        # 2. Determine FLAIR slice count (needed for relative depth mapping)
        flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
        num_slices_flair = 0
        try:
            # Quick listing to get count. OS caching makes this relatively fast.
            if os.path.exists(flair_path):
                num_slices_flair = len(
                    [f for f in os.listdir(flair_path) if f.endswith(".dcm")]
                )
        except Exception:
            pass

        # 3. Construct 24-Channel Volume
        all_channels = []

        # Iterate modalities in fixed order: FLAIR, T1w, T1wCE, T2w
        for mod in Config.MODALITIES:
            mod_path = os.path.join(Config.INPUT_DIR, row[f"path_{mod}"])
            # Returns 6 slices: [Local_1, Local_2, Local_3, Context_1, Context_2, Context_3]
            slices = self.load_modality_slices(mod_path, anchor_idx, num_slices_flair)
            all_channels.extend(slices)

        # Stack along channel dimension: (H, W, 24)
        img_volume = np.stack(all_channels, axis=-1)

        # 4. Apply Augmentations (Train only)
        if self.transform:
            augmented = self.transform(image=img_volume)
            img_volume = augmented["image"]

        # 5. Convert to Tensor (Channels First): (24, H, W)
        img_tensor = torch.from_numpy(img_volume.transpose(2, 0, 1)).float()

        if self.phase == "test":
            return img_tensor, brats_id
        else:
            label = row["MGMT_value"]
            return img_tensor, torch.tensor(label, dtype=torch.float32)


def get_dataloader(split, batch_size=None, shuffle=None, load_cached_data=True):
    """
    Factory function to create DataLoaders.
    Handles metadata loading and ROI cache integration.
    """
    # 1. Load Metadata
    if split == "train":
        df = pd.read_csv(Config.TRAIN_METADATA)
        is_train = True
    elif split == "val":
        df = pd.read_csv(Config.VAL_METADATA)
        is_train = False
    elif split == "test":
        df = pd.read_csv(Config.TEST_METADATA)
        is_train = False
    else:
        raise ValueError(f"Invalid split: {split}")

    # 2. Load/Compute ROI Map (Caching mechanism via utils.py)
    # We pass the loaded dataframe so the utility knows which subjects to compute if missing.
    roi_map = get_roi_map(df, load_cached_data=load_cached_data)

    # 3. Create Dataset
    ds = BraTSDataset(df, roi_map, phase=split)

    # 4. Configure Loader
    bs = batch_size if batch_size else Config.BATCH_SIZE
    shuff = shuffle if shuffle is not None else is_train

    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuff,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
