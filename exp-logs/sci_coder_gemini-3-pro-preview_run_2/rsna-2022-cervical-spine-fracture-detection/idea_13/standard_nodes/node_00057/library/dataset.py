import os
import torch
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_dicom_slice, get_all_study_paths


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.
    Uses ReplayCompose to ensure consistent augmentation across the sequence.
    """
    if mode == "train":
        return A.ReplayCompose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


class CervicalSpineDataset(Dataset):
    def __init__(self, metadata_df, mode="train", transform=None):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing metadata.
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Transformations to apply.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

        # Determine root directory and cache key based on mode
        if mode == "test":
            self.root_dir = Config.TEST_IMAGES_DIR
            self.cache_key = "test"
        else:
            self.root_dir = Config.TRAIN_IMAGES_DIR
            self.cache_key = "train"

        # Load file paths for all studies (cached)
        self.file_paths_map = get_all_study_paths(
            self.root_dir, cache_key=self.cache_key, load_cached_data=True
        )

        # Load bounding boxes for attention supervision (only for training/val)
        self.bbox_df = None
        if mode in ["train", "val"] and os.path.exists(Config.BOUNDING_BOX_PATH):
            self.bbox_df = pd.read_csv(Config.BOUNDING_BOX_PATH)
            # Pre-filter bboxes to speed up lookup
            self.bbox_map = self.bbox_df.groupby("StudyInstanceUID")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row["StudyInstanceUID"]

        # 1. Retrieve file paths
        # If paths are missing (e.g. data download issue), handle gracefully
        all_paths = self.file_paths_map.get(uid, [])
        num_slices = len(all_paths)

        # 2. Determine Sampling Indices
        # We need exactly Config.SEQ_LEN (96) slices.
        if num_slices == 0:
            # Fallback for empty/missing studies
            indices = np.zeros(Config.SEQ_LEN, dtype=int)
            all_paths = []
        else:
            # Uniformly sample indices across the Z-axis
            indices = np.linspace(0, num_slices - 1, Config.SEQ_LEN).astype(int)

        # 3. Prepare Tensors
        # Shape: (Seq_Len, Channels, H, W) -> (96, 3, 384, 384)
        # We initialize a list to collect processed tensors
        processed_slices = []

        # 4. Augmentation State
        # We need to capture the random parameters of the first slice
        # and replay them for the rest of the sequence.
        replay_data = None

        # 5. Load and Process Sequence
        for i, center_idx in enumerate(indices):
            # 2.5D Stacking: [z-1, z, z+1]
            # Clamp indices to valid range [0, num_slices-1]
            stack_indices = [center_idx - 1, center_idx, center_idx + 1]
            stack_indices = [max(0, min(x, num_slices - 1)) for x in stack_indices]

            # Load 3 channels
            channels = []
            for s_idx in stack_indices:
                if num_slices > 0:
                    img = load_dicom_slice(
                        all_paths[s_idx], size=None
                    )  # Resize handled in transform
                else:
                    img = np.zeros(Config.IMAGE_SIZE)
                channels.append(img)

            # Stack to (H, W, 3)
            image_stack = np.stack(channels, axis=-1).astype(np.float32)

            # Apply Transformations
            if self.transform:
                if self.mode == "train":
                    # Use ReplayCompose logic
                    if i == 0:
                        # First slice: Apply and record parameters
                        augmented = self.transform(image=image_stack)
                        image_stack = augmented["image"]
                        replay_data = augmented["replay"]
                    else:
                        # Subsequent slices: Replay exact parameters
                        if replay_data:
                            augmented = A.ReplayCompose.replay(
                                replay_data, image=image_stack
                            )
                            image_stack = augmented["image"]
                        else:
                            # Fallback if replay failed (shouldn't happen)
                            augmented = self.transform(image=image_stack)
                            image_stack = augmented["image"]
                else:
                    # Val/Test: Deterministic transform (Resize/Norm only)
                    augmented = self.transform(image=image_stack)
                    image_stack = augmented["image"]

            processed_slices.append(image_stack)

        # Stack sequence: (Seq, C, H, W)
        images_tensor = torch.stack(processed_slices)

        # 6. Prepare Targets
        # Classification Targets: [C1, C2, C3, C4, C5, C6, C7, patient_overall]
        targets = np.zeros(Config.NUM_CLASSES, dtype=np.float32)

        # Attention Mask Targets: [7, Seq_Len] (One row per vertebrae C1-C7)
        attn_mask = np.zeros((7, Config.SEQ_LEN), dtype=np.float32)
        has_bbox = 0.0

        if self.mode != "test":
            # Fill classification targets
            cols = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]
            for idx_t, col in enumerate(cols):
                targets[idx_t] = row[col]

            # Generate Attention Masks from Bounding Boxes
            if self.bbox_df is not None and uid in self.bbox_map.groups:
                bboxes = self.bbox_map.get_group(uid)
                has_bbox = 1.0

                # We need to map real slice numbers to our sampled indices
                # Extract slice numbers from filenames (e.g., '10.dcm' -> 10)
                try:
                    real_slice_nums = np.array(
                        [int(os.path.basename(p).split(".")[0]) for p in all_paths]
                    )
                except Exception:
                    real_slice_nums = np.arange(num_slices)

                for _, bbox in bboxes.iterrows():
                    fracture_slice_num = bbox["slice_number"]

                    # Find the index in the original file list
                    # Use absolute difference to find nearest slice if exact match missing
                    if len(real_slice_nums) > 0:
                        abs_diff = np.abs(real_slice_nums - fracture_slice_num)
                        true_z_idx = np.argmin(abs_diff)

                        # Now map 'true_z_idx' to our sampled 'indices'
                        # indices contains the indices of files we actually loaded
                        seq_diff = np.abs(indices - true_z_idx)
                        seq_idx = np.argmin(seq_diff)

                        # Generate Gaussian
                        sigma = 2.0
                        x = np.arange(Config.SEQ_LEN)
                        gaussian = np.exp(-0.5 * ((x - seq_idx) / sigma) ** 2)

                        # Normalize to probability distribution (sum=1) for KLDiv
                        # or keep as heatmap (max=1) for MSE.
                        # Using max=1 is often more stable for "attention guidance".
                        gaussian = gaussian / (gaussian.max() + 1e-6)

                        # Assign to relevant channels.
                        # Since we don't know exactly which C-level the bbox is for (csv lacks class),
                        # we apply this attention mask to ALL vertebrae marked as fractured in the CSV labels.
                        for c_idx in range(7):
                            if targets[c_idx] == 1.0:
                                # Combine with max to handle multiple fractures close by
                                attn_mask[c_idx] = np.maximum(
                                    attn_mask[c_idx], gaussian
                                )

        return {
            "images": images_tensor,  # (96, 3, 384, 384)
            "targets": torch.tensor(targets, dtype=torch.float32),  # (8,)
            "attn_mask": torch.tensor(attn_mask, dtype=torch.float32),  # (7, 96)
            "has_bbox": torch.tensor([has_bbox], dtype=torch.float32),  # (1,)
            "row_id": uid,
        }
