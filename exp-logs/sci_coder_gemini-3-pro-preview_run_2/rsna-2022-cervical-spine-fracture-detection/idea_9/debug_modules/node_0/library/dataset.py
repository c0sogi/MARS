import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library.config import Config
from library.utils import load_dicom_slice


def get_slice_cache(metadata_df: pd.DataFrame, load_cached_data: bool = True) -> dict:
    """
    Scans image directories to find available slice numbers for each study.
    Caches the result to a parquet file to speed up future initialization.
    """
    cache_file = os.path.join(Config.CACHE_DIR, "slices_cache.parquet")

    if load_cached_data and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            # Convert dataframe back to dict {uid: [slice_nums]}
            # We assume the column 'slice_indices' contains arrays/lists
            return dict(zip(df["StudyInstanceUID"], df["slice_indices"]))
        except Exception as e:
            print(f"Failed to load slice cache: {e}. Recomputing...")

    # Compute from scratch
    study_to_slices = {}

    # Pre-compute full paths to avoid repeated joins
    # metadata_df has 'image_path' relative to input dir
    # We need to scan these directories

    print("Scanning image directories...")
    for idx, row in metadata_df.iterrows():
        uid = row["StudyInstanceUID"]
        rel_path = row["image_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            study_to_slices[uid] = []
            continue

        # List all .dcm files
        # We use os.scandir for better performance than os.listdir
        slice_nums = []
        try:
            with os.scandir(full_path) as it:
                for entry in it:
                    if entry.name.endswith(".dcm") and entry.is_file():
                        try:
                            # Filename format is usually "int.dcm"
                            num = int(os.path.splitext(entry.name)[0])
                            slice_nums.append(num)
                        except ValueError:
                            pass
        except OSError:
            pass

        study_to_slices[uid] = sorted(slice_nums)

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Create DataFrame for parquet storage
    # Parquet handles columns of lists (arrays) well
    cache_df = pd.DataFrame(
        {
            "StudyInstanceUID": list(study_to_slices.keys()),
            "slice_indices": list(study_to_slices.values()),
        }
    )
    cache_df.to_parquet(cache_file, index=False)

    return study_to_slices


def get_bbox_cache(bbox_path: str, load_cached_data: bool = True) -> dict:
    """
    Processes the bounding box CSV into a dictionary for fast lookup.
    Structure: {StudyInstanceUID: {slice_num: [x, y, w, h]}}
    """
    cache_file = os.path.join(Config.CACHE_DIR, "bbox_cache.parquet")

    if load_cached_data and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            # Reconstruct dictionary
            bbox_dict = {}
            for uid, sub_df in df.groupby("StudyInstanceUID"):
                bbox_dict[uid] = dict(
                    zip(
                        sub_df["slice_number"],
                        sub_df[["x", "y", "width", "height"]].values.tolist(),
                    )
                )
            return bbox_dict
        except Exception as e:
            print(f"Failed to load bbox cache: {e}. Recomputing...")

    # Compute from scratch
    if not os.path.exists(bbox_path):
        return {}

    raw_df = pd.read_csv(bbox_path)

    # Save a processed version to parquet
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    raw_df.to_parquet(cache_file, index=False)

    # Build dictionary
    bbox_dict = {}
    for uid, sub_df in raw_df.groupby("StudyInstanceUID"):
        # Create a dict mapping slice_number -> [x, y, w, h]
        # Note: A slice might theoretically have multiple boxes, but we simplify to one or take the union.
        # For this task, we'll assume one box per slice or take the first one for simplicity
        # as the segmentation mask generation can handle overlapping logic if we expanded it,
        # but here we map slice -> box.
        # If duplicates exist, this takes the last one.
        # A more robust way for segmentation is to iterate, but for cache structure we keep it simple.
        records = sub_df[["slice_number", "x", "y", "width", "height"]].to_dict(
            "records"
        )
        slice_map = {
            r["slice_number"]: [r["x"], r["y"], r["width"], r["height"]]
            for r in records
        }
        bbox_dict[uid] = slice_map

    return bbox_dict


class CervicalSpineDataset(Dataset):
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        study_to_slices: dict,
        study_to_bboxes: dict = None,
        transform: A.Compose = None,
        is_train: bool = True,
        seq_len: int = Config.SEQ_LEN,
    ):
        """
        Args:
            metadata_df: DataFrame containing study metadata.
            study_to_slices: Dict mapping StudyInstanceUID to list of available slice numbers.
            study_to_bboxes: Dict mapping StudyInstanceUID to dict of {slice_num: [x, y, w, h]}.
            transform: Albumentations transform pipeline.
            is_train: Boolean flag for training mode.
            seq_len: Number of slices to sample per study.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.study_to_slices = study_to_slices
        self.study_to_bboxes = study_to_bboxes if study_to_bboxes is not None else {}
        self.transform = transform
        self.is_train = is_train
        self.seq_len = seq_len

        # Define default transform if none provided
        if self.transform is None:
            # Basic resizing is handled by load_dicom_slice, but we ensure tensor conversion here
            self.transform = A.Compose(
                [
                    A.Normalize(
                        mean=0.5, std=0.5
                    ),  # Simple normalization to center data
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        uid = row["StudyInstanceUID"]
        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        # 1. Slice Sampling
        available_slices = self.study_to_slices.get(uid, [])

        if len(available_slices) == 0:
            # Fallback for empty studies
            selected_indices = [0] * self.seq_len
            available_slices = [0]  # Dummy
        else:
            num_slices = len(available_slices)
            if num_slices >= self.seq_len:
                # Uniform sampling
                indices = np.linspace(0, num_slices - 1, self.seq_len).astype(int)
                selected_indices = [available_slices[i] for i in indices]
            else:
                # Padding (repeat last slice)
                indices = np.arange(num_slices)
                # Pad with the last available slice
                padding = [available_slices[-1]] * (self.seq_len - num_slices)
                selected_indices = [available_slices[i] for i in indices] + padding

        # 2. Load Images (2.5D Stacking)
        # We load z-1, z, z+1.
        # We construct a list of 3D arrays (H, W, 3) -> eventually (Seq, H, W, 3)
        images_list = []

        # Pre-calculate paths to optimize IO
        # We need a quick lookup for existence or just assume sorting implies continuity?
        # DICOM filenames are integers. We can compute neighbors mathematically.

        for z in selected_indices:
            # Neighbors
            z_prev = z - 1
            z_next = z + 1

            # Helper to load or pad
            # If neighbor doesn't exist, use current z
            # We check existence in the available_slices list logic or just try load

            # Simple strategy: try load, if fail (returns zeros), that's bad for 2.5D context.
            # Better: check if z_prev is in available_slices. If not, use z.

            s_prev = z_prev if z_prev in available_slices else z
            s_next = z_next if z_next in available_slices else z

            # Load slices
            # Note: load_dicom_slice resizes to Config.IMAGE_SIZE if provided
            path_curr = os.path.join(image_dir, f"{z}.dcm")
            path_prev = os.path.join(image_dir, f"{s_prev}.dcm")
            path_next = os.path.join(image_dir, f"{s_next}.dcm")

            img_curr = load_dicom_slice(path_curr, size=Config.IMAGE_SIZE)
            img_prev = load_dicom_slice(path_prev, size=Config.IMAGE_SIZE)
            img_next = load_dicom_slice(path_next, size=Config.IMAGE_SIZE)

            # Stack to (H, W, 3)
            stack = np.stack([img_prev, img_curr, img_next], axis=-1)
            images_list.append(stack)

        # Convert to numpy array: (Seq, H, W, 3)
        volume = np.array(images_list, dtype=np.float32)

        # 3. Generate Targets
        study_labels = np.zeros(8, dtype=np.float32)
        slice_labels = np.zeros(self.seq_len, dtype=np.float32)
        spatial_masks = np.zeros(
            (self.seq_len, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
        )

        if self.is_train:
            # Study Targets
            # Order: C1..C7, patient_overall
            cols = Config.TARGET_COLS
            study_labels = row[cols].values.astype(np.float32)

            # Slice & Spatial Targets
            study_bboxes = self.study_to_bboxes.get(uid, {})

            for i, z in enumerate(selected_indices):
                if z in study_bboxes:
                    # Slice has a fracture
                    slice_labels[i] = 1.0

                    # Create Mask
                    bbox = study_bboxes[z]  # [x, y, w, h]
                    x, y, w, h = bbox

                    # Bboxes are in original image coordinates.
                    # We need to scale them to Config.IMAGE_SIZE.
                    # Since we don't know original size here easily without opening DICOM header,
                    # we rely on the fact that load_dicom_slice resizes the image.
                    # However, to map bbox correctly, we need the scale factor.
                    # This is a limitation.
                    # Approximation: Most CTs are 512x512.
                    # We will assume 512x512 original size for scaling if not available.
                    # Or better: load_dicom_slice could return scale, but we can't change it.
                    # We will assume the bbox coordinates are relative to a 512x512 standard
                    # or calculate ratio if we assume input is 512.
                    # Let's assume 512 as base.
                    scale = Config.IMAGE_SIZE / 512.0

                    x = int(x * scale)
                    y = int(y * scale)
                    w = int(w * scale)
                    h = int(h * scale)

                    # Draw rectangle
                    cv2.rectangle(spatial_masks[i], (x, y), (x + w, y + h), 1.0, -1)

        # 4. Volumetric Augmentation
        # We need to apply the same geometric transform to all slices in the sequence.
        # Strategy: Stack images channel-wise -> (H, W, Seq*3)
        # Stack masks channel-wise -> (H, W, Seq)
        # Apply transform
        # Unstack

        H, W = Config.IMAGE_SIZE, Config.IMAGE_SIZE

        # Reshape images: (Seq, H, W, 3) -> (H, W, Seq*3)
        # Transpose to (H, W, Seq, 3) first then reshape
        flat_images = volume.reshape(self.seq_len * 3, H, W).transpose(1, 2, 0)

        # Reshape masks: (Seq, H, W) -> (H, W, Seq)
        flat_masks = spatial_masks.transpose(1, 2, 0)

        if self.transform:
            # Albumentations expects 'image' and optional 'mask'
            # We treat the huge stack as one multi-channel image
            augmented = self.transform(image=flat_images, mask=flat_masks)
            flat_images = augmented["image"]
            flat_masks = augmented["mask"]

        # Unstack Images
        # Result is tensor (C, H, W). C = Seq*3
        # We want (Seq, 3, H, W)
        if isinstance(flat_images, torch.Tensor):
            # (Seq*3, H, W)
            images_tensor = flat_images.view(self.seq_len, 3, H, W)
        else:
            # If transform didn't convert to tensor (unlikely with ToTensorV2)
            images_tensor = torch.from_numpy(flat_images.transpose(2, 0, 1)).view(
                self.seq_len, 3, H, W
            )

        # Unstack Masks
        if isinstance(flat_masks, torch.Tensor):
            # (Seq, H, W) -> (Seq, 1, H, W)
            masks_tensor = flat_masks.permute(2, 0, 1).unsqueeze(
                1
            )  # Wait, permute logic depends on input
            # If input was (H, W, Seq), ToTensorV2 makes it (Seq, H, W)
            masks_tensor = flat_masks.unsqueeze(1)  # (Seq, 1, H, W)
        else:
            masks_tensor = torch.from_numpy(flat_masks.transpose(2, 0, 1)).unsqueeze(1)

        # 5. Prepare Return Dict
        targets = {
            "study_labels": torch.tensor(study_labels, dtype=torch.float32),
            "slice_labels": torch.tensor(slice_labels, dtype=torch.float32),
            "spatial_masks": masks_tensor.float(),
        }

        return images_tensor.float(), targets
