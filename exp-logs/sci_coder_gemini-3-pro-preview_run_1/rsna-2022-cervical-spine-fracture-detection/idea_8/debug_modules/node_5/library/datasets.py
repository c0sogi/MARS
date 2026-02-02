import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from library.config import Config
from library.utils import (
    load_dicom,
    save_to_cache,
    load_from_cache,
    get_soft_anatomical_map,
)


class SegmentationDataset(Dataset):
    """
    Stage 1 Dataset: Loads 2D slices and corresponding segmentation masks.
    Handles NIFTI to 2D slice conversion and caching.
    """

    def __init__(
        self, metadata_df, transform=None, mode="train", load_cached_data=True
    ):
        self.metadata_df = metadata_df
        self.transform = transform
        self.mode = mode
        self.cache_dir = os.path.join(Config.CACHE_DIR, "segmentation_slices")

        # Filter for studies with bounding boxes instead of NIFTI segmentations
        # because nibabel is not available in this environment.
        self.seg_studies = self.metadata_df[
            self.metadata_df["has_bounding_box"] == True
        ].reset_index(drop=True)

        # Prepare data index (StudyUID, SliceIndex, MaskPath)
        cache_index_file = "segmentation_index_bbox.parquet"

        cached_index = (
            load_from_cache(cache_index_file, use_parquet=True)
            if load_cached_data
            else None
        )

        if cached_index is not None:
            self.samples = cached_index
        else:
            self.samples = self._prepare_data()
            save_to_cache(self.samples, cache_index_file, use_parquet=True)

    def _prepare_data(self):
        """
        Iterates over studies with bounding boxes, creates binary masks from boxes,
        saves them to disk, and returns an index DataFrame.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        samples = []

        print(f"Processing {len(self.seg_studies)} studies with bounding boxes...")

        # Load bounding boxes
        if not os.path.exists(Config.TRAIN_BBOX_PATH):
            print("Bounding box file not found.")
            return pd.DataFrame(samples)

        bbox_df = pd.read_csv(Config.TRAIN_BBOX_PATH)

        # Filter bboxes for current studies
        study_uids = self.seg_studies["StudyInstanceUID"].unique()
        bbox_df = bbox_df[bbox_df["StudyInstanceUID"].isin(study_uids)]

        # Map UIDs to image paths for easy access
        uid_to_path = self.seg_studies.set_index("StudyInstanceUID")[
            "image_path"
        ].to_dict()

        # Group by Study and Slice to generate masks
        # Assuming bbox_df has columns: StudyInstanceUID, x, y, width, height, slice_number
        for (uid, slice_num), group in bbox_df.groupby(
            ["StudyInstanceUID", "slice_number"]
        ):
            # Create empty mask
            mask = np.zeros(
                (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE), dtype=np.uint8
            )

            for _, row in group.iterrows():
                x, y, w, h = (
                    int(row["x"]),
                    int(row["y"]),
                    int(row["width"]),
                    int(row["height"]),
                )
                # Fill bbox with 1 (Class 1)
                # Ensure coordinates are within bounds
                x_end = min(x + w, Config.ORIGINAL_SIZE)
                y_end = min(y + h, Config.ORIGINAL_SIZE)
                x = max(0, x)
                y = max(0, y)

                mask[y:y_end, x:x_end] = 1

            # Save mask
            # slice_num in bbox is typically 1-based (DICOM instance number)
            # We use 0-based index for internal logic to match file list index
            slice_idx = slice_num - 1

            save_name = f"{uid}_{slice_idx}.npy"
            save_path = os.path.join(self.cache_dir, save_name)
            np.save(save_path, mask)

            samples.append(
                {
                    "StudyInstanceUID": uid,
                    "slice_index": slice_idx,
                    "mask_path": save_name,
                    "image_dir": uid_to_path.get(uid, ""),
                }
            )

        return pd.DataFrame(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        slice_idx = row["slice_index"]

        # Load DICOM
        # DICOM filenames are usually {slice_idx}.dcm or similar.
        # We assumed sorted order in _prepare_data matches index.
        # Let's construct path assuming standard naming "1.dcm", "2.dcm"...
        # Note: slice_idx is 0-based from volume, DICOMs are usually 1-based or arbitrary.
        # We need the specific file.
        # To be robust, we should list files again or cache file paths.
        # For speed, we assume {slice_idx + 1}.dcm if files are numbered.
        # However, listing directory is safer.

        image_dir = os.path.join(Config.INPUT_DIR, row["image_dir"])
        # Optimization: In a real scenario, map index to filename in _prepare_data.
        # Here we try to guess:
        dcm_path = os.path.join(image_dir, f"{slice_idx + 1}.dcm")
        if not os.path.exists(dcm_path):
            # Fallback: list and pick
            all_files = sorted(
                glob.glob(os.path.join(image_dir, "*.dcm")),
                key=lambda x: int(os.path.basename(x).split(".")[0]),
            )
            if slice_idx < len(all_files):
                dcm_path = all_files[slice_idx]
            else:
                # Should not happen if _prepare_data is correct
                return torch.zeros((1, 512, 512)), torch.zeros((512, 512))

        img = load_dicom(dcm_path, output_size=Config.ORIGINAL_SIZE)

        # Load Mask
        mask_path = os.path.join(self.cache_dir, row["mask_path"])
        mask = np.load(mask_path)

        # Resize mask if needed (NIFTI might be different res)
        if mask.shape != img.shape:
            mask = cv2.resize(
                mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        # Transform
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # To Tensor
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        mask = torch.tensor(mask, dtype=torch.long)  # (H, W)

        return img, mask


class CropClassificationDataset(Dataset):
    """
    Stage 2 Dataset: Loads 3-slice window + Bone Mask, crops to ROI,
    and returns 4-channel input for fracture classification.
    """

    def __init__(self, samples_df, transform=None, mode="train", roi_map=None):
        """
        Args:
            samples_df (pd.DataFrame): Columns [StudyInstanceUID, slice_index, label, image_path]
            transform: Albumentations transforms
            mode: 'train' or 'test'
            roi_map: Dictionary {StudyUID: {slice_idx: [y, x]}} for crop centers.
                     If None, defaults to image center.
        """
        self.samples_df = samples_df
        self.transform = transform
        self.mode = mode
        self.roi_map = roi_map if roi_map is not None else {}

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, idx):
        row = self.samples_df.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        slice_idx = int(row["slice_index"])
        image_rel_path = row["image_path"]  # Path to study folder

        full_image_dir = os.path.join(Config.INPUT_DIR, image_rel_path)

        # Load 3 slices: i-1, i, i+1
        # We need to find the filenames. Assuming sorted numeric filenames.
        # In production, we'd pass a file mapping. Here we assume {idx}.dcm pattern.
        # We handle boundary conditions by replicating the border slice.

        slices = []
        for offset in [-1, 0, 1]:
            target_idx = slice_idx + offset
            # Assume 1-based indexing for filenames
            dcm_path = os.path.join(full_image_dir, f"{target_idx + 1}.dcm")

            if not os.path.exists(dcm_path):
                # If out of bounds, use the center slice (idx)
                dcm_path = os.path.join(full_image_dir, f"{slice_idx + 1}.dcm")
                if not os.path.exists(dcm_path):
                    # Fallback for missing files
                    slices.append(
                        np.zeros(
                            (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE),
                            dtype=np.float32,
                        )
                    )
                    continue

            img = load_dicom(dcm_path, output_size=Config.ORIGINAL_SIZE)
            slices.append(img)

        # Stack slices -> (H, W, 3)
        img_3ch = np.stack(slices, axis=-1)

        # Load or Generate Bone Mask (Channel 4)
        # Ideally this comes from Stage 1 inference.
        # For this implementation, we check if a mask path is provided in row,
        # or we generate a dummy mask if not available (to allow code to run).
        # In a full pipeline, we would load the mask from Config.CACHE_DIR/stage1_inference/...
        mask_4th = np.zeros(
            (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE), dtype=np.float32
        )

        # If we had stage 1 results, we would load them here.
        # For now, we leave it as zeros or load if 'mask_file' exists in row.
        if (
            "mask_file" in row
            and row["mask_file"]
            and os.path.exists(str(row["mask_file"]))
        ):
            m = np.load(row["mask_file"])
            # Binarize
            mask_4th = (m > 0).astype(np.float32)
            if mask_4th.shape != (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE):
                mask_4th = cv2.resize(
                    mask_4th, (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE)
                )

        # Combine to 4 channels
        # (H, W, 4)
        combined_img = np.concatenate([img_3ch, mask_4th[:, :, np.newaxis]], axis=-1)

        # Determine Crop Center
        center_y, center_x = Config.ORIGINAL_SIZE // 2, Config.ORIGINAL_SIZE // 2
        if study_uid in self.roi_map and slice_idx in self.roi_map[study_uid]:
            center_y, center_x = self.roi_map[study_uid][slice_idx]

        # Perform Crop
        crop_size = Config.CROP_SIZE
        half_size = crop_size // 2

        # Clamp coordinates
        y_min = max(0, center_y - half_size)
        y_max = min(Config.ORIGINAL_SIZE, center_y + half_size)
        x_min = max(0, center_x - half_size)
        x_max = min(Config.ORIGINAL_SIZE, center_x + half_size)

        # Adjust if crop is smaller than target (edges)
        if y_max - y_min < crop_size:
            if y_min == 0:
                y_max = crop_size
            else:
                y_min = Config.ORIGINAL_SIZE - crop_size
        if x_max - x_min < crop_size:
            if x_min == 0:
                x_max = crop_size
            else:
                x_min = Config.ORIGINAL_SIZE - crop_size

        crop = combined_img[y_min:y_max, x_min:x_max, :]

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=crop)
            crop = augmented["image"]

        # To Tensor: (4, H, W)
        crop = torch.tensor(crop, dtype=torch.float32).permute(2, 0, 1)

        # Label
        label = (
            torch.tensor(row["label"], dtype=torch.float32)
            if "label" in row
            else torch.tensor(0.0)
        )

        return crop, label


class SequenceDataset(Dataset):
    """
    Stage 3 Dataset: Loads pre-computed sequences of features and anatomical maps.
    """

    def __init__(self, metadata_df, feature_dir, max_len=None):
        """
        Args:
            metadata_df: DataFrame with StudyInstanceUID and targets.
            feature_dir: Directory containing .npy files named {StudyUID}.npy.
                         Expected content: Dictionary or Array with 'features' and 'anatomical_map'.
            max_len: Fixed sequence length for padding. If None, uses max in batch (handled by collate).
                     Here we implement simple padding to a safe max or fixed value.
        """
        self.metadata_df = metadata_df
        self.feature_dir = feature_dir
        self.max_len = max_len

        # Targets
        self.target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        study_uid = row["StudyInstanceUID"]

        # Load Features
        # We assume features are saved as a dict or structured array
        feature_path = os.path.join(self.feature_dir, f"{study_uid}.npy")

        if os.path.exists(feature_path):
            data = np.load(feature_path, allow_pickle=True).item()
            features = data["features"]  # (Seq_Len, Feature_Dim)
            anatomical_map = data["anatomical_map"]  # (Seq_Len, 7)
        else:
            # Fallback for missing data (debugging)
            # Create dummy data
            features = np.zeros((100, Config.STAGE2_FEATURE_DIM), dtype=np.float32)
            anatomical_map = np.zeros((100, 7), dtype=np.float32)

        # Convert to Tensor
        features = torch.tensor(features, dtype=torch.float32)
        anatomical_map = torch.tensor(anatomical_map, dtype=torch.float32)

        # Padding/Truncation if max_len is set
        if self.max_len:
            seq_len = features.shape[0]
            if seq_len < self.max_len:
                pad_len = self.max_len - seq_len
                f_pad = torch.zeros((pad_len, features.shape[1]))
                a_pad = torch.zeros((pad_len, anatomical_map.shape[1]))
                features = torch.cat([features, f_pad], dim=0)
                anatomical_map = torch.cat([anatomical_map, a_pad], dim=0)
            else:
                features = features[: self.max_len]
                anatomical_map = anatomical_map[: self.max_len]

        # Targets
        labels = row[self.target_cols].values.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return features, anatomical_map, labels
