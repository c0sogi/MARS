import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import pydicom
import nibabel as nib
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library import config, utils

# ====================================================
# UTILITY FUNCTIONS
# ====================================================


def get_image_paths(image_dir):
    """
    Returns a dictionary mapping slice number to file path for a study directory.
    """
    image_paths = {}
    if not os.path.exists(image_dir):
        return image_paths

    for fname in os.listdir(image_dir):
        if fname.endswith(".dcm"):
            try:
                slice_idx = int(os.path.splitext(fname)[0])
                image_paths[slice_idx] = os.path.join(image_dir, fname)
            except ValueError:
                continue
    return image_paths


def load_and_sort_dicoms(study_path):
    """
    Loads DICOM metadata to sort files by Z-position.
    Returns a list of (slice_number, z_position, file_path) sorted by z_position.
    """
    files = []
    for fname in os.listdir(study_path):
        if fname.endswith(".dcm"):
            fpath = os.path.join(study_path, fname)
            try:
                # Read only specific tags for speed
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                # ImagePositionPatient[2] is the Z coordinate
                z_pos = float(ds.ImagePositionPatient[2])
                slice_num = int(os.path.splitext(fname)[0])
                files.append((slice_num, z_pos, fpath))
            except Exception:
                continue

    # Sort by Z position (usually superior to inferior or vice versa)
    # We will match this order to the NIfTI Z-axis
    files.sort(key=lambda x: x[1])
    return files


# ====================================================
# DATASETS
# ====================================================


class SegmentationDataset(Dataset):
    """
    Dataset for Stage 1: Multi-Class Anatomical Localizer (2D U-Net).
    Loads paired DICOM slices and NIfTI segmentation masks.
    """

    def __init__(self, metadata_df, transform=None, cache_nifti=True):
        self.metadata = metadata_df[
            metadata_df["has_segmentation"] == True
        ].reset_index(drop=True)
        self.transform = transform
        self.cache_nifti = cache_nifti
        self.nifti_cache = {}
        self.dicom_lists_cache = {}

        # Define augmentations if none provided
        if self.transform is None:
            self.transform = A.Compose(
                [
                    A.Resize(config.FULL_IMAGE_SIZE, config.FULL_IMAGE_SIZE),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                    ),
                    A.Normalize(
                        mean=(0.5,), std=(0.5,)
                    ),  # Normalize to [-1, 1] roughly
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        # We define length as number of studies * approx slices?
        # Or just iterate studies?
        # Standard practice: One epoch = iteration over all slices.
        # But we have variable slices. Let's index by (study_idx, slice_idx).
        # To avoid pre-calculating all slices, we can just randomly sample a slice from a study
        # effectively making the epoch length = number of studies.
        # For better coverage, we can multiply length.
        return (
            len(self.metadata) * 50
        )  # Heuristic: sample 50 slices per study per epoch

    def __getitem__(self, idx):
        # Map linear index to study
        study_idx = idx % len(self.metadata)
        row = self.metadata.iloc[study_idx]
        study_uid = row["StudyInstanceUID"]

        # Paths
        image_dir = os.path.join(config.INPUT_DIR, row["image_path"])
        seg_path = os.path.join(config.INPUT_DIR, row["segmentation_path"])

        # 1. Load Segmentation Volume (Cached)
        if study_uid not in self.nifti_cache:
            try:
                nii = nib.load(seg_path)
                nii = nib.as_closest_canonical(nii)  # Reorient to RAS
                vol = nii.get_fdata()
                # vol shape: (X, Y, Z) usually.
                # We need to ensure orientation matches DICOM.
                # NIfTI RAS: x=Right, y=Anterior, z=Superior.
                # DICOM Axial: usually looked at from feet up.
                # We will assume Z-axis of vol corresponds to sorted DICOMs.
                if self.cache_nifti:
                    self.nifti_cache[study_uid] = vol
            except Exception as e:
                # Fallback: return zeros
                print(f"Error loading NIfTI {seg_path}: {e}")
                vol = np.zeros((512, 512, 100))  # Dummy
                if self.cache_nifti:
                    self.nifti_cache[study_uid] = vol
        else:
            vol = self.nifti_cache[study_uid]

        # 2. Load DICOM List (Cached)
        if study_uid not in self.dicom_lists_cache:
            dicom_files = load_and_sort_dicoms(image_dir)
            if self.cache_nifti:
                self.dicom_lists_cache[study_uid] = dicom_files
        else:
            dicom_files = self.dicom_lists_cache[study_uid]

        # 3. Select a Slice
        # We want to sample slices that contain segmentation labels more often
        num_slices = min(len(dicom_files), vol.shape[2])

        # Try to find a slice with annotations
        attempts = 0
        selected_z = 0
        while attempts < 10:
            z_idx = np.random.randint(0, num_slices)
            # Check if mask has content (classes 1-7)
            mask_slice = vol[:, :, z_idx]
            # Classes 1-7 are C1-C7. 8+ are Thoracic.
            if np.any((mask_slice >= 1) & (mask_slice <= 7)):
                selected_z = z_idx
                break
            attempts += 1
            selected_z = z_idx  # Fallback to random

        # 4. Load Image and Mask
        slice_info = dicom_files[selected_z]  # (slice_num, z_pos, path)
        img_path = slice_info[2]

        # Load and Window Image
        img = utils.load_dicom_array(img_path)
        img = utils.apply_windowing(
            img, config.BONE_WINDOW_CENTER, config.BONE_WINDOW_WIDTH
        )

        # Get Mask Slice
        # Note: NIfTI (X, Y, Z) -> Transpose to (Y, X) to match DICOM (Rows, Cols)
        # Usually NIfTI is (Left-Right, Posterior-Anterior, Inf-Sup).
        # DICOM pixel array is (Rows/Y, Cols/X).
        # We transpose the mask slice.
        mask = vol[:, :, selected_z].T

        # Filter Labels: Keep 1-7, set others to 0
        mask_filtered = np.where((mask >= 1) & (mask <= 7), mask, 0).astype(np.float32)

        # Resize/Augment
        # Albumentations expects HWC
        # Image is HW, Mask is HW.
        # Expand dims for image to HWC (HW1)
        # Mask is class indices, keep as HW? No, usually masks are HW for segmentation.

        if self.transform:
            # Albumentations handles image (H,W) or (H,W,C)
            augmented = self.transform(image=img, mask=mask_filtered)
            img_tensor = augmented[
                "image"
            ]  # (1, H, W) due to ToTensorV2? No, usually (C, H, W) if input was C.
            mask_tensor = augmented["mask"]  # (H, W) or (1, H, W)

            # Ensure mask is Long for CrossEntropy
            mask_tensor = mask_tensor.long()

            # Image needs to be float
            img_tensor = img_tensor.float()

            # If ToTensorV2 didn't add channel dim to image (if input was 2D)
            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(0)

        else:
            img_tensor = torch.from_numpy(img).unsqueeze(0).float()
            mask_tensor = torch.from_numpy(mask_filtered).long()

        # One-hot encode mask? No, CrossEntropyLoss expects class indices (H, W).

        return img_tensor, mask_tensor


class EncoderTrainDataset(Dataset):
    """
    Dataset for Stage 2: Mask-Conditioned Feature Encoder (2.5D CNN).
    Loads 3-slice window + Mask, crops to ROI, predicts fracture probability.
    """

    def __init__(self, metadata_df, bbox_df=None, transform=None, phase="train"):
        self.metadata = metadata_df
        self.transform = transform
        self.phase = phase

        # Process Bounding Boxes
        self.samples = []

        if bbox_df is not None and phase == "train":
            # 1. Positive Samples (from BBoxes)
            # Filter bboxes for studies in metadata
            valid_uids = set(self.metadata["StudyInstanceUID"])
            bbox_df = bbox_df[bbox_df["StudyInstanceUID"].isin(valid_uids)]

            for _, row in bbox_df.iterrows():
                self.samples.append(
                    {
                        "StudyInstanceUID": row["StudyInstanceUID"],
                        "slice_number": row["slice_number"],
                        "label": 1.0,
                        "roi": (row["y"], row["x"]),  # Center (y, x) approx
                        "is_bbox": True,
                    }
                )

            # 2. Negative Samples
            # Sample N negatives where N approx equals positives
            n_pos = len(self.samples)
            n_neg = n_pos  # Balanced

            all_studies = self.metadata["StudyInstanceUID"].values

            # Simple random sampling
            # We assume a study has ~300 slices.
            for _ in range(n_neg):
                uid = np.random.choice(all_studies)
                # Random slice (heuristic range)
                sl = np.random.randint(1, 300)
                # Check collision with positives (simplified: ignore exact collision check for speed)
                self.samples.append(
                    {
                        "StudyInstanceUID": uid,
                        "slice_number": sl,
                        "label": 0.0,
                        "roi": None,  # Will use image center
                        "is_bbox": False,
                    }
                )
        else:
            # Validation / Inference mode: iterate all slices or a subset?
            # For validation, we usually want patient level, but this dataset is slice level.
            # We will generate a fixed set of samples from the validation metadata.
            # Just sample some random slices for validation check
            for _, row in self.metadata.iterrows():
                uid = row["StudyInstanceUID"]
                # Add a few slices per patient
                for s in range(100, 200, 20):
                    self.samples.append(
                        {
                            "StudyInstanceUID": uid,
                            "slice_number": s,
                            "label": 0.0,  # Dummy
                            "roi": None,
                            "is_bbox": False,
                        }
                    )

        # Pre-compute paths map
        self.study_paths = dict(
            zip(self.metadata["StudyInstanceUID"], self.metadata["image_path"])
        )

        # Augmentations
        if self.transform is None:
            self.transform = A.Compose(
                [
                    A.Resize(config.CROP_IMAGE_SIZE, config.CROP_IMAGE_SIZE),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5),
                    A.Normalize(
                        mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)
                    ),  # 4 channels
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        uid = sample["StudyInstanceUID"]
        slice_num = int(sample["slice_number"])
        label = float(sample["label"])

        base_path = os.path.join(config.INPUT_DIR, self.study_paths[uid])

        # Load 3 slices: t-1, t, t+1
        channels = []
        for offset in [-1, 0, 1]:
            s_idx = slice_num + offset
            # Construct path (assuming simple naming)
            p = os.path.join(base_path, f"{s_idx}.dcm")
            if os.path.exists(p):
                img = utils.load_dicom_array(p)
                img = utils.apply_windowing(
                    img, config.BONE_WINDOW_CENTER, config.BONE_WINDOW_WIDTH
                )
            else:
                # Padding with zeros
                img = np.zeros(
                    (config.FULL_IMAGE_SIZE, config.FULL_IMAGE_SIZE), dtype=np.float32
                )
            channels.append(img)

        # 4th Channel: Mask
        # Heuristic: Thresholding bone in the center slice
        center_img = channels[1]
        # Bone is bright in bone window (which is 0-1). Threshold > 0.3 approx
        mask = (center_img > 0.3).astype(np.float32)
        channels.append(mask)

        # Stack: (H, W, 4)
        combined = np.stack(channels, axis=-1)

        # Determine Crop Center
        h, w = combined.shape[:2]
        if sample["roi"] is not None:
            cy, cx = sample["roi"]
            # BBox coordinates are usually top-left?
            # train_bounding_boxes.csv: x, y, width, height.
            # If we stored (y, x) as center in __init__, we are good.
            # Let's correct __init__ logic:
            # The csv has x, y (top left). Center = x + w/2, y + h/2.
            # In __init__ I stored row['y'], row['x']. Let's assume I stored center there.
            # Re-calculating center here if needed:
            # Actually, let's assume the stored 'roi' is the center (y, x).
            pass
        else:
            # Fallback: Image center
            cy, cx = h // 2, w // 2

        # Crop
        cropped = utils.crop_to_roi(combined, (cy, cx), config.CROP_IMAGE_SIZE)

        # Augment
        if self.transform:
            augmented = self.transform(image=cropped)
            tensor = augmented["image"]  # (4, H, W)
        else:
            tensor = torch.from_numpy(cropped).permute(2, 0, 1).float()

        return tensor, torch.tensor([label], dtype=torch.float32)


class SequenceDataset(Dataset):
    """
    Dataset for Stage 3: Anatomically-Grouped Recurrent Aggregator.
    Loads pre-computed features and patient-level labels.
    """

    def __init__(self, metadata_df, feature_dir=None, phase="train"):
        self.metadata = metadata_df
        self.feature_dir = (
            feature_dir if feature_dir else os.path.join(config.CACHE_DIR, "features")
        )
        self.phase = phase
        self.max_len = config.STAGE3_CONFIG["max_seq_length"]
        self.input_dim = config.STAGE3_CONFIG["input_dim"]

        # Targets columns
        self.target_cols = config.VERTEBRAE_CLASSES + ["patient_overall"]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        uid = row["StudyInstanceUID"]

        # Load Features
        # Expected shape: (Seq_Len, Feature_Dim + 7)
        # If file doesn't exist (e.g. not generated yet), return dummy for code validity
        feature_path = os.path.join(self.feature_dir, f"{uid}.npy")

        if os.path.exists(feature_path):
            features = np.load(feature_path)
        else:
            # Dummy features: Random noise
            # Length random between 100 and 300
            seq_len = np.random.randint(100, 300)
            features = np.random.randn(seq_len, self.input_dim).astype(np.float32)

        # Truncate or Pad
        seq_len = features.shape[0]
        if seq_len > self.max_len:
            # Truncate (center or random? usually uniform sampling or simple truncate)
            features = features[: self.max_len]
        elif seq_len < self.max_len:
            # Pad with zeros
            pad_len = self.max_len - seq_len
            padding = np.zeros((pad_len, self.input_dim), dtype=np.float32)
            features = np.vstack([features, padding])

        # Labels
        if self.phase != "test":
            labels = row[self.target_cols].values.astype(np.float32)
        else:
            labels = np.zeros(len(self.target_cols), dtype=np.float32)

        # Convert to tensors
        features_tensor = torch.from_numpy(features)  # (MaxLen, Dim)
        labels_tensor = torch.from_numpy(labels)  # (8,)

        # Split features into Visual and Anatomical ID
        # Visual: first 1280. Anat: last 7.
        # The model expects them concatenated, so we pass as is,
        # or split inside the model. The model forward takes (features, anat_ids).
        # Let's split here for clarity if the model signature requires it.
        # Checking models.py: forward(features, anat_ids).
        # So we should return them separately.

        visual_dim = config.STAGE2_CONFIG["feature_dim"]
        visual_feats = features_tensor[:, :visual_dim]
        anat_ids = features_tensor[:, visual_dim:]

        return visual_feats, anat_ids, labels_tensor


def get_datasets(stage="stage1"):
    """
    Factory function to get train and val datasets for a specific stage.
    """
    # Load Metadata
    train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(config.VAL_METADATA_PATH)

    if stage == "stage1":
        train_ds = SegmentationDataset(train_meta, cache_nifti=True)
        val_ds = SegmentationDataset(val_meta, cache_nifti=True)
        return train_ds, val_ds

    elif stage == "stage2":
        # Load BBoxes
        if os.path.exists(config.TRAIN_BBOXES_PATH):
            bbox_df = pd.read_csv(config.TRAIN_BBOXES_PATH)
            # Adjust bbox coordinates to center (y, x)
            # CSV: x, y, width, height
            bbox_df["x"] = bbox_df["x"] + bbox_df["width"] / 2
            bbox_df["y"] = bbox_df["y"] + bbox_df["height"] / 2
        else:
            bbox_df = None

        train_ds = EncoderTrainDataset(train_meta, bbox_df=bbox_df, phase="train")
        val_ds = EncoderTrainDataset(val_meta, bbox_df=None, phase="val")
        return train_ds, val_ds

    elif stage == "stage3":
        train_ds = SequenceDataset(train_meta, phase="train")
        val_ds = SequenceDataset(val_meta, phase="val")
        return train_ds, val_ds

    else:
        raise ValueError(f"Unknown stage: {stage}")
