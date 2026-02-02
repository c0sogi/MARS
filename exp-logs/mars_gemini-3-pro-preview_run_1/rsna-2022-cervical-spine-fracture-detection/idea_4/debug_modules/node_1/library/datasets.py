import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import nibabel as nib
from PIL import Image

# Import library modules
from library.config import Config
from library.utils import load_dicom_windowed, get_spine_crop_coords

# =========================================================================
# Helper: Segmentation Caching
# =========================================================================


def prepare_segmentation_cache(metadata_df, load_cached_data=True):
    """
    Extracts 2D slices from 3D NIFTI segmentation files and saves them as .npy files.
    This ensures fast random access during training.
    """
    cache_dir = os.path.join(Config.CACHE_DIR, "segmentation_slices")

    # Check if cache exists and we want to load it
    if (
        load_cached_data
        and os.path.exists(cache_dir)
        and len(os.listdir(cache_dir)) > 0
    ):
        # Assume valid if directory exists and is not empty
        return cache_dir

    # Compute from scratch
    os.makedirs(cache_dir, exist_ok=True)

    # Filter for studies that have segmentation
    seg_studies = metadata_df[metadata_df["has_segmentation"] == True]

    print(f"Processing {len(seg_studies)} segmentation files for caching...")

    for _, row in seg_studies.iterrows():
        study_uid = row["StudyInstanceUID"]
        seg_path = row["segmentation_path"]

        if not seg_path or pd.isna(seg_path):
            continue

        full_seg_path = os.path.join(Config.INPUT_DIR, seg_path)
        if not os.path.exists(full_seg_path):
            continue

        try:
            # Load NIFTI
            nii = nib.load(full_seg_path)
            # Reorient to canonical (RAS+) to match standard DICOM orientation roughly
            nii = nib.as_closest_canonical(nii)
            data = nii.get_fdata()

            # Data shape is usually (H, W, D) or (W, H, D).
            # We assume the last dimension is the Z-axis (slices).
            # We also need to handle the fact that DICOMs might be inverted in Z.
            # However, matching slice by slice index is the standard heuristic if
            # DICOMs are sorted by ImagePositionPatient Z.
            # Here we save all slices by index.

            num_slices = data.shape[2]

            # Create study dir
            study_cache_dir = os.path.join(cache_dir, study_uid)
            os.makedirs(study_cache_dir, exist_ok=True)

            for z in range(num_slices):
                # Extract slice
                slice_mask = data[:, :, z]

                # NIFTI (X, Y) often maps to DICOM (Y, X) transposed.
                # We rotate/transpose to align with standard pydicom loading.
                # A common transform is rotation by 90 deg or transpose.
                # Without visual feedback, we assume a transpose aligns (W, H) -> (H, W).
                slice_mask = np.transpose(slice_mask)

                # Binarize: The task is spine localization.
                # Labels 1-7 (Cervical) and 8-19 (Thoracic).
                # We treat anything > 0 as "Spine".
                binary_mask = (slice_mask > 0).astype(np.uint8)

                # Save only if there is a mask to save space, or save all?
                # We save all to maintain indexing.
                save_path = os.path.join(study_cache_dir, f"{z}.npy")
                np.save(save_path, binary_mask)

        except Exception as e:
            print(f"Error processing segmentation for {study_uid}: {e}")

    return cache_dir


# =========================================================================
# Dataset 1: Segmentation Dataset (Stage 1)
# =========================================================================


class SegmentationDataset(Dataset):
    def __init__(self, metadata_df, transform=None, load_cached_data=True):
        """
        Args:
            metadata_df: DataFrame containing training metadata.
            transform: Albumentations transforms.
            load_cached_data: Boolean to use cached masks.
        """
        self.transform = transform

        # Filter only studies with segmentation
        self.df = metadata_df[metadata_df["has_segmentation"] == True].reset_index(
            drop=True
        )

        # Prepare cache
        self.cache_dir = prepare_segmentation_cache(
            metadata_df, load_cached_data=load_cached_data
        )

        # Index all available slices
        # We need to map global index -> (StudyUID, SliceIndex, DicomPath)
        self.samples = []

        for _, row in self.df.iterrows():
            study_uid = row["StudyInstanceUID"]
            image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

            # Get all DICOM files
            if not os.path.exists(image_dir):
                continue

            dcm_files = glob.glob(os.path.join(image_dir, "*.dcm"))

            # Sort DICOMs to match NIFTI Z-ordering
            # Ideally we sort by ImagePositionPatient[2], but parsing all is slow.
            # We assume filename integer sorting matches instance number/location.
            # e.g. 1.dcm, 2.dcm...
            try:
                dcm_files.sort(
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                )
            except ValueError:
                dcm_files.sort()

            # Check which masks exist in cache
            study_cache_path = os.path.join(self.cache_dir, study_uid)
            if not os.path.exists(study_cache_path):
                continue

            # We only include slices that have a corresponding mask file
            # AND (optionally) we might only train on slices with positive masks to balance?
            # For localization, we want to learn the spine everywhere.
            for i, dcm_path in enumerate(dcm_files):
                mask_path = os.path.join(study_cache_path, f"{i}.npy")
                if os.path.exists(mask_path):
                    self.samples.append((study_uid, dcm_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        study_uid, dcm_path, mask_path = self.samples[idx]

        # Load Image
        image = load_dicom_windowed(dcm_path)  # (H, W) normalized [0,1]

        # Load Mask
        mask = np.load(mask_path)  # (H, W)

        # Resize if necessary (Config.ORIGINAL_IMAGE_SIZE is 512)
        if image.shape != (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE):
            image = cv2.resize(
                image, (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE)
            )

        if mask.shape != (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE):
            mask = cv2.resize(
                mask,
                (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to Tensor
        # Image: (1, H, W)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        # Mask: (1, H, W)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask


# =========================================================================
# Dataset 2: Cropped Slice Dataset (Stage 2)
# =========================================================================


class CroppedSliceDataset(Dataset):
    def __init__(self, metadata_df, mode="train", transform=None, mask_dir=None):
        """
        Args:
            metadata_df: DataFrame with study metadata.
            mode: 'train', 'val', or 'test'.
            transform: Albumentations transforms.
            mask_dir: Directory containing predicted masks (from Stage 1).
                      If None, tries to use GT or center crop.
        """
        self.metadata_df = metadata_df
        self.mode = mode
        self.transform = transform
        self.mask_dir = mask_dir

        # Load Bounding Boxes for labels
        self.bbox_df = None
        if os.path.exists(Config.TRAIN_BBOX_PATH):
            self.bbox_df = pd.read_csv(Config.TRAIN_BBOX_PATH)

        # Construct sample list
        # For training, we want a balanced set of slices.
        # For inference (test), we iterate all slices (or a stride).
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []

        # Optimize: Create a set of fractured (Study, Slice) tuples
        fractured_slices = set()
        if self.bbox_df is not None:
            for _, row in self.bbox_df.iterrows():
                fractured_slices.add(
                    (row["StudyInstanceUID"], int(row["slice_number"]))
                )

        for _, row in self.metadata_df.iterrows():
            study_uid = row["StudyInstanceUID"]
            image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

            if not os.path.exists(image_dir):
                continue

            # Get list of slices (filenames are integers)
            files = os.listdir(image_dir)
            slice_indices = []
            for f in files:
                if f.endswith(".dcm"):
                    try:
                        slice_indices.append(int(os.path.splitext(f)[0]))
                    except ValueError:
                        pass
            slice_indices.sort()

            if not slice_indices:
                continue

            # Selection Strategy
            if self.mode == "train":
                # Use all positive slices
                # Sample negative slices (e.g., 1:1 or 1:2 ratio)
                # For this implementation, we'll take all positives and a subset of negatives
                study_positives = [
                    s for s in slice_indices if (study_uid, s) in fractured_slices
                ]
                study_negatives = [
                    s for s in slice_indices if (study_uid, s) not in fractured_slices
                ]

                # Simple balancing: Take all positives, and equal number of random negatives
                # If no positives, take a few random negatives
                if study_positives:
                    selected_negatives = np.random.choice(
                        study_negatives,
                        size=min(len(study_negatives), len(study_positives) * 2),
                        replace=False,
                    )
                    selected_slices = study_positives + selected_negatives.tolist()
                else:
                    # Take a few random slices per healthy patient
                    selected_slices = np.random.choice(
                        study_negatives,
                        size=min(len(study_negatives), 10),
                        replace=False,
                    ).tolist()
            else:
                # Validation/Test: Use all slices (or stride for speed if allowed)
                selected_slices = slice_indices

            for s_idx in selected_slices:
                label = 1.0 if (study_uid, s_idx) in fractured_slices else 0.0
                samples.append(
                    {
                        "study_uid": study_uid,
                        "slice_idx": s_idx,
                        "image_dir": image_dir,
                        "label": label,
                        "max_slice": slice_indices[-1],
                    }
                )

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_slice(self, image_dir, slice_idx, max_slice):
        # Clamp index
        idx = max(1, min(slice_idx, max_slice))
        path = os.path.join(image_dir, f"{idx}.dcm")
        return load_dicom_windowed(path)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        study_uid = sample["study_uid"]
        slice_idx = sample["slice_idx"]
        image_dir = sample["image_dir"]
        max_slice = sample["max_slice"]

        # 1. Load 3-slice stack (Context)
        # Channels: [Slice-1, Slice, Slice+1]
        img_prev = self._load_slice(image_dir, slice_idx - 1, max_slice)
        img_curr = self._load_slice(image_dir, slice_idx, max_slice)
        img_next = self._load_slice(image_dir, slice_idx + 1, max_slice)

        # Stack: (H, W, 3)
        image_stack = np.stack([img_prev, img_curr, img_next], axis=-1)

        # 2. Load Mask
        mask = None
        # Try loading from mask_dir (Predicted masks)
        if self.mask_dir:
            mask_path = os.path.join(
                self.mask_dir, study_uid, f"{slice_idx}.npy"
            )  # Assuming specific structure
            # Note: The cache structure in SegmentationDataset was by index 0..N.
            # Here we need to match filename indices.
            # We assume the mask generation step aligned these.
            if os.path.exists(mask_path):
                mask = np.load(mask_path)

        # Fallback: Empty mask (or center crop logic later handles it)
        if mask is None:
            mask = np.zeros(
                (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE),
                dtype=np.float32,
            )

        # 3. Crop
        # Calculate crop coordinates based on mask
        y_min, y_max, x_min, x_max = get_spine_crop_coords(
            mask, image_size=Config.IMAGE_SIZE
        )

        # Apply crop to image stack
        image_crop = image_stack[y_min:y_max, x_min:x_max, :]
        # Apply crop to mask
        mask_crop = mask[y_min:y_max, x_min:x_max]

        # Resize to target size (in case crop was smaller due to edges)
        if image_crop.shape[:2] != (Config.IMAGE_SIZE, Config.IMAGE_SIZE):
            image_crop = cv2.resize(image_crop, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            mask_crop = cv2.resize(
                mask_crop,
                (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )

        # 4. Final Input Construction
        # If using mask input, concat as 4th channel
        if Config.USE_MASK_INPUT:
            # Mask needs to be (H, W, 1)
            mask_crop = np.expand_dims(mask_crop, axis=-1)
            input_tensor = np.concatenate([image_crop, mask_crop], axis=-1)
        else:
            input_tensor = image_crop

        # Normalize/Transform
        # Albumentations expects (H, W, C)
        if self.transform:
            augmented = self.transform(image=input_tensor)
            input_tensor = augmented["image"]
        else:
            # To Tensor (C, H, W)
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32).permute(
                2, 0, 1
            )

        label = torch.tensor(sample["label"], dtype=torch.float32)

        return input_tensor, label


# =========================================================================
# Dataset 3: Feature Sequence Dataset (Stage 3)
# =========================================================================


class FeatureSequenceDataset(Dataset):
    def __init__(self, metadata_df, feature_dir):
        """
        Args:
            metadata_df: DataFrame with one row per study.
            feature_dir: Directory containing .npy feature files (StudyUID.npy).
        """
        self.metadata_df = metadata_df
        self.feature_dir = feature_dir
        self.target_cols = Config.TARGET_COLS  # ['C1', ..., 'patient_overall']

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        study_uid = row["StudyInstanceUID"]

        # Load Features
        feature_path = os.path.join(self.feature_dir, f"{study_uid}.npy")

        if os.path.exists(feature_path):
            features = np.load(feature_path)  # (Seq_Len, Feat_Dim)
        else:
            # Fallback for missing data (should not happen in valid pipeline)
            features = np.zeros((10, Config.ENCODER_HIDDEN_DIM), dtype=np.float32)

        seq_len = features.shape[0]

        # Pad or Truncate
        max_len = Config.MAX_SEQ_LEN

        if seq_len > max_len:
            # Truncate (center or uniform? usually simple truncation or sampling)
            # Uniform sampling to keep context from whole spine
            indices = np.linspace(0, seq_len - 1, max_len).astype(int)
            features = features[indices]
            mask = np.ones(max_len, dtype=np.float32)
        else:
            # Pad
            padding = np.zeros((max_len - seq_len, features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, padding], axis=0)

            mask = np.zeros(max_len, dtype=np.float32)
            mask[:seq_len] = 1.0

        # Labels
        # If labels exist (Train/Val), load them. Else zeros (Test).
        labels = []
        if "patient_overall" in row:
            for col in self.target_cols:
                labels.append(row[col])
            labels = np.array(labels, dtype=np.float32)
        else:
            labels = np.zeros(len(self.target_cols), dtype=np.float32)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )
