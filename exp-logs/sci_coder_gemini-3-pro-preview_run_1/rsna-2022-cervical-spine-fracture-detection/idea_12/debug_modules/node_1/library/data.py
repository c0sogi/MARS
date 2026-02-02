import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import pydicom

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    nib = None
    HAS_NIBABEL = False

from library.config import Config
from library.utils import window_dicom

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------


def load_dicom_array(path, size=None):
    """
    Loads a DICOM file, applies windowing, and optionally resizes.
    """
    try:
        ds = pydicom.dcmread(path)
        pixel_array = ds.pixel_array
        pixel_array = window_dicom(pixel_array)

        if size is not None:
            if pixel_array.shape[0] != size or pixel_array.shape[1] != size:
                pixel_array = cv2.resize(
                    pixel_array, (size, size), interpolation=cv2.INTER_LINEAR
                )

        return pixel_array
    except Exception as e:
        # Fallback for corrupt or missing files
        # print(f"Error loading DICOM {path}: {e}")
        if size is None:
            size = 512
        return np.zeros((size, size), dtype=np.float32)


def prepare_slice_dataframe(load_cached_data=True):
    """
    Prepares a dataframe for Stage 2 (Slice Classification).
    Combines metadata with bounding boxes to create positive/negative samples.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "slice_classification_df.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Load metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    bbox_df = pd.read_csv(Config.TRAIN_BBOX_PATH)

    # Create a mapping of StudyUID -> List of BBoxes
    # BBox format: x, y, width, height, slice_number
    # Note: bbox_df has 'slice_number'

    slice_data = []

    # Iterate over training studies
    for idx, row in train_meta.iterrows():
        study_id = row["StudyInstanceUID"]
        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        if not os.path.exists(image_dir):
            continue

        # Get all slices
        slice_files = sorted(glob.glob(os.path.join(image_dir, "*.dcm")))

        # Get bboxes for this study
        study_bboxes = bbox_df[bbox_df["StudyInstanceUID"] == study_id]

        # Create a map of slice_number -> bbox
        # slice_number in bbox_df usually corresponds to the instance number in DICOM filename (e.g. 10.dcm -> 10)
        fractured_slices = {}
        for _, bbox_row in study_bboxes.iterrows():
            s_num = int(bbox_row["slice_number"])
            fractured_slices[s_num] = [
                bbox_row["x"],
                bbox_row["y"],
                bbox_row["width"],
                bbox_row["height"],
            ]

        for f_path in slice_files:
            file_name = os.path.basename(f_path)
            try:
                slice_num = int(file_name.replace(".dcm", ""))
            except:
                continue

            is_fractured = slice_num in fractured_slices

            # For positive samples, we keep them all.
            # For negative samples, we downsample to avoid huge imbalance
            # Logic: Keep all positives, keep 10% of negatives or 2x positives

            if is_fractured:
                bbox = fractured_slices[slice_num]
                slice_data.append(
                    {
                        "StudyInstanceUID": study_id,
                        "image_path": f_path,
                        "slice_num": slice_num,
                        "is_fractured": 1,
                        "x": bbox[0],
                        "y": bbox[1],
                        "w": bbox[2],
                        "h": bbox[3],
                    }
                )
            else:
                # Simple downsampling strategy:
                # Only keep every 10th negative slice to reduce dataset size for this demo
                if slice_num % 10 == 0:
                    slice_data.append(
                        {
                            "StudyInstanceUID": study_id,
                            "image_path": f_path,
                            "slice_num": slice_num,
                            "is_fractured": 0,
                            "x": np.nan,
                            "y": np.nan,
                            "w": np.nan,
                            "h": np.nan,
                        }
                    )

    df = pd.DataFrame(slice_data)

    # Save to cache
    df.to_parquet(cache_path, index=False)

    return df


# ---------------------------------------------------------
# Stage 1: Segmentation Dataset
# ---------------------------------------------------------


class SegmentationDataset(Dataset):
    def __init__(self, metadata_df, transform=None):
        """
        Args:
            metadata_df: DataFrame containing 'StudyInstanceUID'.
                         If HAS_NIBABEL, filtered by 'has_segmentation'.
                         Else, filtered by 'has_bounding_box'.
            transform: Optional albumentations transform.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.bbox_df = None

        if not HAS_NIBABEL:
            # Load bounding boxes for fallback
            if os.path.exists(Config.TRAIN_BBOX_PATH):
                self.bbox_df = pd.read_csv(Config.TRAIN_BBOX_PATH)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]
        img_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        # -----------------------------------------------------
        # Path A: Use NIFTI Segmentation (Preferred)
        # -----------------------------------------------------
        if HAS_NIBABEL:
            seg_path = os.path.join(Config.INPUT_DIR, row["segmentation_path"])
            try:
                nii = nib.load(seg_path)
                nii_data = nii.get_fdata()
                # Re-orient: Sagittal -> Axial approx
                nii_data = nii_data[:, ::-1, ::-1].transpose(2, 1, 0)
            except Exception:
                return self.dummy_item()

            dcm_files = glob.glob(os.path.join(img_dir, "*.dcm"))
            dcm_files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))

            num_slices = len(dcm_files)
            num_seg_slices = nii_data.shape[0]

            # Select random slice with annotation
            attempts = 0
            selected_idx = 0
            while attempts < 10:
                selected_idx = np.random.randint(0, min(num_slices, num_seg_slices))
                if np.sum(nii_data[selected_idx]) > 0:
                    break
                attempts += 1

            dcm_path = dcm_files[selected_idx]
            image = load_dicom_array(dcm_path, size=Config.IMG_SIZE_MODEL)

            mask = nii_data[selected_idx]
            mask = cv2.resize(
                mask,
                (Config.IMG_SIZE_MODEL, Config.IMG_SIZE_MODEL),
                interpolation=cv2.INTER_NEAREST,
            )

        # -----------------------------------------------------
        # Path B: Fallback using Bounding Boxes
        # -----------------------------------------------------
        else:
            if self.bbox_df is None:
                return self.dummy_item()

            # Get bboxes for this study
            study_bboxes = self.bbox_df[self.bbox_df["StudyInstanceUID"] == study_id]
            if len(study_bboxes) == 0:
                return self.dummy_item()

            # Pick a random slice that has a bbox
            sample_row = study_bboxes.sample(1).iloc[0]
            slice_num = int(sample_row["slice_number"])
            dcm_path = os.path.join(img_dir, f"{slice_num}.dcm")

            if not os.path.exists(dcm_path):
                return self.dummy_item()

            image = load_dicom_array(dcm_path, size=Config.IMG_SIZE_MODEL)

            # Create Mask from BBox
            # Initialize with 0 (Background)
            mask = np.zeros(
                (Config.IMG_SIZE_MODEL, Config.IMG_SIZE_MODEL), dtype=np.float32
            )

            # Determine Class Label (1-7)
            # We check the study-level labels in self.df
            # If C1 is 1, we use 1. If multiple, we pick the first one found.
            label_val = 1  # Default
            for c_idx, c_col in enumerate(
                ["C1", "C2", "C3", "C4", "C5", "C6", "C7"], start=1
            ):
                if c_col in row and row[c_col] == 1:
                    label_val = c_idx
                    break

            # Scale BBox to Model Size
            scale = Config.IMG_SIZE_MODEL / Config.IMG_SIZE_ORIG
            x = int(sample_row["x"] * scale)
            y = int(sample_row["y"] * scale)
            w = int(sample_row["width"] * scale)
            h = int(sample_row["height"] * scale)

            # Draw filled rectangle
            cv2.rectangle(mask, (x, y), (x + w, y + h), float(label_val), -1)

        # -----------------------------------------------------
        # Common Post-Processing
        # -----------------------------------------------------
        # Normalize image
        image = (image - Config.PIXEL_MEAN) / Config.PIXEL_STD
        image = np.expand_dims(image, axis=0)  # (1, H, W)

        # Convert to tensors
        mask = torch.tensor(mask, dtype=torch.long)
        image = torch.tensor(image, dtype=torch.float32)

        return image, mask

    def dummy_item(self):
        img = torch.zeros(
            (1, Config.IMG_SIZE_MODEL, Config.IMG_SIZE_MODEL), dtype=torch.float32
        )
        mask = torch.zeros(
            (Config.IMG_SIZE_MODEL, Config.IMG_SIZE_MODEL), dtype=torch.long
        )
        return img, mask


# ---------------------------------------------------------
# Stage 2: Dual-Stream Slice Dataset
# ---------------------------------------------------------


class DualStreamSliceDataset(Dataset):
    def __init__(self, slice_df, transform=None, phase="train"):
        """
        Args:
            slice_df: DataFrame from prepare_slice_dataframe.
            transform: Optional transforms.
            phase: 'train' or 'test'.
        """
        self.df = slice_df.reset_index(drop=True)
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        # 1. Load Original Image (Full Res 512x512)
        # We load full size first to crop from high res
        image_orig = load_dicom_array(img_path, size=Config.IMG_SIZE_ORIG)

        # 2. Global Stream Input (Resize to 256)
        image_global = cv2.resize(
            image_orig,
            (Config.IMG_SIZE_MODEL, Config.IMG_SIZE_MODEL),
            interpolation=cv2.INTER_LINEAR,
        )
        image_global = (image_global - Config.PIXEL_MEAN) / Config.PIXEL_STD
        image_global = np.expand_dims(image_global, axis=0)  # (1, 256, 256)

        # 3. Local Stream Input (Crop 256 from 512)
        # Determine Crop Center
        if row["is_fractured"] == 1 and not np.isnan(row["x"]):
            # Use BBox center
            cx = row["x"] + row["w"] / 2
            cy = row["y"] + row["h"] / 2
        else:
            # Use Image Center (or ROI if available in df)
            cx, cy = Config.IMG_SIZE_ORIG // 2, Config.IMG_SIZE_ORIG // 2

        # Crop Logic
        crop_size = Config.IMG_SIZE_MODEL
        half_size = crop_size // 2

        start_x = int(np.clip(cx - half_size, 0, Config.IMG_SIZE_ORIG - crop_size))
        start_y = int(np.clip(cy - half_size, 0, Config.IMG_SIZE_ORIG - crop_size))

        image_local = image_orig[
            start_y : start_y + crop_size, start_x : start_x + crop_size
        ]

        # Ensure correct size (handle edge cases)
        if image_local.shape != (crop_size, crop_size):
            image_local = cv2.resize(image_local, (crop_size, crop_size))

        image_local = (image_local - Config.PIXEL_MEAN) / Config.PIXEL_STD

        # 4. Local Mask Channel
        # Ideally load from Stage 1 output. Here we use a dummy or heuristic.
        # Heuristic: Simple threshold to find bone-like structures in the crop
        mask_local = (image_local > 0.2).astype(np.float32)

        # Stack Local: (2, 256, 256)
        image_local_combined = np.stack([image_local, mask_local], axis=0)

        # 5. Label
        label = float(row["is_fractured"]) if "is_fractured" in row else 0.0

        return {
            "local": torch.tensor(image_local_combined, dtype=torch.float32),
            "global": torch.tensor(image_global, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
        }


# ---------------------------------------------------------
# Stage 3: Feature Sequence Dataset
# ---------------------------------------------------------


class FeatureSequenceDataset(Dataset):
    def __init__(self, metadata_df, feature_dir, phase="train"):
        """
        Args:
            metadata_df: DataFrame with StudyInstanceUID and targets.
            feature_dir: Directory containing .npy files (features).
            phase: 'train' or 'test'.
        """
        self.df = metadata_df
        self.feature_dir = feature_dir
        self.phase = phase

        # Target columns
        self.target_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]

        # Load Features
        # Expected shape: (Seq_Len, Feature_Dim)
        feat_path = os.path.join(self.feature_dir, f"{study_id}.npy")

        if os.path.exists(feat_path):
            try:
                data = np.load(feat_path, allow_pickle=True).item()
                features = data["features"]  # (Seq, 1280)
                probs = data["probs"]  # (Seq, 8) - Anatomical probs
            except:
                # Fallback for corrupt files
                features = np.zeros(
                    (100, 2560), dtype=np.float32
                )  # 2560 is DualStream dim
                probs = np.zeros((100, 8), dtype=np.float32)
        else:
            # Dummy data if missing
            features = np.zeros((100, 2560), dtype=np.float32)
            probs = np.zeros((100, 8), dtype=np.float32)

        # Pad or Truncate to fixed length for batching
        # Or return raw and use collate_fn. Let's use fixed length for simplicity.
        MAX_LEN = 300
        curr_len = features.shape[0]

        if curr_len > MAX_LEN:
            # Truncate (center crop preferred, but simple truncate for now)
            start = (curr_len - MAX_LEN) // 2
            features = features[start : start + MAX_LEN]
            probs = probs[start : start + MAX_LEN]
        elif curr_len < MAX_LEN:
            # Pad
            pad_len = MAX_LEN - curr_len
            features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
            probs = np.pad(probs, ((0, pad_len), (0, 0)), mode="constant")

        # Targets
        if self.phase == "train":
            labels = row[self.target_cols].values.astype(np.float32)
            return (
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(probs, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32),
            )
        else:
            return (
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(probs, dtype=torch.float32),
                study_id,
            )
