import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from library.config import Config
from library.utils import read_dicom

# Attempt to import nibabel for potential future NIFTI support
try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


def get_sorted_slice_files(study_dir):
    """
    Returns a sorted list of DICOM files in the directory.
    Assumes filenames are integers (e.g., '1.dcm', '10.dcm').
    """
    if not os.path.exists(study_dir):
        return []
    files = [f for f in os.listdir(study_dir) if f.endswith(".dcm")]
    # Sort by integer value of filename to ensure correct Z-ordering
    try:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        files.sort()
    return files


def process_slice_metadata(
    metadata_df, bbox_df=None, mode="train", load_cached_data=True
):
    """
    Generates a DataFrame where each row is a single slice.
    Caches the result to parquet to speed up subsequent initializations.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_slice_df.parquet")

    # 1. Load from Cache if requested and available
    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    slice_records = []

    # Pre-process bounding boxes for faster lookup
    bbox_lookup = {}
    if bbox_df is not None:
        # Group by StudyUID and slice_number
        # bbox_df columns: StudyInstanceUID, x, y, width, height, slice_number
        if "slice_number" in bbox_df.columns:
            grouped = bbox_df.groupby(["StudyInstanceUID", "slice_number"])
            for (uid, slice_num), group in grouped:
                # Store list of boxes for this slice
                boxes = group[["x", "y", "width", "height"]].values.tolist()
                bbox_lookup[(uid, slice_num)] = boxes

    # Iterate over studies
    # Use a subset for debugging if configured
    if Config.DEBUG:
        metadata_df = metadata_df.head(Config.DEBUG_DATASET_SIZE)

    for idx, row in metadata_df.iterrows():
        study_uid = row["StudyInstanceUID"]

        # Determine image directory based on mode
        if mode == "train":
            image_dir = os.path.join(Config.TRAIN_IMAGES_DIR, study_uid)
        elif mode == "test":
            image_dir = os.path.join(Config.TEST_IMAGES_DIR, study_uid)
        else:  # val
            image_dir = os.path.join(Config.TRAIN_IMAGES_DIR, study_uid)

        slice_files = get_sorted_slice_files(image_dir)

        for s_idx, s_file in enumerate(slice_files):
            # Assuming filename is slice_number.dcm
            try:
                slice_num = int(os.path.splitext(s_file)[0])
            except:
                slice_num = s_idx + 1

            # Check for bounding boxes
            boxes = bbox_lookup.get((study_uid, slice_num), [])
            has_fracture = len(boxes) > 0

            record = {
                "StudyInstanceUID": study_uid,
                "slice_file": s_file,
                "slice_num": slice_num,
                "slice_index": s_idx,  # 0-based index in the stack
                "image_dir": image_dir,
                "has_fracture": 1 if has_fracture else 0,
                "boxes": boxes,  # List of [x, y, w, h]
            }

            # Add patient-level labels if available (Train/Val only)
            if "patient_overall" in row:
                for col in Config.TARGET_COLS:
                    record[f"patient_{col}"] = row[col]

            slice_records.append(record)

    slice_df = pd.DataFrame(slice_records)

    # Save to cache
    slice_df.to_parquet(cache_path)

    return slice_df


class SegmentationDataset(Dataset):
    """
    Dataset for Stage 1: Spine Localization.
    Uses bounding boxes to generate binary masks for localization training.
    """

    def __init__(self, slice_df, transforms=None):
        # Filter only slices that have bounding boxes (positive samples for localization)
        # This ensures the localizer learns to find the spine/fracture area.
        self.df = slice_df[slice_df["has_fracture"] == 1].reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(row["image_dir"], row["slice_file"])

        # Load Image
        image = read_dicom(image_path, Config.WINDOW_CENTER, Config.WINDOW_WIDTH)

        # Create Mask
        mask = np.zeros(
            (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE), dtype=np.float32
        )

        boxes = row["boxes"]  # List of [x, y, w, h]
        if isinstance(boxes, np.ndarray):
            boxes = boxes.tolist()

        for box in boxes:
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            # Draw a filled rectangle representing the ROI
            cv2.rectangle(mask, (x, y), (x + w, y + h), 1.0, -1)

        # Add channel dimension
        # Image: (H, W) -> (1, H, W)
        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask


class FractureCropDataset(Dataset):
    """
    Dataset for Stage 2: Fracture Classification on Crops.
    Loads 2.5D stacks (3 slices) and crops based on provided coordinates.
    """

    def __init__(self, slice_df, coords_map=None, transforms=None, mode="train"):
        self.df = slice_df
        self.coords_map = coords_map  # Dict: (StudyUID, slice_num) -> (x, y)
        self.transforms = transforms
        self.mode = mode

        # If training, balance the dataset
        if mode == "train":
            positives = self.df[self.df["has_fracture"] == 1]
            negatives = self.df[self.df["has_fracture"] == 0]

            # Balance strategy: Keep all positives, subsample negatives (e.g., 1:1 ratio)
            if len(positives) > 0:
                n_samples = len(positives)
                if len(negatives) > n_samples:
                    negatives = negatives.sample(n=n_samples, random_state=Config.SEED)

                self.df = (
                    pd.concat([positives, negatives])
                    .sample(frac=1, random_state=Config.SEED)
                    .reset_index(drop=True)
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        slice_num = row["slice_num"]
        image_dir = row["image_dir"]

        # 2.5D Stacking: Load current, prev, next
        slices_to_load = [slice_num - 1, slice_num, slice_num + 1]
        images = []

        for s_num in slices_to_load:
            path = os.path.join(image_dir, f"{s_num}.dcm")
            if not os.path.exists(path):
                # Fallback: use the current slice (clamping boundary)
                path = os.path.join(image_dir, f"{slice_num}.dcm")

            img = read_dicom(path, Config.WINDOW_CENTER, Config.WINDOW_WIDTH)
            images.append(img)

        # Stack: (H, W, 3)
        stack = np.stack(images, axis=-1)

        # Determine Crop Center
        center_x, center_y = (
            Config.ORIGINAL_IMAGE_SIZE // 2,
            Config.ORIGINAL_IMAGE_SIZE // 2,
        )

        # Priority 1: Use predicted coordinates from Stage 1
        if self.coords_map and (study_uid, slice_num) in self.coords_map:
            center_x, center_y = self.coords_map[(study_uid, slice_num)]
        # Priority 2: Use Ground Truth BBox center (Training only)
        elif self.mode == "train" and row["has_fracture"] == 1:
            boxes = row["boxes"]
            if len(boxes) > 0:
                bx, by, bw, bh = boxes[0]
                center_x = bx + bw // 2
                center_y = by + bh // 2

        # Perform Crop
        crop_size = Config.CROP_SIZE
        half_size = crop_size // 2

        start_x = int(max(0, center_x - half_size))
        start_y = int(max(0, center_y - half_size))
        end_x = start_x + crop_size
        end_y = start_y + crop_size

        # Adjust boundaries
        if end_x > Config.ORIGINAL_IMAGE_SIZE:
            end_x = Config.ORIGINAL_IMAGE_SIZE
            start_x = end_x - crop_size
        if end_y > Config.ORIGINAL_IMAGE_SIZE:
            end_y = Config.ORIGINAL_IMAGE_SIZE
            start_y = end_y - crop_size

        start_x = max(0, start_x)
        start_y = max(0, start_y)

        crop = stack[start_y:end_y, start_x:end_x, :]

        # Resize if crop is invalid (rare edge case)
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            crop = cv2.resize(crop, (crop_size, crop_size))

        # To Tensor: (3, H, W)
        crop = torch.tensor(crop).permute(2, 0, 1)

        # Prepare Labels
        labels = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)

        if self.mode != "test":
            if row["has_fracture"] == 1:
                # Weak Labeling: Assign patient-level labels to fractured slices
                for i, col in enumerate(Config.TARGET_COLS):
                    col_name = f"patient_{col}"
                    if col_name in row:
                        labels[i] = row[col_name]

        return crop, labels
