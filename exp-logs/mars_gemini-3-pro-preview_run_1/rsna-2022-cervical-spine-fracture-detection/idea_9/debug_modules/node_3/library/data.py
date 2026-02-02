import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import configuration and utilities from the provided library
from library.config import Config
from library.utils import load_dicom, apply_windowing, get_roi_coordinates

# Optional import for NIfTI files
try:
    import nibabel as nib
except ImportError:
    nib = None

# =============================================================================
# Dataset Classes
# =============================================================================


class SegmentationDataset(Dataset):
    """
    Dataset for Stage 1: 2D U-Net Training.
    Loads pre-processed .npy slice files (image + mask).
    """

    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load pre-processed numpy arrays
        # Format: [H, W]
        try:
            data = np.load(row["npy_path"], allow_pickle=True).item()
            image = data["image"]
            mask = data["mask"]
        except Exception as e:
            # Fallback for robustness
            print(f"Error loading {row['npy_path']}: {e}")
            image = np.zeros(Config.SEG_IMG_SIZE, dtype=np.float32)
            mask = np.zeros(Config.SEG_IMG_SIZE, dtype=np.uint8)

        # Augmentations
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        else:
            # Convert to tensor if no transforms provided
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


class DualStreamDataset(Dataset):
    """
    Dataset for Stage 2: Dual-Branch Feature Encoder.
    Provides:
    1. Local Stream: High-res crop (224x224) + Bone Mask (4 channels).
    2. Global Stream: Resized full slice (224x224) (3 channels).
    """

    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Define internal resize for global stream
        self.global_resize = A.Compose(
            [A.Resize(Config.GLOBAL_SIZE[0], Config.GLOBAL_SIZE[1]), ToTensorV2()]
        )

        # Define internal resize/crop for local stream if not handled by external transforms
        self.local_resize = A.Compose(
            [
                A.Resize(Config.LOCAL_CROP_SIZE[0], Config.LOCAL_CROP_SIZE[1]),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dcm_path = row["image_path"]

        # 1. Load and Window Image
        try:
            img_hu = load_dicom(dcm_path)
            img = apply_windowing(img_hu, Config.WINDOW_CENTER, Config.WINDOW_WIDTH)
        except Exception:
            img = np.zeros((Config.IMG_SIZE_H, Config.IMG_SIZE_W), dtype=np.float32)

        # 2. Prepare Global Stream (Resize -> 3ch)
        # Stack to 3 channels for backbone compatibility
        img_3ch = np.stack([img, img, img], axis=-1)
        global_input = self.global_resize(image=img_3ch)["image"]

        # 3. Prepare Local Stream (Crop -> 4ch)
        # Determine Crop Center
        h, w = img.shape
        if "roi_x" in row and not pd.isna(row["roi_x"]):
            cx, cy = int(row["roi_x"]), int(row["roi_y"])
        else:
            cx, cy = w // 2, h // 2

        crop_h, crop_w = Config.LOCAL_CROP_SIZE

        # Simple center crop logic around ROI
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)

        # Adjust if out of bounds
        if x2 - x1 < crop_w:
            x1 = max(0, w - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, h - crop_h)

        local_crop = img_3ch[y1 : y1 + crop_h, x1 : x1 + crop_w, :]

        # Ensure correct size (pad if image was smaller than crop)
        if local_crop.shape[0] != crop_h or local_crop.shape[1] != crop_w:
            local_crop = cv2.resize(local_crop, (crop_w, crop_h))

        # 4. Prepare Bone Mask (4th channel)
        # If we have a GT mask path (training), use it. Else threshold.
        mask_crop = None
        if (
            "mask_path" in row
            and pd.notna(row["mask_path"])
            and os.path.exists(row["mask_path"])
        ):
            try:
                # Assuming mask is saved as numpy or image
                mask_full = np.load(row["mask_path"])
                mask_crop = mask_full[y1 : y1 + crop_h, x1 : x1 + crop_w]
            except:
                pass

        if mask_crop is None:
            # Heuristic: Bone is > 0.4 after windowing (approx > 300 HU)
            # Window: 400 center, 1800 width. Range [-500, 1300].
            # 0.5 is 400 HU.
            mask_crop = (local_crop[..., 0] > 0.5).astype(np.float32)
        else:
            # Binarize GT mask for the channel
            mask_crop = (mask_crop > 0).astype(np.float32)

        # Resize mask to ensure match
        if mask_crop.shape[:2] != (crop_h, crop_w):
            mask_crop = cv2.resize(
                mask_crop, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
            )

        # Concatenate: (H, W, 3) + (H, W, 1) -> (H, W, 4)
        local_input_np = np.concatenate(
            [local_crop, mask_crop[..., np.newaxis]], axis=-1
        )

        # Apply transforms and convert to tensor
        if self.transforms:
            local_input = self.transforms(image=local_input_np)["image"]
        else:
            local_input = self.local_resize(image=local_input_np)["image"]

        # 5. Target
        label = (
            torch.tensor(row["label"], dtype=torch.float32)
            if "label" in row
            else torch.tensor(0.0)
        )

        return (local_input, global_input), label


class SequenceDataset(Dataset):
    """
    Dataset for Stage 3: Bi-GRU Aggregator.
    Loads pre-computed feature sequences for a patient.
    """

    def __init__(self, df, feature_dir, mode="train"):
        self.df = df
        self.feature_dir = feature_dir
        self.mode = mode
        self.study_ids = self.df["StudyInstanceUID"].unique()

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]

        # Load features
        feat_path = os.path.join(self.feature_dir, f"{study_id}.npy")
        if os.path.exists(feat_path):
            # Shape: (Seq_Len, Feature_Dim + 7)
            features = np.load(feat_path)
        else:
            # Fallback: Zero sequence
            features = np.zeros((100, Config.RNN_INPUT_SIZE), dtype=np.float32)

        features = torch.tensor(features, dtype=torch.float32)

        # Load Targets
        if self.mode != "test":
            # Get labels from dataframe (assuming df is study-level or we aggregate)
            # We need 8 labels: C1-C7, patient_overall
            row = self.df[self.df["StudyInstanceUID"] == study_id].iloc[0]
            targets = [
                row["C1"],
                row["C2"],
                row["C3"],
                row["C4"],
                row["C5"],
                row["C6"],
                row["C7"],
                row["patient_overall"],
            ]
            targets = torch.tensor(targets, dtype=torch.float32)
        else:
            targets = torch.tensor([-1.0] * 8, dtype=torch.float32)  # Dummy

        return features, targets, study_id


# =============================================================================
# Data Processing & Caching
# =============================================================================


def process_segmentation_data(load_cached_data=True):
    """
    Prepares data for Stage 1.
    Extracts slices from NIfTI files and saves them as .npy for fast loading.
    """
    cache_file = os.path.join(Config.CACHE_DIR, "segmentation_index.parquet")
    output_dir = os.path.join(Config.CACHE_DIR, "segmentation_slices")
    os.makedirs(output_dir, exist_ok=True)

    if load_cached_data and os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    records = []

    # ---------------------------------------------------------
    # Fallback Strategy: Use Bounding Boxes if nibabel is missing
    # ---------------------------------------------------------
    if nib is None:
        print(
            "nibabel not installed. Generating segmentation data from Bounding Boxes."
        )

        if not os.path.exists(Config.TRAIN_BBOXES_PATH):
            print("No bounding boxes found. Cannot generate segmentation data.")
            return pd.DataFrame()

        df_bbox = pd.read_csv(Config.TRAIN_BBOXES_PATH)
        # Filter to studies in our current train metadata (e.g. mini_train)
        train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
        valid_uids = set(train_meta["StudyInstanceUID"].unique())
        df_bbox = df_bbox[df_bbox["StudyInstanceUID"].isin(valid_uids)]

        if df_bbox.empty:
            print("No bounding boxes match the current training set.")
            # Return empty df, but this might still crash if we need data.
            # In demo mode, we might need to be lenient.
            return pd.DataFrame()

        # Group by Study and Slice
        grouped = df_bbox.groupby(["StudyInstanceUID", "slice_number"])

        for (study_id, slice_num), group in grouped:
            try:
                # Construct image path
                # Metadata has relative path "train_images/UID"
                # We need to find the folder.
                study_rows = train_meta[train_meta["StudyInstanceUID"] == study_id]
                if study_rows.empty:
                    continue

                rel_path = study_rows.iloc[0]["image_path"]
                image_dir = os.path.join(Config.INPUT_DIR, rel_path)
                dcm_path = os.path.join(image_dir, f"{int(slice_num)}.dcm")

                # Load Image
                img_hu = load_dicom(dcm_path)
                img = apply_windowing(img_hu)
                h, w = img.shape

                # Create Mask
                mask = np.zeros((h, w), dtype=np.uint8)

                for _, row in group.iterrows():
                    x, y, bw, bh = row["x"], row["y"], row["width"], row["height"]
                    x1 = int(max(0, x))
                    y1 = int(max(0, y))
                    x2 = int(min(w, x + bw))
                    y2 = int(min(h, y + bh))

                    # Assign class 1 (Fracture) to the bbox area
                    # Since we don't have specific vertebra labels in bbox csv, use 1.
                    mask[y1:y2, x1:x2] = 1

                # Resize
                img_res = cv2.resize(img, Config.SEG_IMG_SIZE)
                mask_res = cv2.resize(
                    mask, Config.SEG_IMG_SIZE, interpolation=cv2.INTER_NEAREST
                )

                # Save
                save_name = f"{study_id}_{slice_num}.npy"
                save_path = os.path.join(output_dir, save_name)

                np.save(
                    save_path, {"image": img_res, "mask": mask_res.astype(np.uint8)}
                )

                records.append(
                    {
                        "StudyInstanceUID": study_id,
                        "slice_index": int(slice_num),
                        "npy_path": save_path,
                    }
                )

            except Exception as e:
                # print(f"Skipping {study_id} slice {slice_num}: {e}")
                continue

        df_result = pd.DataFrame(records)
        if not df_result.empty:
            df_result.to_parquet(cache_file)
        return df_result

    # ---------------------------------------------------------
    # Standard Strategy: Use NIfTI Segmentations
    # ---------------------------------------------------------
    print("Processing segmentation data from scratch (NIfTI)...")

    # Load train metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    # Filter for studies with segmentation
    df_seg = df_train[df_train["has_segmentation"] == True]

    for idx, row in df_seg.iterrows():
        study_id = row["StudyInstanceUID"]
        dcm_dir = os.path.join(Config.INPUT_DIR, row["image_path"])
        nii_path = os.path.join(Config.INPUT_DIR, row["segmentation_path"])

        if not os.path.exists(nii_path):
            continue

        try:
            # Load NIfTI
            nii = nib.load(nii_path)
            # Reorient to canonical (RAS) if possible to handle orientation
            nii = nib.as_closest_canonical(nii)
            mask_data = nii.get_fdata()  # (X, Y, Z) usually

            # Load DICOM file list to determine Z-axis matching
            dcm_files = glob.glob(os.path.join(dcm_dir, "*.dcm"))
            # Sort by slice number (assuming filename is number.dcm)
            dcm_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

            # Check dimensions
            # Usually NIfTI Z matches number of DICOM slices
            # If mask_data is (512, 512, Z), we iterate Z
            if mask_data.shape[2] == len(dcm_files):
                iterator = range(len(dcm_files))
                get_slice = lambda z: mask_data[:, :, z]
            elif mask_data.shape[0] == len(dcm_files):
                iterator = range(len(dcm_files))
                get_slice = lambda z: mask_data[z, :, :]
            else:
                # Mismatch, skip
                continue

            for z in iterator:
                mask_slice = get_slice(z)

                # Skip empty masks to save space/time (optional, but good for training UNet)
                if np.sum(mask_slice) == 0 and np.random.rand() > 0.1:
                    continue

                # Load corresponding DICOM
                dcm_path = dcm_files[z]
                img_hu = load_dicom(dcm_path)
                img = apply_windowing(img_hu)

                # Resize to SEG_IMG_SIZE
                img_res = cv2.resize(img, Config.SEG_IMG_SIZE)
                mask_res = cv2.resize(
                    mask_slice, Config.SEG_IMG_SIZE, interpolation=cv2.INTER_NEAREST
                )

                # Save
                save_name = f"{study_id}_{z}.npy"
                save_path = os.path.join(output_dir, save_name)

                np.save(
                    save_path, {"image": img_res, "mask": mask_res.astype(np.uint8)}
                )

                records.append(
                    {
                        "StudyInstanceUID": study_id,
                        "slice_index": z,
                        "npy_path": save_path,
                    }
                )

        except Exception as e:
            print(f"Failed to process {study_id}: {e}")
            continue

    df_result = pd.DataFrame(records)
    df_result.to_parquet(cache_file)
    return df_result


def process_classification_data(load_cached_data=True):
    """
    Prepares data for Stage 2 (Encoder).
    Creates a balanced dataset of slices:
    - Positive: Slices with bounding boxes.
    - Negative: Random slices from training set without boxes.
    """
    cache_file = os.path.join(Config.CACHE_DIR, "classification_index.parquet")

    if load_cached_data and os.path.exists(cache_file):
        return pd.read_parquet(cache_file)

    print("Processing classification data from scratch...")

    # 1. Load Bounding Boxes (Positives)
    if os.path.exists(Config.TRAIN_BBOXES_PATH):
        df_bbox = pd.read_csv(Config.TRAIN_BBOXES_PATH)
        # BBox df has: StudyInstanceUID, slice_number, x, y, width, height
        # Construct path
        df_bbox["image_path"] = df_bbox.apply(
            lambda row: os.path.join(
                Config.TRAIN_IMAGES_DIR,
                row["StudyInstanceUID"],
                f"{int(row['slice_number'])}.dcm",
            ),
            axis=1,
        )
        df_bbox["label"] = 1.0
        df_bbox["roi_x"] = df_bbox["x"] + df_bbox["width"] / 2
        df_bbox["roi_y"] = df_bbox["y"] + df_bbox["height"] / 2

        positives = df_bbox[
            ["StudyInstanceUID", "image_path", "label", "roi_x", "roi_y"]
        ]
    else:
        positives = pd.DataFrame(
            columns=["StudyInstanceUID", "image_path", "label", "roi_x", "roi_y"]
        )

    # 2. Sample Negatives
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    negatives_list = []

    # Sample 2x negatives
    n_neg = len(positives) * 2 if len(positives) > 0 else 1000

    studies = df_train["StudyInstanceUID"].unique()

    count = 0
    while count < n_neg:
        study = np.random.choice(studies)
        study_dir = os.path.join(Config.TRAIN_IMAGES_DIR, study)
        if not os.path.exists(study_dir):
            continue

        files = os.listdir(study_dir)
        if not files:
            continue

        f = np.random.choice(files)
        # Ensure it's not in positives
        slice_num = int(os.path.splitext(f)[0])

        # Check if this slice is in positives
        is_pos = False
        if len(positives) > 0:
            match = positives[
                (positives["StudyInstanceUID"] == study)
                & (positives["image_path"].str.endswith(f"/{slice_num}.dcm"))
            ]
            if not match.empty:
                is_pos = True

        if not is_pos:
            negatives_list.append(
                {
                    "StudyInstanceUID": study,
                    "image_path": os.path.join(study_dir, f),
                    "label": 0.0,
                    "roi_x": np.nan,  # Will default to center
                    "roi_y": np.nan,
                }
            )
            count += 1

    df_neg = pd.DataFrame(negatives_list)

    df_final = pd.concat([positives, df_neg], ignore_index=True)
    df_final.to_parquet(cache_file)
    return df_final


# =============================================================================
# Augmentations
# =============================================================================


def get_transforms(stage="train", img_size=(224, 224)):
    if stage == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Resize(img_size[0], img_size[1]),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Resize(img_size[0], img_size[1]), ToTensorV2()])
