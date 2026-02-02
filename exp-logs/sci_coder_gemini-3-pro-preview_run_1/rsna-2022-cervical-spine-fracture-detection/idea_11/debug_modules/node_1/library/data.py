import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import warnings
from library.config import Config
from library.utils import load_dicom, cache_data

# Attempt to import nibabel for NIFTI processing
try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

# =============================================================================
# Preprocessing / Caching Functions
# =============================================================================


def prepare_segmentation_cache(load_cached_data=True):
    """
    Preprocesses NIFTI segmentation files into individual slice .npy files
    stored in the cache directory. Returns a DataFrame indexing these slices.
    """
    cache_file = "segmentation_index.parquet"

    # Try loading from cache
    if load_cached_data:
        df = cache_data(cache_file, load_cached_data=True)
        if df is not None:
            return df

    print("Building segmentation cache from source NIFTI files...")

    if not HAS_NIBABEL:
        print("Warning: nibabel not installed. Cannot process NIFTI files.")
        return pd.DataFrame()

    meta_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    seg_studies = meta_df[meta_df["has_segmentation"] == True]

    slice_records = []
    output_dir = os.path.join(Config.CACHE_DIR, "segmentation_slices")
    os.makedirs(output_dir, exist_ok=True)

    for _, row in seg_studies.iterrows():
        study_uid = row["StudyInstanceUID"]
        seg_path = os.path.join(Config.INPUT_DIR, row["segmentation_path"])
        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        try:
            # Load NIFTI
            nii = nib.load(seg_path)
            # Reorient to canonical (RAS) to attempt to match DICOM orientation
            nii = nib.as_closest_canonical(nii)
            vol = nii.get_fdata()
            # NIFTI is usually (H, W, D) or (W, H, D).
            # We assume the last dimension corresponds to the Z-axis (slices).
            # Note: Orientation matching between NIFTI and DICOM is complex.
            # We assume the Z-dimension order matches the DICOM instance numbers
            # but NIFTI might be reversed.
            # For this implementation, we assume a direct mapping Z index -> Instance Number - 1
            # (or reverse depending on the dataset specifics, here we map linearly).

            num_slices = vol.shape[2]

            # Get DICOM files to map instance numbers
            dcm_files = glob.glob(os.path.join(image_dir, "*.dcm"))
            # Map instance number to filename
            instance_map = {}
            for f in dcm_files:
                try:
                    # Filename is usually 'slice_number.dcm' or similar
                    # We rely on the filename being the instance number for speed
                    # or strictly we should read headers.
                    # RSNA dataset: filename is instance number.
                    fname = os.path.basename(f)
                    inst_num = int(os.path.splitext(fname)[0])
                    instance_map[inst_num] = f
                except:
                    continue

            # Determine direction.
            # If NIFTI z=0 matches min instance number or max?
            # We assume standard ascending order for now.
            sorted_instances = sorted(instance_map.keys())

            # Safety check on dimensions
            if num_slices != len(sorted_instances):
                # If mismatch, we might skip or try to align.
                # For robustness, we take the minimum length.
                limit = min(num_slices, len(sorted_instances))
            else:
                limit = num_slices

            # Save non-empty slices to save space
            for z in range(limit):
                mask_slice = vol[:, :, z]
                if np.max(mask_slice) > 0:
                    # Rotate/Flip to match Axial DICOM if needed.
                    # NIFTI (sagittal/coronal base) -> Axial is often a transpose.
                    # Here we save as is, assuming canonical handled it,
                    # but typically requires visual verification.
                    # We transpose to (H, W) if it came in as (W, H).
                    mask_slice = np.rot90(mask_slice, k=1)  # Common adjustment
                    mask_slice = np.fliplr(mask_slice)  # Common adjustment

                    inst_num = sorted_instances[
                        z
                    ]  # Assuming Z index aligns with sorted instances

                    save_name = f"{study_uid}_{inst_num}.npy"
                    save_path = os.path.join(output_dir, save_name)
                    np.save(save_path, mask_slice.astype(np.uint8))

                    slice_records.append(
                        {
                            "StudyInstanceUID": study_uid,
                            "slice_number": inst_num,
                            "mask_path": save_path,
                            "image_path": instance_map[inst_num],
                        }
                    )

        except Exception as e:
            print(f"Error processing segmentation for {study_uid}: {e}")
            continue

    df = pd.DataFrame(slice_records)
    cache_data(cache_file, df, load_cached_data=False)
    return df


def prepare_slice_classification_metadata(load_cached_data=True):
    """
    Prepares a DataFrame of slices for Stage 2 training.
    Merges train_metadata with bounding boxes to create slice-level targets.
    """
    cache_file = "slice_classification_meta.parquet"

    if load_cached_data:
        df = cache_data(cache_file, load_cached_data=True)
        if df is not None:
            return df

    print("Building slice classification metadata...")

    # Load sources
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    bbox_df = pd.read_csv(Config.TRAIN_BOUNDING_BOXES_PATH)

    # Filter for studies that have bounding boxes (subset of training data)
    bbox_studies = set(bbox_df["StudyInstanceUID"].unique())
    # We can also use studies that are negative overall, but for the encoder
    # we want to learn fracture features, so we focus on the bbox subset + negatives within them.

    records = []

    for _, row in train_meta.iterrows():
        study_uid = row["StudyInstanceUID"]

        # If this study has no bounding boxes recorded at all, but is fractured,
        # we might not know WHICH slice is fractured.
        # If it is NOT fractured (patient_overall=0), all slices are negative.
        # If it IS fractured but not in bbox_df, we skip it for Stage 2 training
        # because we lack slice-level labels.
        if study_uid not in bbox_studies and row["patient_overall"] == 1:
            continue

        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])
        dcm_files = glob.glob(os.path.join(image_dir, "*.dcm"))

        # Get bboxes for this study
        study_bboxes = bbox_df[bbox_df["StudyInstanceUID"] == study_uid]

        for f in dcm_files:
            try:
                fname = os.path.basename(f)
                slice_num = int(os.path.splitext(fname)[0])

                # Check if this slice has a fracture
                # A slice might have multiple bboxes
                slice_box = study_bboxes[study_bboxes["slice_number"] == slice_num]

                is_fractured = 0
                box_coords = None  # [x, y, w, h]

                if len(slice_box) > 0:
                    is_fractured = 1
                    # Take the first box or merge? Taking first for simplicity
                    b = slice_box.iloc[0]
                    box_coords = [b["x"], b["y"], b["width"], b["height"]]

                # For negative slices in positive patients or negative patients
                # we keep them as negatives.

                records.append(
                    {
                        "StudyInstanceUID": study_uid,
                        "slice_number": slice_num,
                        "image_path": f,
                        "fractured": is_fractured,
                        "box": box_coords,
                    }
                )

            except:
                continue

    df = pd.DataFrame(records)

    # Balancing: The dataset might be huge. We might want to downsample negatives.
    # For this implementation, we return all, let the DataLoader sampler handle balancing if needed.

    cache_data(cache_file, df, load_cached_data=False)
    return df


# =============================================================================
# Datasets
# =============================================================================


class SegmentationDataset(Dataset):
    """
    Stage 1 Dataset: Returns (Image, Mask) pairs.
    """

    def __init__(self, metadata_df, transforms=None, mode="train"):
        self.df = metadata_df
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load Image
        img = load_dicom(row["image_path"], resize_to=Config.IMAGE_SIZE_LOCAL)
        # Add channel dim: (1, H, W)
        img = np.expand_dims(img, axis=0)

        # Load Mask
        mask = np.load(row["mask_path"])
        # Resize mask to match image
        mask = cv2.resize(
            mask,
            (Config.IMAGE_SIZE_LOCAL, Config.IMAGE_SIZE_LOCAL),
            interpolation=cv2.INTER_NEAREST,
        )

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        mask_tensor = torch.from_numpy(mask).long()

        return img_tensor, mask_tensor


class DualStreamSliceDataset(Dataset):
    """
    Stage 2 Dataset: Returns (Local_Crop, Global_Resize, Label).
    """

    def __init__(self, metadata_df, transforms=None, mode="train"):
        self.df = metadata_df
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load Full Image (Original Size)
        # Note: load_dicom resizes if requested, we want original here to crop
        img_orig = load_dicom(row["image_path"], resize_to=Config.IMAGE_SIZE_ORIGINAL)

        # --- Global Stream ---
        # Resize to 256x256
        img_global = cv2.resize(
            img_orig,
            (Config.IMAGE_SIZE_GLOBAL, Config.IMAGE_SIZE_GLOBAL),
            interpolation=cv2.INTER_LINEAR,
        )
        img_global = np.expand_dims(img_global, axis=0)

        # --- Local Stream ---
        # Determine ROI Center
        # If box exists, use it. Else use center.
        if row["box"] is not None and str(row["box"]) != "nan":
            # box is [x, y, w, h] or string rep
            if isinstance(row["box"], str):
                box = eval(row["box"])
            else:
                box = row["box"]
            cx = box[0] + box[2] / 2
            cy = box[1] + box[3] / 2
        else:
            cx = Config.IMAGE_SIZE_ORIGINAL / 2
            cy = Config.IMAGE_SIZE_ORIGINAL / 2

        # Crop
        crop_size = Config.IMAGE_SIZE_LOCAL
        half_size = crop_size // 2

        start_x = int(
            max(0, min(cx - half_size, Config.IMAGE_SIZE_ORIGINAL - crop_size))
        )
        start_y = int(
            max(0, min(cy - half_size, Config.IMAGE_SIZE_ORIGINAL - crop_size))
        )

        img_local = img_orig[
            start_y : start_y + crop_size, start_x : start_x + crop_size
        ]

        # Pad if crop is smaller (edge cases)
        if img_local.shape[0] != crop_size or img_local.shape[1] != crop_size:
            pad_h = crop_size - img_local.shape[0]
            pad_w = crop_size - img_local.shape[1]
            img_local = np.pad(img_local, ((0, pad_h), (0, pad_w)), mode="constant")

        img_local = np.expand_dims(img_local, axis=0)

        # Targets
        label = float(row["fractured"]) if "fractured" in row else 0.0

        return {
            "local": torch.from_numpy(img_local).float(),
            "global": torch.from_numpy(img_global).float(),
            "label": torch.tensor(label).float(),
        }


class FeatureSequenceDataset(Dataset):
    """
    Stage 3 Dataset: Returns (Sequence_Features, Patient_Targets).
    Assumes features are pre-computed and stored in Config.CACHE_DIR/features/
    """

    def __init__(self, study_ids, targets_df=None, mode="train"):
        self.study_ids = study_ids
        self.targets_df = (
            targets_df  # Should contain columns for C1-C7, patient_overall
        )
        self.mode = mode
        self.feature_dir = os.path.join(Config.CACHE_DIR, "features")

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_uid = self.study_ids[idx]

        # Load features
        feat_path = os.path.join(self.feature_dir, f"{study_uid}.npy")
        if os.path.exists(feat_path):
            features = np.load(feat_path)
        else:
            # Fallback for missing features (e.g. during dev)
            # Create dummy features
            features = np.zeros((100, 512), dtype=np.float32)

        # Load Targets
        if self.mode != "test" and self.targets_df is not None:
            row = self.targets_df[
                self.targets_df["StudyInstanceUID"] == study_uid
            ].iloc[0]
            # Targets: C1-C7, patient_overall
            target_cols = Config.TARGET_COLS  # ["C1", ... "patient_overall"]
            labels = row[target_cols].values.astype(np.float32)
            return torch.from_numpy(features).float(), torch.from_numpy(labels).float()
        else:
            return torch.from_numpy(features).float(), torch.tensor([])


# =============================================================================
# Data Loaders
# =============================================================================


def get_segmentation_dataloader(batch_size=Config.STAGE1_BATCH_SIZE, split="train"):
    df = prepare_segmentation_cache()
    if df.empty:
        return None

    # Simple split if needed, but usually we use the provided metadata split
    # Here we just return all cached slices for the training phase
    ds = SegmentationDataset(df, mode="train")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )


def get_slice_classification_dataloader(
    batch_size=Config.STAGE2_BATCH_SIZE, split="train"
):
    df = prepare_slice_classification_metadata()
    if df.empty:
        return None

    # For validation, we might want to filter by validation studies
    if split == "val":
        val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
        val_uids = set(val_meta["StudyInstanceUID"])
        df = df[df["StudyInstanceUID"].isin(val_uids)]
    else:
        train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
        train_uids = set(train_meta["StudyInstanceUID"])
        df = df[df["StudyInstanceUID"].isin(train_uids)]

        # Balance dataset for training: Downsample negatives
        pos_df = df[df["fractured"] == 1]
        neg_df = df[df["fractured"] == 0]

        if len(pos_df) > 0:
            # Keep all positives, sample equal negatives
            neg_df = neg_df.sample(
                n=min(len(neg_df), len(pos_df) * 3), random_state=Config.SEED
            )
            df = (
                pd.concat([pos_df, neg_df])
                .sample(frac=1, random_state=Config.SEED)
                .reset_index(drop=True)
            )

    ds = DualStreamSliceDataset(df, mode=split)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )


def get_sequence_dataloader(batch_size=Config.STAGE3_BATCH_SIZE, split="train"):
    if split == "test":
        meta_df = pd.read_csv(Config.TEST_METADATA_PATH)
        study_ids = meta_df["StudyInstanceUID"].tolist()
        ds = FeatureSequenceDataset(study_ids, mode="test")
    else:
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        else:
            meta_path = Config.VAL_METADATA_PATH

        meta_df = pd.read_csv(meta_path)
        study_ids = meta_df["StudyInstanceUID"].tolist()
        ds = FeatureSequenceDataset(study_ids, targets_df=meta_df, mode=split)

    # Collate function to handle variable sequence lengths
    def collate_fn(batch):
        features, targets = zip(*batch)
        # Pad features to max length in batch
        # features is tuple of (Seq, Dim) tensors
        lengths = [f.shape[0] for f in features]
        max_len = max(lengths)
        dim = features[0].shape[1]

        padded_feats = torch.zeros(len(features), max_len, dim)
        for i, f in enumerate(features):
            end = f.shape[0]
            padded_feats[i, :end, :] = f

        if len(targets[0].shape) > 0:
            targets = torch.stack(targets)
        else:
            targets = None

        return padded_feats, targets, torch.tensor(lengths)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
