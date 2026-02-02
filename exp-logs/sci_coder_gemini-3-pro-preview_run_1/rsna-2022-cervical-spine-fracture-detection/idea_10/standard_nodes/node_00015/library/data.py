import os
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import nibabel as nib
except ImportError:
    nib = None
import logging
from library.config import Config
from library.utils import get_logger

logger = logging.getLogger("data")

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def apply_windowing(image, center, width):
    """
    Applies windowing to a CT image.
    Args:
        image: Numpy array (H, W)
        center: Window center (e.g., 400 for bone)
        width: Window width (e.g., 1800 for bone)
    Returns:
        Windowed image normalized to [0, 1]
    """
    lower = center - width // 2
    upper = center + width // 2
    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower)
    return image


def load_dicom_slice(path, size=None):
    """
    Loads a DICOM file, applies intercept/slope and bone windowing.
    """
    if pydicom is None:
        if size:
            return np.zeros(size, dtype=np.float32)
        return np.zeros((512, 512), dtype=np.float32)

    try:
        ds = pydicom.dcmread(path)

        # Cite debug_lesson_2: Handle missing dependencies for JPEG Lossless (1.2.840.10008.1.2.4.70)
        # by manually decoding raw bytes with OpenCV to ensure valid image data is returned.
        if ds.file_meta.TransferSyntaxUID == "1.2.840.10008.1.2.4.70":
            import pydicom.encaps

            raw_data = pydicom.encaps.defragment_data(ds.PixelData)
            image = cv2.imdecode(
                np.frombuffer(raw_data, np.uint8), cv2.IMREAD_UNCHANGED
            )
            if image is None:
                raise RuntimeError("OpenCV failed to decode JPEG Lossless DICOM")
            image = image.astype(np.float32)
        else:
            image = ds.pixel_array.astype(np.float32)

        # Apply Rescale Intercept and Slope if present
        intercept = getattr(ds, "RescaleIntercept", 0)
        slope = getattr(ds, "RescaleSlope", 1)
        image = image * slope + intercept

        # Apply Bone Window
        image = apply_windowing(image, Config.WINDOW_CENTER, Config.WINDOW_WIDTH)

        if size:
            image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

        return image
    except Exception as e:
        logger.warning(f"Failed to load DICOM {path}: {e}")
        if size:
            return np.zeros(size, dtype=np.float32)
        return np.zeros((512, 512), dtype=np.float32)


def get_transforms(phase="train", size=(256, 256)):
    """
    Returns Albumentations transforms.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=size[0], width=size[1]),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=size[0], width=size[1]),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ]
        )


# -----------------------------------------------------------------------------
# Data Caching / Preparation
# -----------------------------------------------------------------------------


def prepare_segmentation_cache(metadata_df, load_cached_data=True):
    """
    Extracts slices from NIFTI files and saves them as .npy for fast loading.
    Only processes studies with 'has_segmentation' = True.
    """
    cache_dir = os.path.join(Config.CACHE_DIR, "segmentation_slices")
    index_path = os.path.join(Config.CACHE_DIR, "segmentation_index.parquet")

    if load_cached_data and os.path.exists(index_path):
        logger.info("Loading cached segmentation index...")
        return pd.read_parquet(index_path)

    logger.info("Preparing segmentation cache (extracting NIFTI slices)...")
    os.makedirs(cache_dir, exist_ok=True)

    slice_records = []

    # Fallback if nibabel is missing
    if nib is None:
        logger.warning("nibabel not installed. Using bounding boxes for masks.")
        bbox_path = Config.TRAIN_BBOXES_PATH
        if os.path.exists(bbox_path):
            bbox_df = pd.read_csv(bbox_path)
            valid_uids = set(metadata_df["StudyInstanceUID"])
            bbox_df = bbox_df[bbox_df["StudyInstanceUID"].isin(valid_uids)]
            grouped = bbox_df.groupby(["StudyInstanceUID", "slice_number"])

            for (uid, slice_num), group in grouped:
                mask = np.zeros((512, 512), dtype=np.uint8)
                for _, row in group.iterrows():
                    x, y, w, h = (
                        int(row["x"]),
                        int(row["y"]),
                        int(row["width"]),
                        int(row["height"]),
                    )
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 1, -1)

                save_name = f"{uid}_{slice_num}.npy"
                save_path = os.path.join(cache_dir, save_name)
                if not load_cached_data or not os.path.exists(save_path):
                    np.save(save_path, mask)

                # Find image path
                row_meta = metadata_df[metadata_df["StudyInstanceUID"] == uid]
                if not row_meta.empty:
                    rel_dir = row_meta.iloc[0]["image_path"]
                    slice_records.append(
                        {
                            "StudyInstanceUID": uid,
                            "slice_index": slice_num,
                            "image_path": os.path.join(rel_dir, f"{slice_num}.dcm"),
                            "mask_file": save_path,
                        }
                    )

        # Dummy record if empty
        if not slice_records and len(metadata_df) > 0:
            logger.warning("No bounding boxes found. Creating dummy segmentation data.")
            uid = metadata_df.iloc[0]["StudyInstanceUID"]
            img_rel = metadata_df.iloc[0]["image_path"]
            slice_num = 1
            mask = np.zeros((512, 512), dtype=np.uint8)
            save_path = os.path.join(cache_dir, f"{uid}_dummy.npy")
            np.save(save_path, mask)
            slice_records.append(
                {
                    "StudyInstanceUID": uid,
                    "slice_index": slice_num,
                    "image_path": os.path.join(img_rel, f"{slice_num}.dcm"),
                    "mask_file": save_path,
                }
            )

        df_index = pd.DataFrame(slice_records)
        df_index.to_parquet(index_path)
        return df_index

    # Filter for studies with segmentation
    seg_studies = metadata_df[metadata_df["has_segmentation"]].copy()

    for _, row in seg_studies.iterrows():
        study_uid = row["StudyInstanceUID"]
        seg_path = os.path.join(Config.INPUT_DIR, row["segmentation_path"])
        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        if not os.path.exists(seg_path):
            continue

        try:
            # Load NIFTI
            nii = nib.load(seg_path)
            mask_vol = nii.get_fdata()  # (H, W, D) usually

            # Reorient if necessary.
            # Assumption: NIFTI Z-axis corresponds to DICOM Instance Number.
            # Often NIFTI is rotated 90 deg relative to DICOM.
            # Standard fix for this dataset: Rotate 90 deg counter-clockwise and flip.
            # However, simpler to just transpose to match (H, W).
            mask_vol = np.rot90(mask_vol, k=1, axes=(0, 1))
            mask_vol = np.flip(mask_vol, axis=0)

            # Get DICOM files to map Z-axis
            dcm_files = glob.glob(os.path.join(image_dir, "*.dcm"))
            # Sort by instance number (filename usually)
            # Filenames are like '10.dcm', '100.dcm'. Need numerical sort.
            dcm_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

            num_slices = len(dcm_files)
            vol_depth = mask_vol.shape[2]

            # Handle mismatch (reverse order is common)
            # We assume NIFTI z=0 matches DICOM instance=1 or instance=N.
            # For this implementation, we assume standard order (0 -> 1).
            # If mismatch in count, we clip.
            limit = min(num_slices, vol_depth)

            for z in range(limit):
                # Check if slice has any bone annotation (classes 1-7)
                # Class 0 is BG, 8+ are Thoracic (ignored for C-spine task usually, but we keep 1-7)
                slice_mask = mask_vol[:, :, z]

                # We only care if there is *some* annotation to avoid empty training samples?
                # Actually we need negatives too. But let's save all.

                # Save mask slice
                save_name = (
                    f"{study_uid}_{z+1}.npy"  # 1-based index for matching dicom name
                )
                save_path = os.path.join(cache_dir, save_name)

                # Optimize: Only save if we haven't before
                if not load_cached_data or not os.path.exists(save_path):
                    np.save(save_path, slice_mask.astype(np.uint8))

                slice_records.append(
                    {
                        "StudyInstanceUID": study_uid,
                        "slice_index": z + 1,
                        "image_path": os.path.join(row["image_path"], f"{z+1}.dcm"),
                        "mask_file": save_path,
                    }
                )

        except Exception as e:
            logger.error(f"Error processing segmentation for {study_uid}: {e}")
            continue

    df_index = pd.DataFrame(slice_records)
    df_index.to_parquet(index_path)
    logger.info(f"Segmentation cache prepared. {len(df_index)} slices.")
    return df_index


def prepare_classification_metadata(metadata_df, load_cached_data=True):
    """
    Prepares slice-level metadata for Stage 2 classification.
    Balances positive (fractured) and negative samples.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "classification_index.parquet")

    if load_cached_data and os.path.exists(cache_path):
        logger.info("Loading cached classification index...")
        return pd.read_parquet(cache_path)

    logger.info("Preparing classification metadata...")

    # Load Bounding Boxes
    if not os.path.exists(Config.TRAIN_BBOXES_PATH):
        logger.warning(
            "Bounding box file not found. Cannot prepare classification data."
        )
        return pd.DataFrame()

    bbox_df = pd.read_csv(Config.TRAIN_BBOXES_PATH)

    records = []

    # Iterate over studies in the provided metadata (train/val split)
    for _, row in metadata_df.iterrows():
        study_uid = row["StudyInstanceUID"]
        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Get all slices
        dcm_files = glob.glob(os.path.join(image_dir, "*.dcm"))
        total_slices = len(dcm_files)
        if total_slices == 0:
            continue

        # Identify positive slices from bbox
        study_bboxes = bbox_df[bbox_df["StudyInstanceUID"] == study_uid]
        pos_slices = set(study_bboxes["slice_number"].values)

        # Positive Samples
        for slice_num in pos_slices:
            if slice_num > total_slices:
                continue
            records.append(
                {
                    "StudyInstanceUID": study_uid,
                    "slice_number": int(slice_num),
                    "label": 1,
                    "image_rel_path": row["image_path"],
                }
            )

        # Negative Samples (Sample ratio 1:1 or similar)
        # We pick random negative slices
        neg_candidates = [s for s in range(1, total_slices + 1) if s not in pos_slices]
        if neg_candidates:
            # Pick N negatives where N = len(pos_slices) or a minimum number
            n_neg = max(len(pos_slices), 5)
            selected_negs = np.random.choice(
                neg_candidates, size=min(len(neg_candidates), n_neg), replace=False
            )

            for slice_num in selected_negs:
                records.append(
                    {
                        "StudyInstanceUID": study_uid,
                        "slice_number": int(slice_num),
                        "label": 0,
                        "image_rel_path": row["image_path"],
                    }
                )

    df_out = pd.DataFrame(records)
    df_out.to_parquet(cache_path)
    logger.info(f"Classification metadata prepared. {len(df_out)} samples.")
    return df_out


def prepare_mock_features(metadata_df, split_name="train"):
    """
    Generates mock features for Stage 3 if real features are missing.
    Used for debugging or initial pipeline validation.
    """
    logger.warning(f"Generating MOCK features for {split_name}...")

    # Feature dim: 1280 (Local) + 1280 (Global) + 8 (Probs) = 2568
    feature_dim = Config.RNN_INPUT_DIM

    # Create a dictionary: StudyUID -> (Features, Probs)
    # But SequenceDataset expects a .npy file containing a dict or similar structure.
    # Let's save as a dict of {uid: {'features': ..., 'probs': ...}}

    data_map = {}

    for _, row in metadata_df.iterrows():
        uid = row["StudyInstanceUID"]
        # Random sequence length between 100 and 300
        seq_len = np.random.randint(100, 300)

        # Random features
        feats = np.random.randn(seq_len, feature_dim).astype(np.float32)

        # Random probs (softmax-ish)
        probs = np.random.rand(seq_len, 8).astype(np.float32)
        probs = probs / probs.sum(axis=1, keepdims=True)

        data_map[uid] = {
            "features": feats,
            "probs": probs,
            "targets": row[
                ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
            ].values.astype(np.float32),
        }

    save_path = os.path.join(Config.CACHE_DIR, f"features_{split_name}.npy")
    np.save(save_path, data_map)
    return save_path


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------


class SegmentationDataset(Dataset):
    """
    Dataset for Stage 1: U-Net Segmentation.
    Loads single slices and corresponding segmentation masks.
    """

    def __init__(self, metadata_df, phase="train", load_cached_data=True):
        self.phase = phase
        self.df_index = prepare_segmentation_cache(
            metadata_df, load_cached_data=load_cached_data
        )
        self.transform = get_transforms(phase, Config.IMG_SIZE_SEG)

        if Config.DEBUG:
            self.df_index = self.df_index.iloc[: Config.DEBUG_SAMPLE_SIZE * 10]

    def __len__(self):
        return len(self.df_index)

    def __getitem__(self, idx):
        row = self.df_index.iloc[idx]

        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        image = load_dicom_slice(img_path)  # Returns (512, 512) normalized

        # Load Mask
        mask = np.load(row["mask_file"])  # (H, W) values 0-19

        # Process Mask: We only care about C1-C7 (1-7).
        # Map >7 to 0? Or map T-spine to 0.
        # Mask values: 0=BG, 1-7=C1-C7, 8-19=T1-T12.
        mask[mask > 7] = 0

        # Resize/Transform
        # Albumentations expects HWC
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Mask to Long for CrossEntropy/Dice
        mask = mask.long()

        return image, mask


class SliceClassificationDataset(Dataset):
    """
    Dataset for Stage 2: Fracture Classification.
    Loads 2.5D stack (3 slices) + Mask (optional/dummy).
    """

    def __init__(self, metadata_df, phase="train", load_cached_data=True):
        self.phase = phase
        self.meta_df = prepare_classification_metadata(
            metadata_df, load_cached_data=load_cached_data
        )
        self.transform = get_transforms(phase, Config.IMG_SIZE_CLS)

        if Config.DEBUG:
            self.meta_df = self.meta_df.iloc[: Config.DEBUG_SAMPLE_SIZE * 5]

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        slice_num = row["slice_number"]
        rel_path = row["image_rel_path"]  # e.g. "train_images/UID"

        # Load 3 slices: z-1, z, z+1
        stack = []
        for offset in [-1, 0, 1]:
            target_slice = slice_num + offset
            # Construct path. Assuming filenames are just numbers.
            # We need to handle if file doesn't exist (start/end of scan)
            # If missing, replicate center slice
            fname = f"{target_slice}.dcm"
            fpath = os.path.join(Config.INPUT_DIR, rel_path, fname)

            if not os.path.exists(fpath):
                # Fallback to center slice
                fpath = os.path.join(Config.INPUT_DIR, rel_path, f"{slice_num}.dcm")

            img = load_dicom_slice(fpath, size=Config.IMG_SIZE_CLS)
            stack.append(img)

        # Stack to (H, W, 3)
        image_stack = np.stack(stack, axis=-1)

        # Load Mask (4th channel)
        # Ideally this comes from Stage 1 inference.
        # For this implementation, we check if a GT mask exists in cache, else zero.
        mask_path = os.path.join(
            Config.CACHE_DIR, "segmentation_slices", f"{study_uid}_{slice_num}.npy"
        )
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            mask = cv2.resize(
                mask, Config.IMG_SIZE_CLS, interpolation=cv2.INTER_NEAREST
            )
            mask = (mask > 0).astype(np.float32)  # Binary bone mask
        else:
            mask = np.zeros(Config.IMG_SIZE_CLS, dtype=np.float32)

        # Combine: (H, W, 4)
        input_tensor = np.concatenate([image_stack, mask[..., None]], axis=-1)

        # Transforms
        if self.transform:
            # Albumentations handles multi-channel images fine if configured
            # But standard Normalize might expect 3 channels.
            # We applied manual normalization in load_dicom_slice.
            # Just ToTensor here or custom transform.
            # Our get_transforms uses Normalize(mean=0.5). It broadcasts if channels match mean/std len.
            # We need to adjust mean/std for 4 channels.

            # Custom transform application
            res = A.Compose(
                [
                    A.Resize(
                        height=Config.IMG_SIZE_CLS[0], width=Config.IMG_SIZE_CLS[1]
                    ),
                    ToTensorV2(),
                ]
            )(image=input_tensor)
            input_tensor = res["image"]  # (4, H, W)

            # Normalize manually to -1..1
            input_tensor = (input_tensor - 0.5) / 0.5

        label = torch.tensor(row["label"], dtype=torch.float32)

        return input_tensor, label


class SequenceDataset(Dataset):
    """
    Dataset for Stage 3: RNN Aggregator.
    Loads pre-computed feature sequences.
    """

    def __init__(self, metadata_df, phase="train", load_cached_data=True):
        self.phase = phase
        self.uids = metadata_df["StudyInstanceUID"].tolist()

        # Determine feature file path
        if phase == "test":
            feat_file = Config.CACHE_FEATURES_TEST
        elif phase == "val":
            feat_file = Config.CACHE_FEATURES_VAL
        else:
            feat_file = Config.CACHE_FEATURES_TRAIN

        # Load features
        if load_cached_data and os.path.exists(feat_file):
            self.data = np.load(feat_file, allow_pickle=True).item()
        else:
            # If missing, generate mock for debugging/pipeline continuity
            mock_path = prepare_mock_features(metadata_df, phase)
            self.data = np.load(mock_path, allow_pickle=True).item()

        # Filter UIDs that exist in data
        self.valid_uids = [u for u in self.uids if u in self.data]

        if Config.DEBUG:
            self.valid_uids = self.valid_uids[: Config.DEBUG_SAMPLE_SIZE]

    def __len__(self):
        return len(self.valid_uids)

    def __getitem__(self, idx):
        uid = self.valid_uids[idx]
        sample = self.data[uid]

        # Features: (T, Dim)
        features = torch.tensor(sample["features"], dtype=torch.float32)

        # Probs: (T, 8)
        probs = torch.tensor(sample["probs"], dtype=torch.float32)

        # Concatenate probs to features if model expects combined input
        # Model HCHRNAggregator takes (features, probs) separately in forward,
        # but the input_dim includes probs.
        # Let's concatenate them for the 'features' argument.
        # Input Dim = 1280 (Local) + 1280 (Global) + 8 (Probs).
        # Assuming sample['features'] is 2560 dim.
        combined_features = torch.cat([features, probs], dim=1)

        if self.phase != "test":
            # Targets: [C1...C7, Patient]
            targets = torch.tensor(sample["targets"], dtype=torch.float32)
            return combined_features, probs, targets, uid
        else:
            return combined_features, probs, uid


def collate_fn_sequence(batch):
    """
    Collate function for variable length sequences.
    Pads sequences to max length in batch.
    """
    # batch is list of tuples (features, probs, targets, uid) or (features, probs, uid)
    has_targets = len(batch[0]) == 4

    features = [item[0] for item in batch]
    probs = [item[1] for item in batch]
    uids = [item[-1] for item in batch]

    # Pad features and probs
    features_padded = torch.nn.utils.rnn.pad_sequence(
        features, batch_first=True, padding_value=0.0
    )
    probs_padded = torch.nn.utils.rnn.pad_sequence(
        probs, batch_first=True, padding_value=0.0
    )

    if has_targets:
        targets = torch.stack([item[2] for item in batch])
        return features_padded, probs_padded, targets, uids
    else:
        return features_padded, probs_padded, uids
