import os
import glob
import numpy as np
import pandas as pd
import cv2
import torch
import pydicom
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("data_module")


def load_dicom_array(path, size, window_center=400, window_width=1800):
    """
    Loads a DICOM file, applies bone windowing, resizes, and normalizes.
    Returns a 2D numpy array in range [0, 1].
    """
    try:
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype(np.float32)

        # Apply RescaleSlope and RescaleIntercept if present
        slope = getattr(dicom, "RescaleSlope", 1.0)
        intercept = getattr(dicom, "RescaleIntercept", 0.0)
        img = img * slope + intercept

        # Apply Windowing (Bone Window)
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)

        # Normalize to [0, 1]
        img = (img - img_min) / (img_max - img_min)

        # Resize
        if img.shape != size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

        return img

    except Exception as e:
        # Return black image on failure
        return np.zeros(size, dtype=np.float32)


def generate_anatomical_map(config, load_cached_data=True):
    """
    Generates or loads a map of (StudyInstanceUID, SliceNumber) -> Anatomical Label (0-7).
    0: Background
    1-7: C1-C7
    -1: Ignore (Unknown)
    """
    cache_path = os.path.join(config.WORKING_DIR, "anatomical_map.parquet")

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading anatomical map from {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info("Generating anatomical map from scratch...")

    records = []

    # 1. Process Bounding Boxes (Sparse Labels)
    if os.path.exists(config.TRAIN_BOUNDING_BOXES):
        bbox_df = pd.read_csv(config.TRAIN_BOUNDING_BOXES)
        # Map class strings 'C1' -> 1
        class_map = {f"C{i}": i for i in range(1, 8)}
        bbox_df["label"] = bbox_df["class"].map(class_map)

        for _, row in bbox_df.iterrows():
            records.append(
                {
                    "StudyInstanceUID": row["StudyInstanceUID"],
                    "slice_number": int(row["slice_number"]),
                    "aux_label": int(row["label"]),
                    "source": "bbox",
                }
            )

    # 2. Process Segmentations (Dense Labels)
    # SKIPPED: nibabel is not available in the environment.
    # We will rely solely on bounding boxes for anatomical supervision.
    if os.path.exists(config.SEGMENTATION_DIR):
        logger.warning(
            "Segmentation directory exists but nibabel is missing. Skipping NIfTI processing."
        )

    # Convert to DataFrame
    if not records:
        map_df = pd.DataFrame(columns=["StudyInstanceUID", "slice_number", "aux_label"])
    else:
        map_df = pd.DataFrame(records)

    # Deduplicate: Prefer Segmentation over BBox if conflict
    if not map_df.empty:
        source_priority = {"seg": 2, "bbox": 1}
        map_df["priority"] = map_df["source"].map(source_priority)
        map_df = map_df.sort_values("priority", ascending=True)
        map_df = map_df.drop_duplicates(
            subset=["StudyInstanceUID", "slice_number"], keep="last"
        )
        map_df = map_df[["StudyInstanceUID", "slice_number", "aux_label"]]

    # Save
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    map_df.to_parquet(cache_path, index=False)
    logger.info(f"Anatomical map saved with {len(map_df)} records.")

    return map_df


class CervicalSpineDataset(Dataset):
    def __init__(self, df, aux_map, config, mode="train"):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.mode = mode

        # Create fast lookup for auxiliary labels
        self.aux_lookup = {}
        if not aux_map.empty:
            grouped = aux_map.groupby("StudyInstanceUID")
            for uid, group in grouped:
                self.aux_lookup[uid] = dict(zip(group.slice_number, group.aux_label))

        self.use_aug = config.USE_AUGMENTATION and (mode == "train")

    def __len__(self):
        if self.config.DEBUG:
            return min(len(self.df), self.config.DEBUG_SAMPLE_SIZE)
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        uid = row["StudyInstanceUID"]

        # Construct full path to image directory
        study_dir = os.path.join(self.config.INPUT_DIR, row["image_path"])

        # List and sort DICOM files
        # Sorting by integer filename ensures correct Z-order
        try:
            files = sorted(
                glob.glob(os.path.join(study_dir, "*.dcm")),
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
            )
        except ValueError:
            files = sorted(glob.glob(os.path.join(study_dir, "*.dcm")))

        num_slices = len(files)
        seq_len = self.config.SEQ_LENGTH

        # Determine indices to sample
        if num_slices == 0:
            indices = np.zeros(seq_len, dtype=int)
            files = ["dummy"]
        elif num_slices < seq_len:
            # Pad if too few slices
            indices = np.linspace(0, num_slices - 1, seq_len).round().astype(int)
        else:
            # Uniform sampling
            indices = np.linspace(0, num_slices - 1, seq_len).round().astype(int)

        # Initialize containers
        images = np.zeros(
            (seq_len, 3, self.config.IMAGE_SIZE[0], self.config.IMAGE_SIZE[1]),
            dtype=np.float32,
        )
        aux_labels = np.full(seq_len, self.config.AUX_LOSS_IGNORE_INDEX, dtype=np.int64)

        # Calculate Augmentation Matrix (applied consistently to all slices)
        M = None
        if self.use_aug:
            angle = np.random.uniform(
                -self.config.AUG_ROTATION, self.config.AUG_ROTATION
            )
            scale = np.random.uniform(
                1 - self.config.AUG_SCALE, 1 + self.config.AUG_SCALE
            )
            shift_x = (
                np.random.uniform(-self.config.AUG_SHIFT, self.config.AUG_SHIFT)
                * self.config.IMAGE_SIZE[1]
            )
            shift_y = (
                np.random.uniform(-self.config.AUG_SHIFT, self.config.AUG_SHIFT)
                * self.config.IMAGE_SIZE[0]
            )

            center = (self.config.IMAGE_SIZE[1] // 2, self.config.IMAGE_SIZE[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += shift_x
            M[1, 2] += shift_y

        # Load and process slices
        study_aux_map = self.aux_lookup.get(uid, {})

        for i, idx in enumerate(indices):
            # 2.5D Stacking: (idx-1, idx, idx+1)
            triplet_indices = [max(0, idx - 1), idx, min(num_slices - 1, idx + 1)]

            slice_img = np.zeros(
                (self.config.IMAGE_SIZE[0], self.config.IMAGE_SIZE[1], 3),
                dtype=np.float32,
            )

            if num_slices > 0:
                # Load channels
                for c, t_idx in enumerate(triplet_indices):
                    img = load_dicom_array(files[t_idx], self.config.IMAGE_SIZE)
                    slice_img[:, :, c] = img

                # Get Auxiliary Label for the center slice
                # Parse slice number from filename
                fname = os.path.basename(files[idx])
                try:
                    slice_num = int(os.path.splitext(fname)[0])
                    if slice_num in study_aux_map:
                        aux_labels[i] = study_aux_map[slice_num]
                except:
                    pass

            # Apply Augmentation
            if M is not None:
                slice_img = cv2.warpAffine(
                    slice_img,
                    M,
                    (self.config.IMAGE_SIZE[1], self.config.IMAGE_SIZE[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )

            # Transpose to (Channels, H, W)
            images[i] = slice_img.transpose(2, 0, 1)

        # Prepare Fracture Targets
        fracture_labels = np.zeros(self.config.NUM_CLASSES, dtype=np.float32)
        if self.mode != "test":
            cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
            fracture_labels = row[cols].values.astype(np.float32)

        return {
            "images": torch.tensor(images, dtype=torch.float32),
            "fracture_labels": torch.tensor(fracture_labels, dtype=torch.float32),
            "aux_labels": torch.tensor(aux_labels, dtype=torch.long),
            "study_id": str(uid),
        }


def get_dataloaders(config):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Load Metadata
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # Generate/Load Auxiliary Map
    aux_map = generate_anatomical_map(config, load_cached_data=True)

    # Create Datasets
    train_dataset = CervicalSpineDataset(train_df, aux_map, config, mode="train")
    val_dataset = CervicalSpineDataset(val_df, aux_map, config, mode="val")
    test_dataset = CervicalSpineDataset(test_df, aux_map, config, mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
