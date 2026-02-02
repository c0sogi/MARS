import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import configuration
from library.config import Config

# ==========================================
# Image Loading Utility
# ==========================================


def load_dicom_image(path, target_size=None):
    """
    Reads a DICOM file by scanning for image headers (JPEG/JPEG2000)
    and decoding the byte stream. This bypasses the need for pydicom
    and handles compressed formats robustly.
    """
    if not os.path.exists(path):
        # Return a blank image if file missing
        if target_size:
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        return np.zeros((640, 640, 3), dtype=np.uint8)

    try:
        with open(path, "rb") as f:
            b = f.read()

        # JPEG Magic Number: FF D8
        # JPEG2000 Magic Number: 00 00 00 0C 6A 50 20 20 0D 0A 87 0A
        # We search for common headers.

        # Try finding JPEG header
        idx = b.find(b"\xff\xd8")
        if idx != -1:
            img_bytes = b[idx:]
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is not None:
                return _process_cv2_image(img, target_size)

        # Try finding JPEG2000 header (codestream) - FF 4F FF 51
        # Or JP2 file format signature
        idx_jp2 = b.find(b"\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a")
        if idx_jp2 != -1:
            img_bytes = b[idx_jp2:]
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is not None:
                return _process_cv2_image(img, target_size)

        # Fallback: Try decoding the whole buffer (sometimes works for uncompressed or simple wrappers)
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is not None:
            return _process_cv2_image(img, target_size)

    except Exception:
        pass

    # Final fallback: Blank image
    if target_size:
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    return np.zeros((640, 640, 3), dtype=np.uint8)


def _process_cv2_image(img, target_size):
    """Helper to handle grayscale conversion, resizing, and channel replication."""
    if img is None:
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

    # Handle 16-bit images (rescale to 8-bit)
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        # Normalize float or other types
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize
    if target_size is not None:
        img = cv2.resize(
            img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
        )

    return img


# ==========================================
# Augmentation Pipelines
# ==========================================


def get_transforms(phase="train", img_size=(640, 640)):
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


# ==========================================
# Dataset Class
# ==========================================


class BreastCancerBagDataset(Dataset):
    """
    Multi-Instance Learning Dataset.
    Groups images by Breast (Patient + Laterality) into 'bags'.
    """

    def __init__(self, csv_path, images_dir, phase="train", load_cached_data=True):
        self.phase = phase
        self.images_dir = images_dir
        self.transforms = get_transforms(phase, Config.IMG_SIZE)

        # Caching logic for the grouped dataframe
        cache_name = (
            f"grouped_{phase}_{os.path.basename(csv_path).replace('.csv', '')}.parquet"
        )
        cache_path = os.path.join(Config.OUTPUT_DIR, cache_name)

        if load_cached_data and os.path.exists(cache_path):
            self.data = pd.read_parquet(cache_path)
        else:
            df = pd.read_csv(csv_path)

            # Debugging subset
            if Config.DEBUG:
                df = df.sample(
                    n=min(len(df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
                ).reset_index(drop=True)

            # Preprocessing Metadata
            # Fill NaNs
            if "age" in df.columns:
                df["age"] = df["age"].fillna(df["age"].mean())
            else:
                df["age"] = 50.0  # Default

            if "implant" in df.columns:
                df["implant"] = df["implant"].fillna(0).astype(int)
            else:
                df["implant"] = 0

            if "machine_id" in df.columns:
                df["machine_id"] = df["machine_id"].fillna(df["machine_id"].mode()[0])
            else:
                df["machine_id"] = 0

            # Grouping
            # For Train/Val: Group by patient_id + laterality
            # For Test: Group by prediction_id
            if "prediction_id" in df.columns:
                group_cols = ["prediction_id"]
            else:
                # Create a surrogate ID for grouping if prediction_id doesn't exist (Train/Val)
                df["group_id"] = df["patient_id"].astype(str) + "_" + df["laterality"]
                group_cols = ["group_id"]

            # Aggregation dictionary
            agg_dict = {
                "file_path": list,
                "age": "first",
                "implant": "first",
                "machine_id": "first",
            }
            if "cancer" in df.columns:
                agg_dict["cancer"] = (
                    "max"  # If any view is cancer, the breast is cancer
                )

            # Perform Grouping
            self.data = df.groupby(group_cols).agg(agg_dict).reset_index()

            # Save to cache
            os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
            self.data.to_parquet(cache_path, index=False)

        # Pre-compute metadata normalization stats (simple min-max for age)
        self.age_min = 20.0
        self.age_max = 90.0

        # Machine ID mapping (simplified for this context)
        # In a real scenario, we'd fit an encoder. Here we hash or map common ones.
        self.machine_ids = sorted(self.data["machine_id"].unique())
        self.machine_id_map = {mid: i for i, mid in enumerate(self.machine_ids)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1. Load Images (Bag)
        file_paths = row["file_path"]
        # If it's a numpy array (from parquet), convert to list
        if isinstance(file_paths, np.ndarray):
            file_paths = file_paths.tolist()

        images = []
        for rel_path in file_paths:
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            img = load_dicom_image(full_path, Config.IMG_SIZE)

            # Apply transforms
            if self.transforms:
                res = self.transforms(image=img)
                img_tensor = res["image"]
            else:
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

            images.append(img_tensor)

        # Stack images: (V, C, H, W)
        if len(images) > 0:
            images = torch.stack(images)
        else:
            # Fallback for empty bag (should not happen)
            images = torch.zeros((1, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1]))

        # 2. Process Metadata
        age = (row["age"] - self.age_min) / (self.age_max - self.age_min)
        implant = float(row["implant"])
        machine_code = self.machine_id_map.get(row["machine_id"], 0)
        # Simple one-hot for machine_id (top 10 common ones usually)
        # We will just pass the index and let the model handle embedding or
        # just pass the raw normalized value if we want to keep it simple.
        # Let's pass a vector: [age, implant, machine_idx]
        meta_vec = torch.tensor(
            [age, implant, float(machine_code)], dtype=torch.float32
        )

        # 3. Label
        label = 0.0
        if "cancer" in row:
            label = float(row["cancer"])

        # Return
        # prediction_id is useful for submission generation
        pred_id = row["prediction_id"] if "prediction_id" in row else row["group_id"]

        return {
            "images": images,  # (V, C, H, W)
            "metadata": meta_vec,  # (3,)
            "label": torch.tensor([label], dtype=torch.float32),
            "id": pred_id,
        }


# ==========================================
# Collate Function
# ==========================================


def collate_bags(batch):
    """
    Collate function to handle variable number of views per breast.
    Pads bags to the maximum number of views in the batch.
    """
    # batch is a list of dicts

    # Find max views in this batch
    max_views = 0
    for item in batch:
        views = item["images"].shape[0]
        if views > max_views:
            max_views = views

    batch_size = len(batch)
    c, h, w = batch[0]["images"].shape[1:]

    # Initialize tensors
    padded_images = torch.zeros((batch_size, max_views, c, h, w))
    attention_mask = torch.zeros((batch_size, max_views))  # 1 for real, 0 for pad
    metadata = []
    labels = []
    ids = []

    for i, item in enumerate(batch):
        views = item["images"].shape[0]

        # Fill images
        padded_images[i, :views] = item["images"]

        # Fill mask
        attention_mask[i, :views] = 1.0

        metadata.append(item["metadata"])
        labels.append(item["label"])
        ids.append(item["id"])

    return {
        "images": padded_images,  # (B, Max_V, C, H, W)
        "mask": attention_mask,  # (B, Max_V)
        "metadata": torch.stack(metadata),  # (B, Meta_Dim)
        "labels": torch.stack(labels),  # (B, 1)
        "ids": ids,  # List of IDs
    }


# ==========================================
# DataLoader Factory
# ==========================================


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    """

    # Train Set
    train_ds = BreastCancerBagDataset(
        csv_path=Config.TRAIN_CSV,
        images_dir=Config.TRAIN_IMAGES_DIR,
        phase="train",
        load_cached_data=load_cached_data,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_bags,
        drop_last=True,
    )

    # Validation Set
    val_ds = BreastCancerBagDataset(
        csv_path=Config.VAL_CSV,
        images_dir=Config.TRAIN_IMAGES_DIR,
        phase="val",
        load_cached_data=load_cached_data,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_bags,
    )

    # Test Set
    test_ds = BreastCancerBagDataset(
        csv_path=Config.TEST_CSV,
        images_dir=Config.TEST_IMAGES_DIR,
        phase="test",
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_bags,
    )

    return train_loader, val_loader, test_loader
