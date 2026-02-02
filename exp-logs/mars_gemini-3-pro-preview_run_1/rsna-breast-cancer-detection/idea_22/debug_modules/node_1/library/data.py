import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


def compute_age_stats(df_train, cache_dir, load_cached=True):
    """
    Computes or loads cached age statistics (mean, std) for normalization.

    Args:
        df_train (pd.DataFrame): Training metadata.
        cache_dir (str): Directory to save/load cache.
        load_cached (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (mean_age, std_age)
    """
    cache_path = os.path.join(cache_dir, "age_stats.npy")

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    if load_cached and os.path.exists(cache_path):
        try:
            stats = np.load(cache_path)
            return stats[0], stats[1]
        except Exception:
            pass  # Fallback to recomputing if load fails

    # Compute stats
    # Filter out impossible ages if any, though analysis showed min 26 max 89
    valid_ages = df_train["age"].dropna()
    mean_age = valid_ages.mean()
    std_age = valid_ages.std()

    # Save to cache
    np.save(cache_path, np.array([mean_age, std_age]))

    return mean_age, std_age


class BreastCancerPairedDataset(Dataset):
    """
    Dataset that yields paired images (Target, Contralateral) for Siamese Network.
    Constructs 3-channel input: [Image, Age, Implant].
    """

    def __init__(self, df, age_stats, img_size, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            age_stats (tuple): (mean_age, std_age) for normalization.
            img_size (tuple): (height, width).
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.age_mean, self.age_std = age_stats
        self.img_size = img_size
        self.transforms = transforms
        self.mode = mode

        # Build Lookup for Contralateral Images
        # Key: (patient_id, view), Value: {laterality: file_path}
        self.pair_lookup = {}

        # Pre-process dataframe to build lookup efficiently
        # We only need patient_id, view, laterality, file_path for pairing
        subset = self.df[["patient_id", "view", "laterality", "file_path"]]
        for idx, row in subset.iterrows():
            pid = row["patient_id"]
            view = row["view"]
            lat = row["laterality"]
            path = row["file_path"]

            key = (pid, view)
            if key not in self.pair_lookup:
                self.pair_lookup[key] = {}
            self.pair_lookup[key][lat] = path

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        """
        Loads an image using OpenCV with robust fallbacks for DICOM/Raw formats.
        Normalizes to [0, 1].
        """
        full_path = os.path.join(Config.INPUT_DIR, path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        # 1. Try standard cv2 load
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        # 2. Fallback: Cascading Ingestion (Cite debug_lesson_8)
        if img is None:
            try:
                with open(full_path, "rb") as f:
                    content = f.read()

                # A. Try decoding embedded stream (JPEG/JP2)
                img = cv2.imdecode(
                    np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED
                )

                # B. Try Raw Binary Fallback (Cite debug_lesson_29)
                if img is None:
                    # Calculate expected bytes for 8-bit image
                    expected_pixels = self.img_size[0] * self.img_size[1]

                    # Heuristic: If file is large enough, read raw pixels from end
                    if len(content) >= expected_pixels:
                        pixel_data = content[-expected_pixels:]
                        img = np.frombuffer(pixel_data, dtype=np.uint8).reshape(
                            self.img_size
                        )
            except Exception:
                pass

        if img is None:
            raise ValueError(
                f"Failed to load image (corrupt or unsupported format): {full_path}"
            )

        # Handle multi-channel (convert to grayscale if needed)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize
        img = cv2.resize(
            img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR
        )

        # Normalize to [0, 1] (Min-Max per image)
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img.astype(np.float32) - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Identify Target and Contralateral Paths
        target_path = row["file_path"]
        pid = row["patient_id"]
        view = row["view"]
        lat = row["laterality"]

        # Find contralateral (same patient, same view, opposite laterality)
        contra_lat = "R" if lat == "L" else "L"
        contra_path = self.pair_lookup.get((pid, view), {}).get(contra_lat, None)

        # 2. Load Images
        # Target: Must exist
        img_target = self._load_image(target_path)

        # Contra: Might not exist
        if contra_path:
            try:
                img_contra = self._load_image(contra_path)
            except (FileNotFoundError, ValueError):
                # If specifically missing/corrupt, treat as no contralateral
                img_contra = np.zeros_like(img_target, dtype=np.float32)
        else:
            img_contra = np.zeros_like(img_target, dtype=np.float32)

        # 3. Apply Augmentations (Synchronized)
        if self.transforms:
            # Pass contra as additional target to ensure same geometric transform
            augmented = self.transforms(image=img_target, image_contra=img_contra)
            img_target = augmented["image"]
            img_contra = augmented["image_contra"]

        # 4. Construct 3-Channel Input [Image, Age, Implant]
        # Prepare Metadata Features
        age = row["age"] if not pd.isna(row["age"]) else self.age_mean
        age_norm = (age - self.age_mean) / (self.age_std + 1e-6)

        implant = 1.0 if row["implant"] == 1 else 0.0

        # Create Spatial Maps (H, W)
        h, w = img_target.shape
        age_map = np.full((h, w), age_norm, dtype=np.float32)
        implant_map = np.full((h, w), implant, dtype=np.float32)

        # Stack Channels -> (3, H, W)
        # Note: Albumentations ToTensorV2 usually handles HWC->CHW, but here we construct manually
        # If transforms output numpy (which they do before ToTensor), shape is (H, W)

        # Check if transforms included ToTensorV2 (which returns Tensor) or not
        if isinstance(img_target, torch.Tensor):
            # If ToTensorV2 was used, it adds a channel dim if input was HxW?
            # Actually ToTensorV2 on 2D array returns (1, H, W) or (H, W)?
            # Usually we handle ToTensor manually for custom channel stacking
            img_target = img_target.numpy()
            img_contra = img_contra.numpy()

        # Ensure we have (H, W)
        if len(img_target.shape) == 3:
            img_target = img_target.squeeze(0)
        if len(img_contra.shape) == 3:
            img_contra = img_contra.squeeze(0)

        # Stack
        target_tensor = np.stack(
            [img_target, age_map, implant_map], axis=0
        )  # (3, H, W)
        contra_tensor = np.stack(
            [img_contra, age_map, implant_map], axis=0
        )  # (3, H, W)

        # Convert to Torch
        target_tensor = torch.from_numpy(target_tensor).float()
        contra_tensor = torch.from_numpy(contra_tensor).float()

        # 5. Return
        sample = {
            "target": target_tensor,
            "contra": contra_tensor,
        }

        if self.mode in ["train", "val"]:
            sample["label"] = torch.tensor(row["cancer"], dtype=torch.float32)

        if self.mode == "test":
            sample["prediction_id"] = row["prediction_id"]

        return sample


def get_transforms(mode="train", img_size=(768, 768)):
    """
    Returns Albumentations transforms.
    Synchronized geometric augmentations for Siamese inputs.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Limit rotation to avoid losing too much context
                A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                # Shift only, no scaling/zooming to preserve density resolution
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.0,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
            ],
            additional_targets={"image_contra": "image"},
        )
    else:
        # No test-time augmentation for now, resizing handled in loader
        return None


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_stats=True
):
    """
    Factory function to create DataLoaders for Train, Val, and Test.
    """
    set_seed(Config.SEED)

    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Compute/Load Age Stats (from Train set only)
    age_stats = compute_age_stats(
        df_train, Config.CACHE_DIR, load_cached=load_cached_stats
    )

    # 3. Create Datasets
    train_dataset = BreastCancerPairedDataset(
        df_train,
        age_stats,
        Config.IMG_SIZE,
        transforms=get_transforms("train", Config.IMG_SIZE),
        mode="train",
    )

    val_dataset = BreastCancerPairedDataset(
        df_val,
        age_stats,
        Config.IMG_SIZE,
        transforms=get_transforms("val", Config.IMG_SIZE),
        mode="val",
    )

    test_dataset = BreastCancerPairedDataset(
        df_test,
        age_stats,
        Config.IMG_SIZE,
        transforms=get_transforms("test", Config.IMG_SIZE),
        mode="test",
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
