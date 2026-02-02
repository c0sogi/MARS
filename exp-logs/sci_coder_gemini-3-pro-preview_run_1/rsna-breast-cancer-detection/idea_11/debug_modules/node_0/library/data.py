import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_age_stats(df_train, load_cached_data=True):
    """
    Computes or loads the mean and standard deviation of patient age.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "age_stats.npy")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            stats = np.load(cache_path)
            return stats[0], stats[1]
        except Exception:
            pass  # Fallback to compute if load fails

    # Compute stats
    valid_ages = df_train["age"].dropna()
    mean_age = valid_ages.mean()
    std_age = valid_ages.std()

    # Handle edge cases
    if pd.isna(std_age) or std_age == 0:
        std_age = 1.0
    if pd.isna(mean_age):
        mean_age = 50.0

    stats = np.array([mean_age, std_age])
    np.save(cache_path, stats)

    return mean_age, std_age


class SiameseMammographyDataset(Dataset):
    def __init__(self, df, img_dir, age_stats, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            img_dir (str): Base directory for images.
            age_stats (tuple): (mean_age, std_age) for normalization.
            transform (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mean_age, self.std_age = age_stats
        self.transform = transform
        self.mode = mode

        # Create lookup for contralateral images
        # Key: (patient_id, view, laterality) -> file_path
        self.lookup = {}
        for idx, row in self.df.iterrows():
            key = (row["patient_id"], row["view"], row["laterality"])
            self.lookup[key] = row["file_path"]

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path):
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Fail Loudly check
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")

        try:
            # Load as grayscale
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"cv2 returned None for {full_path}")
        except Exception as e:
            raise ValueError(f"Error loading image {full_path}: {e}")

        # Handle different depths/channels
        if len(img.shape) == 3:
            img = img[:, :, 0]  # Take first channel if RGB/RGBA

        # Normalize to 0-1 float32
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32) / 255.0

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Identify Target
        target_path = row["file_path"]
        patient_id = row["patient_id"]
        view = row["view"]
        lat = row["laterality"]

        # 2. Identify Contralateral
        contra_lat = "R" if lat == "L" else "L"
        contra_key = (patient_id, view, contra_lat)
        contra_path = self.lookup.get(contra_key, None)

        # 3. Load Images
        img_target = self._load_image(target_path)

        if contra_path:
            img_contra = self._load_image(contra_path)
        else:
            # Create zero tensor matching target size
            img_contra = np.zeros_like(img_target)

        # Resize to Config.IMG_SIZE before augmentation
        target_h, target_w = img_target.shape
        if (target_h, target_w) != Config.IMG_SIZE:
            img_target = cv2.resize(
                img_target,
                (Config.IMG_SIZE[1], Config.IMG_SIZE[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            img_contra = cv2.resize(
                img_contra,
                (Config.IMG_SIZE[1], Config.IMG_SIZE[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # 4. Augmentation (Synchronized)
        # Albumentations expects HWC, so expand dims
        img_target = np.expand_dims(img_target, axis=-1)
        img_contra = np.expand_dims(img_contra, axis=-1)

        if self.transform:
            # Pass both images to the transform
            res = self.transform(image=img_target, image_contra=img_contra)
            img_target = res["image"]
            img_contra = res["image_contra"]

        # Convert to CHW (Transpose)
        # Assuming transform returns numpy arrays (H, W, 1)
        img_target = img_target.transpose(2, 0, 1)
        img_contra = img_contra.transpose(2, 0, 1)

        # 5. Construct Metadata Maps
        # Age
        age = row["age"]
        if pd.isna(age):
            age = self.mean_age
        age_norm = (age - self.mean_age) / self.std_age

        # Implant
        implant = row["implant"]
        if pd.isna(implant):
            implant = 0.0
        implant = float(implant)

        # Create maps: Shape (1, H, W)
        H, W = Config.IMG_SIZE
        age_map = np.full((1, H, W), age_norm, dtype=np.float32)
        implant_map = np.full((1, H, W), implant, dtype=np.float32)

        # 6. Stack Channels -> [Image, Age, Implant]
        tensor_target = np.concatenate([img_target, age_map, implant_map], axis=0)
        tensor_contra = np.concatenate([img_contra, age_map, implant_map], axis=0)

        # Convert to Torch Tensor
        tensor_target = torch.from_numpy(tensor_target).float()
        tensor_contra = torch.from_numpy(tensor_contra).float()

        # 7. Return
        if self.mode in ["train", "val"]:
            label = torch.tensor(row["cancer"], dtype=torch.float32)
            return (tensor_target, tensor_contra), label
        else:
            return (tensor_target, tensor_contra), row["prediction_id"]


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    Synchronized augmentation for Siamese network.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5
                ),
            ],
            additional_targets={"image_contra": "image"},
        )
    else:
        return None


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test.
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # Debugging option
    if Config.DEBUG:
        df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_val = df_val.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_test = df_test.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 2. Get Age Stats (Cached)
    age_stats = get_age_stats(df_train, load_cached_data=load_cached_data)

    # 3. Create Datasets
    train_dataset = SiameseMammographyDataset(
        df_train,
        Config.TRAIN_IMG_DIR,
        age_stats,
        transform=get_transforms("train"),
        mode="train",
    )

    val_dataset = SiameseMammographyDataset(
        df_val,
        Config.TRAIN_IMG_DIR,
        age_stats,
        transform=get_transforms("val"),
        mode="val",
    )

    test_dataset = SiameseMammographyDataset(
        df_test,
        Config.TEST_IMG_DIR,
        age_stats,
        transform=get_transforms("test"),
        mode="test",
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
