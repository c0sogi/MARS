import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from library.config import Config
from library.utils import pad_image, rle_decode


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(Config.SEED)


def load_cached_data(metadata_df, cache_prefix, load_cached_data=True):
    """
    Loads image, mask, depth, and id data.
    Uses caching mechanism to store processed numpy arrays.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_path = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    mask_path = os.path.join(cache_dir, f"{cache_prefix}_masks.npy")
    depth_path = os.path.join(cache_dir, f"{cache_prefix}_depths.npy")
    id_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    has_masks = "rle_mask" in metadata_df.columns

    # Check if files exist
    files_exist = (
        os.path.exists(img_path)
        and os.path.exists(depth_path)
        and os.path.exists(id_path)
    )
    if has_masks:
        files_exist = files_exist and os.path.exists(mask_path)

    if load_cached_data and files_exist:
        print(f"Loading {cache_prefix} data from cache...")
        images = np.load(img_path)
        depths = np.load(depth_path)
        ids = np.load(id_path, allow_pickle=True)
        if has_masks:
            masks = np.load(mask_path)
        else:
            masks = None
        return images, masks, depths, ids

    print(f"Processing {cache_prefix} data from scratch...")
    images = []
    masks = []
    depths = []
    ids = []

    for _, row in metadata_df.iterrows():
        # Load Image
        # Metadata paths are relative to input dir
        full_img_path = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Load as Color (BGR) then convert to RGB
        # We load as RGB to apply ImageNet normalization correctly later
        img = cv2.imread(full_img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found at {full_img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

        # Load Mask if present
        if has_masks:
            rle = row["rle_mask"]
            if pd.isna(rle) or rle == "":
                mask = np.zeros(
                    (Config.ORIG_IMG_SIZE, Config.ORIG_IMG_SIZE), dtype=np.uint8
                )
            else:
                mask = rle_decode(
                    rle, shape=(Config.ORIG_IMG_SIZE, Config.ORIG_IMG_SIZE)
                )
            masks.append(mask)

        # Depth and ID
        depths.append(row["z"])
        ids.append(row["id"])

    images = np.array(images, dtype=np.uint8)
    depths = np.array(depths, dtype=np.float32)
    ids = np.array(ids)

    np.save(img_path, images)
    np.save(depth_path, depths)
    np.save(id_path, ids)

    if has_masks:
        masks = np.array(masks, dtype=np.uint8)
        np.save(mask_path, masks)
    else:
        masks = None

    return images, masks, depths, ids


class SaltDataset(Dataset):
    def __init__(
        self,
        images,
        depths,
        ids,
        masks=None,
        phase="train",
        depth_stats=(0.0, 1.0),
        transform=None,
    ):
        self.images = images
        self.depths = depths
        self.ids = ids
        self.masks = masks
        self.phase = phase
        self.depth_mean, self.depth_std = depth_stats
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]  # (H, W, 3)
        depth = self.depths[idx]  # float
        img_id = self.ids[idx]  # str

        mask = None
        if self.masks is not None:
            mask = self.masks[idx]  # (H, W)

        # 1. Augmentation (Train only)
        if self.phase == "train" and self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # 2. Padding (Reflection)
        image = pad_image(image)
        if mask is not None:
            mask = pad_image(mask)

        # 3. Normalization & Tensor Conversion
        # Normalize Image (ImageNet stats)
        image = image.astype(np.float32) / 255.0
        mean = np.array(Config.NORM_MEAN, dtype=np.float32)
        std = np.array(Config.NORM_STD, dtype=np.float32)
        image = (image - mean) / std

        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        # Collapse to 1 channel (Mean) as per strategy
        # This matches the model adaptation where RGB weights are summed.
        image = torch.mean(image, dim=0, keepdim=True)

        # Mask to Tensor
        if mask is not None:
            mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)

        # 4. Depth Handling
        # Normalize depth
        z = (depth - self.depth_mean) / self.depth_std

        # Bernoulli Masking / Injection
        if self.phase == "train":
            # With probability DEPTH_DROP_RATE, set z to 0 (mean)
            if np.random.random() < Config.DEPTH_DROP_RATE:
                z = 0.0
        else:
            # Val/Test: Always inject 0 (Fallback mode)
            z = 0.0

        z = torch.tensor([z], dtype=torch.float32)

        if mask is not None:
            return image, mask, z, img_id
        return image, z, img_id


def get_dataloaders(use_cache=True):
    """
    Creates and returns dataloaders for train, val, and test sets.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Calculate Depth Stats from Training Data
    depth_mean = train_df["z"].mean()
    depth_std = train_df["z"].std()
    depth_stats = (depth_mean, depth_std)

    print(f"Depth Stats - Mean: {depth_mean:.4f}, Std: {depth_std:.4f}")

    # Load Data (Cached or Scratch)
    train_imgs, train_masks, train_depths, train_ids = load_cached_data(
        train_df, "train", use_cache
    )
    val_imgs, val_masks, val_depths, val_ids = load_cached_data(
        val_df, "val", use_cache
    )
    test_imgs, test_masks, test_depths, test_ids = load_cached_data(
        test_df, "test", use_cache
    )

    # Define Augmentations
    train_transform = A.Compose(
        [
            A.ElasticTransform(
                alpha=Config.ELASTIC_ALPHA, sigma=Config.ELASTIC_SIGMA, p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT,
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
        ],
        p=Config.AUG_PROB,
    )

    # Create Datasets
    train_dataset = SaltDataset(
        train_imgs,
        train_depths,
        train_ids,
        train_masks,
        phase="train",
        depth_stats=depth_stats,
        transform=train_transform,
    )

    val_dataset = SaltDataset(
        val_imgs, val_depths, val_ids, val_masks, phase="val", depth_stats=depth_stats
    )

    test_dataset = SaltDataset(
        test_imgs, test_depths, test_ids, None, phase="test", depth_stats=depth_stats
    )

    # Create Dataloaders
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
