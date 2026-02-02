import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG


class CyclicTimeRoll(A.ImageOnlyTransform):
    """
    Cyclic shift of the image along the time axis (width).
    This preserves the temporal pattern while shifting its position,
    acting as a strong augmentation for periodic or continuous signals.
    """

    def __init__(self, roll_limit=0.5, always_apply=False, p=0.5):
        super(CyclicTimeRoll, self).__init__(always_apply, p)
        self.roll_limit = roll_limit

    def apply(self, img, **params):
        # img is H x W x C
        # Roll along axis 1 (Width/Time)
        w = img.shape[1]
        limit = int(w * self.roll_limit)
        # Generate a random shift
        shift = np.random.randint(-limit, limit)
        return np.roll(img, shift, axis=1)

    def get_transform_init_args_names(self):
        return ("roll_limit",)


def get_transforms(data="train"):
    """
    Returns the Albumentations composition for the specified data split.
    """
    if data == "train":
        return A.Compose(
            [
                # Cyclic Time Roll: Key augmentation for this task
                CyclicTimeRoll(roll_limit=0.4, p=0.5),
                # SpecAugment Simulation: Masking blocks of time/frequency
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(CFG.img_height * 0.15),
                    max_width=int(CFG.img_width * 0.15),
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                # Standard ImageNet Normalization
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # TTA or Test
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def load_image_dict(df, load_cached_data=True, cache_name="cache"):
    """
    Loads images for all unique rec_ids in the provided dataframe.
    Returns a dictionary mapping rec_id -> image_array (H, W, 3).
    Implements caching using .npy files in the working directory.
    """
    # Ensure working directory exists
    os.makedirs(CFG.working_dir, exist_ok=True)

    imgs_path = os.path.join(CFG.working_dir, f"{cache_name}_imgs.npy")
    ids_path = os.path.join(CFG.working_dir, f"{cache_name}_ids.npy")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(imgs_path) and os.path.exists(ids_path):
        try:
            imgs = np.load(imgs_path)
            ids = np.load(ids_path)
            # Reconstruct dictionary
            print(f"Loaded {len(ids)} images from cache: {cache_name}")
            return {rec_id: img for rec_id, img in zip(ids, imgs)}
        except Exception as e:
            print(f"Cache load failed for {cache_name}: {e}. Recomputing...")

    # Process images from scratch
    print(f"Processing images for {cache_name}...")
    unique_df = df.drop_duplicates(subset=["rec_id"])
    img_list = []
    id_list = []

    for idx, row in unique_df.iterrows():
        rec_id = row["rec_id"]

        # Construct path: Use Filtered Spectrograms
        rel_path = row["file_path_spec"]
        # Replace 'spectrograms' with 'filtered_spectrograms'
        rel_path = rel_path.replace("spectrograms", "filtered_spectrograms")
        full_path = os.path.join(CFG.input_dir, rel_path)

        # Load Image
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        # Fallback to original spectrogram if filtered version is missing
        if img is None:
            fallback_path = os.path.join(CFG.input_dir, row["file_path_spec"])
            img = cv2.imread(fallback_path, cv2.IMREAD_GRAYSCALE)

        # Handle missing files (should not happen with correct metadata)
        if img is None:
            img = np.zeros((CFG.img_height, CFG.img_width), dtype=np.uint8)

        # Resize to target dimensions (Freq x Time)
        img = cv2.resize(img, (CFG.img_width, CFG.img_height))

        # Pseudo-RGB: Replicate channels
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img_list.append(img)
        id_list.append(rec_id)

    imgs_np = np.array(img_list, dtype=np.uint8)
    ids_np = np.array(id_list, dtype=np.int64)

    # Save to cache
    np.save(imgs_path, imgs_np)
    np.save(ids_path, ids_np)
    print(f"Saved {len(ids_np)} images to cache: {cache_name}")

    return {rec_id: img for rec_id, img in zip(ids_np, imgs_np)}


class BirdDataset(Dataset):
    def __init__(self, df, image_dict, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_dict = image_dict
        self.transforms = transforms

        # Extract labels if available
        self.label_cols = [c for c in df.columns if c.startswith("species_")]
        if self.label_cols:
            self.labels = self.df[self.label_cols].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rec_id = self.df.iloc[idx]["rec_id"]

        # Retrieve image from dictionary
        image = self.image_dict.get(rec_id)

        # Safety fallback
        if image is None:
            image = np.zeros((CFG.img_height, CFG.img_width, 3), dtype=np.uint8)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            image = ToTensorV2()(image=image)["image"]

        # Return image and label
        if self.labels is not None:
            return image, torch.tensor(self.labels[idx])
        else:
            # Return dummy label for test set
            return image, torch.tensor(0.0)


def get_loaders(train_df, val_df, batch_size=CFG.batch_size, load_cached_data=True):
    """
    Creates DataLoaders for training and validation.
    Pre-processes and caches the union of images in train_df and val_df.
    """
    # Combine dataframes to create a unified cache for the current fold/split
    combined_df = pd.concat([train_df, val_df], axis=0)

    # Create a cache key based on the size of the combined data
    # Ideally, this should be unique per fold, but since we use a dict,
    # we can cache the entire development set once if we wanted.
    # Here we use "dev_pool" to imply the pool of development data.
    image_dict = load_image_dict(
        combined_df, load_cached_data=load_cached_data, cache_name="dev_pool"
    )

    train_ds = BirdDataset(train_df, image_dict, transforms=get_transforms("train"))
    val_ds = BirdDataset(val_df, image_dict, transforms=get_transforms("valid"))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(test_df, batch_size=CFG.batch_size, load_cached_data=True):
    """
    Creates DataLoader for the test set.
    """
    image_dict = load_image_dict(
        test_df, load_cached_data=load_cached_data, cache_name="test_pool"
    )

    test_ds = BirdDataset(test_df, image_dict, transforms=get_transforms("valid"))

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    return test_loader
