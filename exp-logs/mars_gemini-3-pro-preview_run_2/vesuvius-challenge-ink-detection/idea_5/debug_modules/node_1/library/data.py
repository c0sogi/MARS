import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(phase):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        phase (str): 'train' or 'valid'.

    Returns:
        A.Compose: Composed transformations.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def load_or_generate_fragment_mips(fragment_id, volume_dir, load_cached_data=True):
    """
    Loads cached MIPs from disk or generates them from raw TIFF slices if not found.

    Args:
        fragment_id (str): ID of the fragment (e.g., '1', '2', 'a').
        volume_dir (str): Path to the directory containing .tif slices.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: 3-channel image of shape (H, W, 3) containing stratified MIPs.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{fragment_id}_mips.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            mips = np.load(cache_path)
            # Basic validation of shape
            if mips.ndim == 3 and mips.shape[2] == Config.IN_CHANNELS:
                return mips
            else:
                print(
                    f"Cached file {cache_path} has incorrect shape {mips.shape}. Regenerating."
                )
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Regenerating.")

    # 2. Generate from scratch
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    channels = []
    # Z_RANGES = [(22, 29), (29, 36), (36, 43)]
    for start_z, end_z in Config.Z_RANGES:
        sub_volume = []
        for z in range(start_z, end_z):
            slice_path = os.path.join(Config.INPUT_DIR, volume_dir, f"{z:02d}.tif")
            if not os.path.exists(slice_path):
                # Fallback or error; here we assume data integrity based on metadata script
                raise FileNotFoundError(f"Slice not found: {slice_path}")

            # Load slice
            img = cv2.imread(slice_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read image: {slice_path}")
            sub_volume.append(img)

        # Stack slices for this range: (H, W, D_sub)
        sub_volume = np.stack(sub_volume, axis=-1)

        # Compute MIP (Maximum Intensity Projection) along depth
        mip = np.max(sub_volume, axis=-1)
        channels.append(mip)

    # Stack channels: (H, W, C)
    full_mips = np.stack(channels, axis=-1)

    # Save to cache
    np.save(cache_path, full_mips)

    return full_mips


class InkDataset(Dataset):
    def __init__(self, metadata_df, fragment_data, transforms=None, mode="train"):
        """
        Args:
            metadata_df (pd.DataFrame): Dataframe containing patch coordinates.
            fragment_data (dict): Dictionary mapping fragment_id to 3D numpy arrays (H, W, C).
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'valid', or 'test'.
        """
        self.metadata = metadata_df
        self.fragment_data = fragment_data
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        frag_id = str(row["fragment_id"])
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        # Retrieve the full cached volume for this fragment
        # Shape: (Full_H, Full_W, C)
        full_image = self.fragment_data[frag_id]

        # Crop the specific patch
        # Note: Metadata generation ensured x+w and y+h are within bounds or handled.
        # We perform a safe crop just in case.
        img_h, img_w = full_image.shape[:2]
        y_end = min(y + h, img_h)
        x_end = min(x + w, img_w)

        image_patch = full_image[y:y_end, x:x_end, :]

        # Pad if necessary (though metadata usually filters small edge chunks,
        # consistent size is good for batching)
        if image_patch.shape[0] != h or image_patch.shape[1] != w:
            pad_h = h - image_patch.shape[0]
            pad_w = w - image_patch.shape[1]
            image_patch = np.pad(
                image_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
            )

        # Normalize: uint16 -> float32 [0, 1]
        image_patch = image_patch.astype(np.float32) / 65535.0

        mask_patch = None

        if self.mode in ["train", "valid"]:
            # Load Mask and Label
            # Note: We load these from disk on the fly as they are small 2D PNGs.
            # Construct paths relative to input dir
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            label_path = os.path.join(Config.INPUT_DIR, row["label_path"])

            # Load binary mask (valid area) and label (ink)
            # Use IMREAD_GRAYSCALE
            valid_mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            ink_label_full = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            # Crop
            valid_mask_patch = valid_mask_full[y:y_end, x:x_end]
            ink_label_patch = ink_label_full[y:y_end, x:x_end]

            # Pad if necessary
            if valid_mask_patch.shape[0] != h or valid_mask_patch.shape[1] != w:
                valid_mask_patch = np.pad(
                    valid_mask_patch, ((0, pad_h), (0, pad_w)), mode="constant"
                )
                ink_label_patch = np.pad(
                    ink_label_patch, ((0, pad_h), (0, pad_w)), mode="constant"
                )

            # Normalize to [0, 1]
            valid_mask_patch = (valid_mask_patch > 0).astype(np.float32)
            ink_label_patch = (ink_label_patch > 0).astype(np.float32)

            # For training, the target is the ink label.
            # We might use the valid_mask for loss weighting if needed,
            # but standard setup is just predicting ink.
            mask_patch = ink_label_patch

        # Apply Transforms
        if self.transforms:
            if mask_patch is not None:
                augmented = self.transforms(image=image_patch, mask=mask_patch)
                image_patch = augmented["image"]
                mask_patch = augmented["mask"]
            else:
                augmented = self.transforms(image=image_patch)
                image_patch = augmented["image"]

        if self.mode in ["train", "valid"]:
            # Ensure mask is (1, H, W)
            if mask_patch.ndim == 2:
                mask_patch = mask_patch.unsqueeze(0)
            return image_patch, mask_patch
        else:
            return image_patch


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Prepares and returns DataLoaders for training and validation.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.
        debug (bool): If True, subsamples data for quick testing.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # 1. Load Metadata
    train_csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(Config.METADATA_DIR, "validation.csv")

    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)

    if debug:
        df_train = df_train.sample(
            n=min(len(df_train), 50), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), 20), random_state=Config.SEED
        ).reset_index(drop=True)

    # 2. Identify all unique fragments needed
    # We need to load/generate MIPs for all fragments in train and val
    unique_fragments = pd.concat(
        [df_train["fragment_id"], df_val["fragment_id"]]
    ).unique()

    # We also need the volume directory paths.
    # We can get this from the dataframe.
    # Create a map: fragment_id -> volume_path
    frag_to_vol = {}

    # Helper to fill map
    def fill_map(df):
        for _, row in df.iterrows():
            fid = str(row["fragment_id"])
            if fid not in frag_to_vol:
                frag_to_vol[fid] = row["volume_path"]

    fill_map(df_train)
    fill_map(df_val)

    # 3. Load/Cache Fragment Data
    fragment_storage = {}
    print(f"Preparing data for fragments: {unique_fragments}")

    for fid in unique_fragments:
        fid = str(fid)
        vol_path = frag_to_vol[fid]
        print(f"Processing fragment {fid}...")

        # This function handles the logic: Load Cache OR Compute & Save
        mips = load_or_generate_fragment_mips(
            fid, vol_path, load_cached_data=load_cached_data
        )
        fragment_storage[fid] = mips

    # 4. Create Datasets
    train_dataset = InkDataset(
        metadata_df=df_train,
        fragment_data=fragment_storage,
        transforms=get_transforms("train"),
        mode="train",
    )

    val_dataset = InkDataset(
        metadata_df=df_val,
        fragment_data=fragment_storage,
        transforms=get_transforms("valid"),
        mode="valid",
    )

    # 5. Create Loaders
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

    return train_loader, val_loader
