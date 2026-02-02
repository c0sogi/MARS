import os
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import load_image


def get_age_stats(df, load_cached_data=True):
    """
    Computes or loads cached age statistics (mean, std) for normalization.

    Args:
        df (pd.DataFrame): DataFrame containing the 'age' column.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (mean_age, std_age)
    """
    cache_path = os.path.join(Config.CACHE_DIR, "age_stats.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            stats = np.load(cache_path)
            return stats[0], stats[1]
        except Exception:
            pass  # Fallback to recompute if corrupt

    # 2. Compute from scratch
    # Filter valid ages (drop NaNs)
    valid_ages = df["age"].dropna()
    mean_age = valid_ages.mean()
    std_age = valid_ages.std()

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_path, np.array([mean_age, std_age]))

    return mean_age, std_age


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.
    Uses 'additional_targets' to ensure synchronized geometric transforms
    for the Siamese pair.
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
                # Normalize to roughly standard normal for images
                A.Normalize(mean=(0.2,), std=(0.22,)),
                ToTensorV2(),
            ],
            # Apply same transforms to 'image_contra'
            additional_targets={"image_contra": "image"},
        )
    else:
        return A.Compose(
            [A.Normalize(mean=(0.2,), std=(0.22,)), ToTensorV2()],
            additional_targets={"image_contra": "image"},
        )


class BreastCancerDataset(Dataset):
    """
    Dataset for Siamese Network.
    Loads paired images (Target, Contralateral) and constructs 3-channel inputs
    (Image, Age Map, Implant Map).
    """

    def __init__(self, df, transform=None, age_mean=58.7, age_std=10.0):
        self.df = df
        self.transform = transform
        self.age_mean = age_mean
        self.age_std = age_std

        # Group by patient and view for fast contralateral lookup
        # We need to find the image with the same patient_id and view, but opposite laterality
        self.grouped = self.df.groupby(["patient_id", "view"])

        self.pairs = []
        self._prepare_pairs()

    def _prepare_pairs(self):
        """
        Pre-computes the list of (target, contralateral) pairs to avoid
        pandas lookups during training loop.
        """
        for idx, row in self.df.iterrows():
            patient_id = row["patient_id"]
            view = row["view"]
            laterality = row["laterality"]

            # Target info
            target_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            # Contralateral logic
            opp_laterality = "R" if laterality == "L" else "L"

            contra_path = None
            try:
                # Retrieve potential matches
                group = self.grouped.get_group((patient_id, view))
                # Filter for opposite laterality
                contra_rows = group[group["laterality"] == opp_laterality]

                if not contra_rows.empty:
                    # Use the first match found
                    contra_path = os.path.join(
                        Config.INPUT_DIR, contra_rows.iloc[0]["file_path"]
                    )
            except KeyError:
                # No group found (should be rare if data is consistent)
                pass

            self.pairs.append(
                {
                    "target_path": target_path,
                    "contra_path": contra_path,
                    "age": row["age"],
                    "implant": row["implant"],
                    "label": row.get("cancer", 0),  # Default to 0 for test set
                    "prediction_id": row.get(
                        "prediction_id", f"{patient_id}_{laterality}"
                    ),
                }
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        data = self.pairs[idx]

        # 1. Load Target Image (Fail Loudly)
        try:
            img_target = load_image(data["target_path"], size=Config.IMG_SIZE)
        except Exception as e:
            raise ValueError(f"Error loading target image {data['target_path']}: {e}")

        # 2. Load Contralateral Image
        img_contra = None
        if data["contra_path"]:
            # If metadata says it exists, the file MUST exist.
            if os.path.exists(data["contra_path"]):
                try:
                    img_contra = load_image(data["contra_path"], size=Config.IMG_SIZE)
                except Exception as e:
                    raise ValueError(
                        f"Error loading contralateral image {data['contra_path']}: {e}"
                    )
            else:
                # Fail loudly if file is missing but metadata expects it
                raise FileNotFoundError(
                    f"Contralateral file expected but not found: {data['contra_path']}"
                )

        # If no contralateral exists anatomically (not in metadata), use zero tensor
        if img_contra is None:
            img_contra = np.zeros_like(img_target)

        # 3. Augmentation (Synchronized)
        if self.transform:
            res = self.transform(image=img_target, image_contra=img_contra)
            img_target = res["image"]
            img_contra = res["image_contra"]
        else:
            # Fallback manual conversion if no transform provided
            img_target = torch.from_numpy(img_target).float().unsqueeze(0)
            img_contra = torch.from_numpy(img_contra).float().unsqueeze(0)

        # 4. Metadata Processing
        # Normalize Age
        age_val = (data["age"] - self.age_mean) / self.age_std
        if np.isnan(age_val):
            age_val = 0.0

        implant_val = 1.0 if data["implant"] == 1 else 0.0

        # 5. Channel Expansion (Broadcasting)
        # img tensor is [C, H, W], where C is usually 1 after ToTensorV2 (grayscale)
        _, h, w = img_target.shape

        age_map = torch.full((1, h, w), age_val, dtype=torch.float32)
        implant_map = torch.full((1, h, w), implant_val, dtype=torch.float32)

        # Concatenate: Image + Age + Implant = 3 Channels
        input_target = torch.cat([img_target, age_map, implant_map], dim=0)
        input_contra = torch.cat([img_contra, age_map, implant_map], dim=0)

        return (
            input_target,
            input_contra,
            torch.tensor(data["label"], dtype=torch.float32),
            data["prediction_id"],
        )


def get_loaders(debug=False, load_cached_data=True):
    """
    Prepares DataLoaders for training and validation.

    Args:
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): Whether to use cached statistics.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    # Debug Sampling
    if debug:
        df_train = df_train.sample(
            n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Get Age Stats (Cached or Computed)
    age_mean, age_std = get_age_stats(df_train, load_cached_data=load_cached_data)

    # Instantiate Datasets
    train_dataset = BreastCancerDataset(
        df_train, transform=get_transforms("train"), age_mean=age_mean, age_std=age_std
    )
    val_dataset = BreastCancerDataset(
        df_val, transform=get_transforms("val"), age_mean=age_mean, age_std=age_std
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader
