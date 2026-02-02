import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import set_seed


def load_and_cache_data(df, input_dir, cache_prefix, output_dir, load_cached_data=True):
    """
    Loads images and labels from the dataframe/directory.
    Caches the result as .npy files to speed up future runs.
    """
    os.makedirs(output_dir, exist_ok=True)

    imgs_cache_path = os.path.join(output_dir, f"{cache_prefix}_imgs.npy")
    lbls_cache_path = os.path.join(output_dir, f"{cache_prefix}_lbls.npy")
    ids_cache_path = os.path.join(output_dir, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(imgs_cache_path)
        and os.path.exists(lbls_cache_path)
        and os.path.exists(ids_cache_path)
    ):
        # print(f"Loading cached data from {output_dir}...") # Silent execution preferred
        imgs = np.load(imgs_cache_path)
        lbls = np.load(lbls_cache_path)
        ids = np.load(ids_cache_path, allow_pickle=True)
        return imgs, lbls, ids

    # 2. Process from scratch
    # print(f"Processing data for {cache_prefix}...")
    img_list = []
    lbl_list = []
    id_list = []

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            continue  # Skip missing files

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_list.append(img)
        lbl_list.append(row["has_cactus"])
        id_list.append(row["id"])

    imgs = np.array(img_list, dtype=np.uint8)
    lbls = np.array(lbl_list, dtype=np.float32)
    ids = np.array(id_list)

    # 3. Save to cache
    np.save(imgs_cache_path, imgs)
    np.save(lbls_cache_path, lbls)
    np.save(ids_cache_path, ids)

    return imgs, lbls, ids


class CactusDataset(Dataset):
    def __init__(self, images, labels, image_ids, transform=None):
        self.images = images
        self.labels = labels
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is uint8 (H, W, C)
        img = self.images[idx]
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            # torchvision transforms expect PIL Image or Tensor
            # ToPILImage is usually the first step if passing numpy array
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


def get_transforms(img_size=32, phase="train"):
    """
    Returns the transformation pipeline.
    """
    # Normalization statistics for ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # Photometric augmentation as per Idea
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


def get_dataloaders(config: Config, fold_id: int = 0, mode: str = "train"):
    """
    Factory function to create DataLoaders.

    Args:
        config: Configuration object.
        fold_id: The fold index [0, n_folds-1] to use for validation.
        mode: 'train' (returns train_loader, val_loader) or 'test' (returns test_loader).
    """
    set_seed(config.seed)

    if mode == "train":
        # Load Metadata
        df_train = pd.read_csv(config.train_metadata_path)

        # Debugging: Limit dataset size
        if config.debug and config.debug_sample_size:
            df_train = df_train.iloc[: config.debug_sample_size]

        # Load Data (Cached)
        # We use 'train_full' prefix because this contains the data before splitting into folds
        imgs, lbls, ids = load_and_cache_data(
            df_train,
            config.input_dir,
            cache_prefix=f"train_full_{len(df_train)}",
            output_dir=config.output_dir,
            load_cached_data=True,
        )

        # Stratified K-Fold Split
        skf = StratifiedKFold(
            n_splits=config.n_folds, shuffle=True, random_state=config.seed
        )

        # Get indices for the requested fold
        # We convert lbls to integer for stratification logic if they are float
        y_strat = lbls.astype(int)
        splits = list(skf.split(imgs, y_strat))

        if fold_id >= config.n_folds:
            raise ValueError(
                f"Fold ID {fold_id} out of range (n_folds={config.n_folds})"
            )

        train_idx, val_idx = splits[fold_id]

        # Subset data
        X_train, y_train, id_train = imgs[train_idx], lbls[train_idx], ids[train_idx]
        X_val, y_val, id_val = imgs[val_idx], lbls[val_idx], ids[val_idx]

        # Create Datasets
        train_dataset = CactusDataset(
            X_train,
            y_train,
            id_train,
            transform=get_transforms(config.img_size, phase="train"),
        )
        val_dataset = CactusDataset(
            X_val, y_val, id_val, transform=get_transforms(config.img_size, phase="val")
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,  # Useful for Mixup stability
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader, val_loader

    elif mode == "test":
        # Load Metadata
        df_test = pd.read_csv(config.test_metadata_path)

        # Load Data (Cached)
        imgs, lbls, ids = load_and_cache_data(
            df_test,
            config.input_dir,
            cache_prefix="test_full",
            output_dir=config.output_dir,
            load_cached_data=True,
        )

        # Create Dataset
        test_dataset = CactusDataset(
            imgs, lbls, ids, transform=get_transforms(config.img_size, phase="test")
        )

        # Create DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return test_loader

    else:
        raise ValueError(f"Unknown mode: {mode}")
