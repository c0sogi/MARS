import os
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import read_dicom_robust, get_flair_anchor


def process_subject(row, input_dir, img_size=224):
    """
    Process a single subject:
    1. Identify FLAIR anchor slice.
    2. Extract 3 slices (Center, Upper, Lower) for all 4 modalities (Cite solution_lesson_node_00029).
    3. Stack into (12, H, W) tensor.
    """
    # 1. Identify Anchor from FLAIR
    flair_path_rel = row["path_FLAIR"]
    flair_full_path = os.path.join(input_dir, flair_path_rel)
    anchor_idx = get_flair_anchor(flair_full_path)

    # 2. Define Slices (Cite solution_lesson_node_00010)
    # Stride 5. Center Slice at anchor. Upper at anchor-5. Lower at anchor+5.
    slice_indices = [
        anchor_idx - Config.SLICE_STRIDE,
        anchor_idx,
        anchor_idx + Config.SLICE_STRIDE,
    ]

    modalities = ["path_FLAIR", "path_T1w", "path_T1wCE", "path_T2w"]
    all_slices = []

    for mod_col in modalities:
        mod_path_rel = row[mod_col]
        mod_full_path = os.path.join(input_dir, mod_path_rel)

        for idx in slice_indices:
            # Construct filename: Image-{idx}.dcm
            fname = f"Image-{idx}.dcm"
            fpath = os.path.join(mod_full_path, fname)

            # Load slice (returns zeros if missing/invalid)
            img = read_dicom_robust(fpath)

            # Resize to target size
            if img.shape != (img_size, img_size):
                img = cv2.resize(
                    img, (img_size, img_size), interpolation=cv2.INTER_LINEAR
                )

            # Normalize to [0, 1] per channel
            if img.max() > 0:
                img = img.astype(np.float32)
                img = (img - img.min()) / (img.max() - img.min())
            else:
                img = np.zeros((img_size, img_size), dtype=np.float32)

            all_slices.append(img)

    # Stack: (12, H, W)
    # Order: FLAIR(S1, S2, S3), T1w(S1, S2, S3), etc.
    img_tensor = np.stack(all_slices, axis=0)
    return img_tensor


def prepare_data_split(metadata_path, cache_path, load_cached_data=True):
    """
    Loads metadata, processes images (or loads from cache), and returns arrays.
    """
    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data_dict = np.load(cache_path, allow_pickle=True).item()
            return data_dict["images"], data_dict["labels"], data_dict["ids"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    images_list = []
    labels_list = []
    ids_list = []

    total = len(df)
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"  Processed {idx}/{total} subjects...")

        img_tensor = process_subject(row, Config.INPUT_DIR, Config.IMG_SIZE)
        images_list.append(img_tensor)
        ids_list.append(row["BraTS21ID"])

        if "MGMT_value" in row:
            labels_list.append(row["MGMT_value"])
        else:
            labels_list.append(-1)  # Placeholder for test

    images_array = np.array(images_list, dtype=np.float32)
    labels_array = np.array(labels_list, dtype=np.float32)
    ids_array = np.array(ids_list, dtype=np.int64)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Saving processed data to {cache_path}...")
    np.save(
        cache_path, {"images": images_array, "labels": labels_array, "ids": ids_array}
    )

    return images_array, labels_array, ids_array


class MIPDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None, is_test=False):
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image is (C, H, W)
        image = self.images[idx]

        if self.transforms:
            # Albumentations expects (H, W, C)
            image = np.transpose(image, (1, 2, 0))
            augmented = self.transforms(image=image)
            image = augmented["image"]  # Returns Tensor (C, H, W) via ToTensorV2
        else:
            image = torch.from_numpy(image).float()

        if self.is_test:
            return image

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    Handles caching, processing, and transform definition.
    """
    # Define Augmentations
    # Train: Geometric augmentations (Flip, Rotate)
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=Config.ROTATION_DEGREES, p=0.5),
            ToTensorV2(),
        ]
    )

    # Val/Test: Just convert to Tensor
    val_transform = A.Compose([ToTensorV2()])

    # 1. Train Data
    print("Preparing Training Data...")
    train_imgs, train_lbls, _ = prepare_data_split(
        Config.TRAIN_METADATA_PATH, Config.CACHE_TRAIN_PATH, load_cached_data
    )
    train_dataset = MIPDataset(train_imgs, train_lbls, transforms=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # 2. Val Data
    print("Preparing Validation Data...")
    val_imgs, val_lbls, _ = prepare_data_split(
        Config.VAL_METADATA_PATH, Config.CACHE_VAL_PATH, load_cached_data
    )
    val_dataset = MIPDataset(val_imgs, val_lbls, transforms=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Test Data
    print("Preparing Test Data...")
    test_imgs, _, test_ids = prepare_data_split(
        Config.TEST_METADATA_PATH, Config.CACHE_TEST_PATH, load_cached_data
    )
    test_dataset = MIPDataset(
        test_imgs, labels=None, transforms=val_transform, is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
