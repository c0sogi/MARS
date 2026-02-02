import os
import ast
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def process_annotations(load_cached_data=True):
    """
    Processes train_annotations.csv to map StudyInstanceUID to a list of polyline strings.
    Implements caching using parquet.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "cached_annotations.parquet")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_grouped = pd.read_parquet(cache_path)
            # Convert the list column back from whatever format parquet stored it in if necessary
            # Typically pandas handles lists in parquet fine, but let's ensure it's a dict mapping
            # We return a dictionary: UID -> list of polyline strings
            return df_grouped.set_index("StudyInstanceUID")["data"].to_dict()
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    if not os.path.exists(Config.ANNOTATIONS_PATH):
        # If file doesn't exist (e.g. in a test environment without annotations), return empty dict
        return {}

    df_ann = pd.read_csv(Config.ANNOTATIONS_PATH)

    # Group by StudyInstanceUID and aggregate 'data' (Polyline Strings) into a list
    # We filter out NaNs in 'data' just in case
    df_grouped = df_ann.groupby("StudyInstanceUID")["data"].apply(list).reset_index()

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df_grouped.to_parquet(cache_path, index=False)

    return df_grouped.set_index("StudyInstanceUID")["data"].to_dict()


class CatheterDataset(Dataset):
    def __init__(self, df, annotations_map=None, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            annotations_map (dict): Mapping from StudyInstanceUID to list of polyline strings.
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.annotations_map = annotations_map or {}
        self.transforms = transforms
        self.mode = mode
        self.labels = Config.LABELS

        # Pre-compute paths to avoid os.path.join in __getitem__
        # The metadata file_path is relative to input directory
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in df["file_path"].values
        ]
        self.uids = df["StudyInstanceUID"].values

        if self.mode != "test":
            self.targets = df[self.labels].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load Image
        img_path = self.file_paths[idx]
        image = cv2.imread(img_path)

        if image is None:
            # Fallback for missing images (should not happen based on EDA)
            # Create a black image of default size
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load/Generate Mask
        # Default mask is zeros (H, W)
        h, w = image.shape[:2]
        # Use uint8 for OpenCV drawing, convert to float later
        mask = np.zeros((h, w), dtype=np.uint8)

        uid = self.uids[idx]

        # If we have annotations for this image, decode and combine them
        if self.mode in ["train", "val"] and uid in self.annotations_map:
            lines_list = self.annotations_map[uid]
            for line_str in lines_list:
                try:
                    # Parse string "[[x,y], ...]" to list of points
                    points = ast.literal_eval(line_str)
                    points = np.array(points, dtype=np.int32)

                    # Reshape for cv2.polylines: (N, 1, 2)
                    points = points.reshape((-1, 1, 2))

                    # Draw line
                    # Thickness 15 is chosen to be visible after resizing (original images are ~2k-3k px)
                    cv2.polylines(mask, [points], isClosed=False, color=1, thickness=15)
                except Exception:
                    # Ignore malformed annotations
                    pass

        # Convert mask to float32 for transforms/training
        mask = mask.astype(np.float32)

        # Apply Transforms
        if self.transforms:
            # Albumentations expects mask to be passed as 'mask'
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Handle Mask Dimensions for PyTorch (Add channel dim: HxW -> 1xHxW)
        # ToTensorV2 converts image to C,H,W but mask usually stays H,W if not explicitly handled or if it's 2D
        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0)
        else:
            mask = torch.from_numpy(mask).unsqueeze(0)

        # Return Data
        if self.mode == "test":
            return image, uid
        else:
            target = torch.tensor(self.targets[idx])
            return image, target, mask


def get_transforms(mode="train", img_size=640):
    """
    Returns albumentations transforms for the specific mode.
    Implements Letterboxing (LongestMaxSize + PadIfNeeded) and CLAHE.
    """
    if mode == "train":
        return A.Compose(
            [
                # Aspect-ratio preserving resizing
                A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_CUBIC),
                A.PadIfNeeded(
                    min_height=img_size,
                    min_width=img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
                # Signal enhancement
                A.CLAHE(p=0.5),
                # Basic augmentations
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
                ),
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.2),
                # Normalization and Tensor conversion
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Val/Test: Deterministic resizing
        return A.Compose(
            [
                A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_CUBIC),
                A.PadIfNeeded(
                    min_height=img_size,
                    min_width=img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    img_size=Config.IMG_SIZE,
    debug=Config.DEBUG,
    load_cached_data=True,
):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_META)
    df_val = pd.read_csv(Config.VAL_META)
    df_test = pd.read_csv(Config.TEST_META)

    # 2. Handle Debug Mode
    if debug:
        df_train = df_train.sample(
            n=min(len(df_train), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        # Test set is usually small enough, but we can sample it too if needed
        df_test = df_test.sample(
            n=min(len(df_test), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # 3. Load Annotations Map (Cached)
    annotations_map = process_annotations(load_cached_data=load_cached_data)

    # 4. Create Datasets
    train_dataset = CatheterDataset(
        df_train,
        annotations_map=annotations_map,
        transforms=get_transforms(mode="train", img_size=img_size),
        mode="train",
    )

    val_dataset = CatheterDataset(
        df_val,
        annotations_map=annotations_map,
        transforms=get_transforms(mode="val", img_size=img_size),
        mode="val",
    )

    test_dataset = CatheterDataset(
        df_test,
        annotations_map=None,  # No annotations for test
        transforms=get_transforms(mode="test", img_size=img_size),
        mode="test",
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
