import os
import ast
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def decode_polyline(polyline_str, shape, thickness=15):
    """
    Decodes a stringified list of coordinates into a binary mask.

    Args:
        polyline_str (str): String representation of list of points [[x, y], ...].
        shape (tuple): (height, width) of the output mask.
        thickness (int): Thickness of the line to draw.

    Returns:
        np.ndarray: Binary mask of shape (height, width).
    """
    if not isinstance(polyline_str, str) or pd.isna(polyline_str):
        return np.zeros(shape, dtype=np.uint8)

    try:
        points = ast.literal_eval(polyline_str)
        points = np.array(points, dtype=np.int32)
    except (ValueError, SyntaxError):
        return np.zeros(shape, dtype=np.uint8)

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.polylines(mask, [points], isClosed=False, color=1, thickness=thickness)

    return mask


def process_annotations(load_cached_data=True):
    """
    Loads and processes segmentation annotations. Caches the result to parquet.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame with 'StudyInstanceUID' and 'rle_list'.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "cached_annotations.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass  # Fallback to re-processing if cache is corrupt

    # 2. Process from scratch
    if not os.path.exists(Config.ANNOTATION_PATH):
        # Return empty DF if file missing
        return pd.DataFrame(columns=["StudyInstanceUID", "rle_list"])

    df = pd.read_csv(Config.ANNOTATION_PATH)

    # Determine RLE column name (Standard is 'data' or 'EncodedPixels')
    rle_col = "data"
    if rle_col not in df.columns:
        if "EncodedPixels" in df.columns:
            rle_col = "EncodedPixels"
        else:
            # Fallback: return empty if format is unknown
            return pd.DataFrame(columns=["StudyInstanceUID", "rle_list"])

    # Group RLEs by StudyInstanceUID to get all lines for one image
    grouped = df.groupby("StudyInstanceUID")[rle_col].apply(list).reset_index()
    grouped.rename(columns={rle_col: "rle_list"}, inplace=True)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    grouped.to_parquet(cache_path)

    return grouped


def get_transforms(data="train"):
    """
    Returns Albumentations transforms for the dataset.

    Strategy:
    - CLAHE for contrast enhancement.
    - Aspect-Ratio Preserving Resizing (Letterboxing) to 640x640.
    - Normalization.
    """
    transforms_list = [
        A.CLAHE(p=1.0),
        A.LongestMaxSize(max_size=Config.IMAGE_SIZE[0], p=1.0),
        A.PadIfNeeded(
            min_height=Config.IMAGE_SIZE[0],
            min_width=Config.IMAGE_SIZE[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=1.0,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(transforms_list)


class CatheterDataset(Dataset):
    def __init__(self, df, annotation_map=None, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            annotation_map (pd.DataFrame): DF mapping UID to rle_list.
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'valid', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Convert annotation dataframe to a dictionary for O(1) lookup
        self.annotations = {}
        if annotation_map is not None and not annotation_map.empty:
            self.annotations = dict(
                zip(annotation_map["StudyInstanceUID"], annotation_map["rle_list"])
            )

        self.label_cols = Config.LABEL_COLS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row["StudyInstanceUID"]

        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        image = cv2.imread(img_path)

        if image is None:
            # Fallback for missing/corrupt images
            image = np.zeros(
                (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create Mask
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if uid in self.annotations:
            rle_list = self.annotations[uid]
            for rle in rle_list:
                decoded = decode_polyline(rle, (h, w))
                mask = np.maximum(mask, decoded)

        # Apply Transforms
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Handle Mask Shape for PyTorch (Add channel dim: 1xHxW)
        if isinstance(mask, torch.Tensor):
            mask = mask.float().unsqueeze(0)
        else:
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        # Handle Labels
        if self.mode in ["train", "valid"]:
            labels = row[self.label_cols].values.astype(np.float32)
            labels = torch.tensor(labels)
            return image, labels, mask
        else:
            # Test mode: return zeros for labels
            return image, torch.zeros(len(self.label_cols)), mask


def get_loaders(
    train_batch_size=Config.BATCH_SIZE,
    val_batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Creates DataLoaders for train and validation sets.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Process Annotations
    annotation_map = process_annotations(load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = CatheterDataset(
        train_df,
        annotation_map=annotation_map,
        transforms=get_transforms("train"),
        mode="train",
    )

    val_dataset = CatheterDataset(
        val_df,
        annotation_map=annotation_map,
        transforms=get_transforms("valid"),
        mode="valid",
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=Config.BATCH_SIZE):
    """
    Creates DataLoader for the test set.
    """
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    test_dataset = CatheterDataset(
        test_df, annotation_map=None, transforms=get_transforms("test"), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
