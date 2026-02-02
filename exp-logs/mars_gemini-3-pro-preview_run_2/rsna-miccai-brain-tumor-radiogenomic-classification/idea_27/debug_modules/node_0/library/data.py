import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library.config import Config, seed_everything
from library.utils import read_dicom_robust, resize_image, normalize_min_max

# Ensure reproducibility
seed_everything(Config.SEED)


def get_sorted_files(dir_path):
    """
    Returns a sorted list of file paths in a directory.
    Assumes filenames are like 'Image-123.dcm'.
    """
    if not os.path.exists(dir_path):
        return []

    files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]
    # Sort by the integer number in the filename
    try:
        files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
    except IndexError:
        files.sort()  # Fallback
    return files


def calculate_anchor(flair_path):
    """
    Calculates the optimal anchor slice index based on Sum of Intensity
    within the 15%-85% depth range.
    """
    files = get_sorted_files(flair_path)
    num_slices = len(files)

    if num_slices == 0:
        return 0

    # Define search range
    start_idx = int(num_slices * Config.ROI_SEARCH_MIN)
    end_idx = int(num_slices * Config.ROI_SEARCH_MAX)

    # Safety check for very small volumes
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = num_slices

    max_intensity_sum = -1
    best_idx = 0

    # Iterate through the search range
    for i in range(start_idx, end_idx):
        f_path = os.path.join(flair_path, files[i])
        img = read_dicom_robust(f_path)

        # Calculate metric: Sum of Intensity
        current_sum = np.sum(img)

        if current_sum > max_intensity_sum:
            max_intensity_sum = current_sum
            best_idx = i

    return best_idx


def get_roi_anchors(df, load_cached_data=True):
    """
    Generates or loads the ROI anchor indices for the given dataframe.
    Implements the required caching logic.
    """
    cache_path = Config.ROI_CACHE_PATH

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            # Convert to dict for O(1) lookup: {BraTS21ID: anchor_idx}
            # Ensure types match
            cache_dict = dict(zip(cache_df["BraTS21ID"], cache_df["anchor_idx"]))

            # Check if all IDs in current df are in cache. If not, we might need to recompute or partial compute.
            # For simplicity in this context, if cache exists but misses keys, we recompute or append.
            # Here we will assume if cache exists it's valid for the dataset, otherwise recompute.
            missing_ids = [bid for bid in df["BraTS21ID"] if bid not in cache_dict]
            if not missing_ids:
                print(f"Loaded ROI anchors from cache: {cache_path}")
                return cache_dict
            else:
                print(
                    f"Cache found but missing {len(missing_ids)} IDs. Recomputing/Updating..."
                )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing ROI anchors (this may take a while)...")
    anchors = {}

    # We need to process unique IDs. The df might be train, val, or test.
    # To be safe, we should ideally process all available data or just what's requested.
    # Here we process the rows in the provided df.

    unique_ids = df["BraTS21ID"].unique()

    for subject_id in unique_ids:
        # Find the row for this subject
        row = df[df["BraTS21ID"] == subject_id].iloc[0]
        full_flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])

        anchor = calculate_anchor(full_flair_path)
        anchors[subject_id] = anchor

    # 3. Save to cache
    # We merge with existing cache if possible to avoid losing previous work
    final_dict = anchors
    if os.path.exists(cache_path):
        try:
            existing_df = pd.read_parquet(cache_path)
            existing_dict = dict(
                zip(existing_df["BraTS21ID"], existing_df["anchor_idx"])
            )
            existing_dict.update(anchors)
            final_dict = existing_dict
        except:
            pass

    cache_out_df = pd.DataFrame(
        [{"BraTS21ID": k, "anchor_idx": v} for k, v in final_dict.items()]
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_out_df.to_parquet(cache_path, index=False)
    print(f"Saved ROI anchors to cache: {cache_path}")

    return final_dict


class MRIDataset(Dataset):
    def __init__(self, df, anchor_dict, is_train=False, transform=None):
        self.df = df.reset_index(drop=True)
        self.anchor_dict = anchor_dict
        self.is_train = is_train

        # Define Augmentations
        if transform is not None:
            self.transform = transform
        else:
            if self.is_train:
                self.transform = A.Compose(
                    [
                        A.HorizontalFlip(p=Config.AUG_PROB),
                        A.VerticalFlip(p=Config.AUG_PROB),
                        A.Rotate(
                            limit=Config.AUG_ROTATION_DEGREES,
                            p=Config.AUG_PROB,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                        ),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # Get Anchor
        anchor_idx = self.anchor_dict.get(subject_id, 0)

        # Construct Tensor
        image_tensor = self.construct_tensor(row, anchor_idx)

        # Apply Transforms
        # Albumentations expects (H, W, C)
        augmented = self.transform(image=image_tensor)
        image = augmented["image"]  # Returns (C, H, W) due to ToTensorV2

        # Get Label
        if "MGMT_value" in row:
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
        else:
            label = torch.tensor(0.0, dtype=torch.float32)  # Dummy for test

        return image, label

    def construct_tensor(self, row, anchor_idx):
        """
        Constructs the 12-channel input tensor based on the Asymmetric Grouped EfficientNet logic.
        """
        # Paths
        paths = {
            "FLAIR": os.path.join(Config.INPUT_DIR, row["path_FLAIR"]),
            "T1w": os.path.join(Config.INPUT_DIR, row["path_T1w"]),
            "T1wCE": os.path.join(Config.INPUT_DIR, row["path_T1wCE"]),
            "T2w": os.path.join(Config.INPUT_DIR, row["path_T2w"]),
        }

        # Get sorted file lists
        files_map = {mod: get_sorted_files(path) for mod, path in paths.items()}

        # Define offsets and clamping logic
        offsets = [-Config.STRIDE, 0, Config.STRIDE]

        channels = []

        # Helper to read specific slice
        def read_slice(modality, offset):
            file_list = files_map[modality]
            n_files = len(file_list)

            if n_files == 0:
                return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

            # Calculate target index
            target_idx = anchor_idx + offset

            # Edge Clamping
            if target_idx < 0:
                target_idx = 0
            if target_idx >= n_files:
                target_idx = n_files - 1

            # Read
            img_path = os.path.join(paths[modality], file_list[target_idx])
            img = read_dicom_robust(img_path)
            img = resize_image(img, size=Config.IMG_SIZE)
            img = normalize_min_max(img)
            return img

        # Group 1 (Slice -5): [FLAIR, T1w, T1wCE]
        channels.append(read_slice("FLAIR", offsets[0]))
        channels.append(read_slice("T1w", offsets[0]))
        channels.append(read_slice("T1wCE", offsets[0]))

        # Group 2 (Slice 0): [FLAIR, T1w, T1wCE]
        channels.append(read_slice("FLAIR", offsets[1]))
        channels.append(read_slice("T1w", offsets[1]))
        channels.append(read_slice("T1wCE", offsets[1]))

        # Group 3 (Slice +5): [FLAIR, T1w, T1wCE]
        channels.append(read_slice("FLAIR", offsets[2]))
        channels.append(read_slice("T1w", offsets[2]))
        channels.append(read_slice("T1wCE", offsets[2]))

        # Group 4 (Context): [T2w(-5), T2w(0), T2w(+5)]
        channels.append(read_slice("T2w", offsets[0]))
        channels.append(read_slice("T2w", offsets[1]))
        channels.append(read_slice("T2w", offsets[2]))

        # Stack channels -> (H, W, 12)
        img_stack = np.stack(channels, axis=-1)

        return img_stack.astype(np.float32)


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Combine unique IDs to generate cache for all
    all_ids_df = pd.concat(
        [
            train_df[["BraTS21ID", "path_FLAIR"]],
            val_df[["BraTS21ID", "path_FLAIR"]],
            test_df[["BraTS21ID", "path_FLAIR"]],
        ]
    ).drop_duplicates(subset=["BraTS21ID"])

    # Generate/Load Anchors
    anchor_dict = get_roi_anchors(all_ids_df, load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = MRIDataset(train_df, anchor_dict, is_train=True)
    val_dataset = MRIDataset(val_df, anchor_dict, is_train=False)
    test_dataset = MRIDataset(test_df, anchor_dict, is_train=False)

    # Create DataLoaders
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
