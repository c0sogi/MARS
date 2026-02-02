import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def read_dicom(path):
    """
    Reads a DICOM file. Tries pydicom first, then OpenCV.
    Returns a numpy array.
    """
    # Attempt to use pydicom if available
    try:
        import pydicom

        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        return img
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback to OpenCV
    try:
        # cv2.IMREAD_UNCHANGED is needed for 16-bit DICOMs
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    # Return zero placeholder if reading fails
    return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)


class BraTSDataset(Dataset):
    def __init__(self, metadata_df, split="train", load_cached_data=True):
        """
        Args:
            metadata_df: DataFrame containing subject IDs and paths.
            split: 'train', 'val', or 'test'.
            load_cached_data: Whether to load/save cached file lists.
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.split = split
        self.is_train = split == "train"

        # Cache setup
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        # Unique cache name based on split to avoid collisions
        # Changed name to avoid conflict with volumetric cache
        self.cache_path = os.path.join(
            self.cache_dir, f"cached_file_lists_2d_{split}.parquet"
        )

        # Load or generate file lists (Geometric Sampling: Middle Slice)
        self.data = self._prepare_data(load_cached_data)

        # Augmentations
        # We apply augmentations to the stacked volume to ensure spatial consistency across the sequence.
        # Input to transform will be (H, W, Sequence*Channels)
        if self.is_train:
            self.transform = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5),
                    A.OneOf(
                        [
                            A.GridDistortion(p=0.5),
                            A.ElasticTransform(p=0.5),
                        ],
                        p=0.3,
                    ),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                ]
            )

    def _prepare_data(self, load_cached_data):
        """
        Generates or loads the list of specific file paths for each subject.
        Implements Multi-Slice Geometric Sampling (Cite solution_lesson_node_00009):
        Selects adjacent slices around the middle to provide volumetric context.
        """
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                cached_df = pd.read_parquet(self.cache_path)
                if len(cached_df) == len(self.metadata_df):
                    return cached_df
                else:
                    print(f"Cache mismatch for {self.split}. Regenerating...")
            except Exception:
                pass

        records = []

        for _, row in self.metadata_df.iterrows():
            sid = row["BraTS21ID"]
            record = {
                "BraTS21ID": sid,
                "MGMT_value": row.get("MGMT_value", -1),
            }

            for mod in Config.SELECTED_MODALITIES:
                col_name = f"{mod.lower()}_path"
                rel_path = row[col_name]
                full_dir_path = os.path.join(Config.INPUT_DIR, rel_path)

                selected_files = []
                if os.path.exists(full_dir_path):
                    files = [f for f in os.listdir(full_dir_path) if f.endswith(".dcm")]
                    if files:
                        try:
                            files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
                        except:
                            files.sort()

                        # Cite solution_lesson_node_00002: Address Middle Slice Fallacy by taking neighbors
                        mid_idx = len(files) // 2

                        # Select 3 slices: mid-1, mid, mid+1
                        # Cite solution_lesson_node_00009: Use small N (3) to avoid signal dilution
                        offsets = np.arange(
                            -(Config.SLICE_DEPTH // 2), (Config.SLICE_DEPTH // 2) + 1
                        )
                        indices = [mid_idx + i for i in offsets]

                        # Clamp indices to valid range
                        indices = [max(0, min(i, len(files) - 1)) for i in indices]

                        selected_files = [
                            os.path.join(full_dir_path, files[i]) for i in indices
                        ]

                # Ensure we have the correct number of files (pad with None if missing)
                while len(selected_files) < Config.SLICE_DEPTH:
                    selected_files.append(None)

                record[f"{mod}_files"] = selected_files

            records.append(record)

        df_processed = pd.DataFrame(records)
        try:
            df_processed.to_parquet(self.cache_path)
        except Exception as e:
            print(f"Warning: Failed to save cache to {self.cache_path}: {e}")

        return df_processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        channel_imgs = []
        # Load channels: FLAIR (3 slices), T1wCE (3 slices), T2w (3 slices)
        for mod in Config.SELECTED_MODALITIES:
            files = row[f"{mod}_files"]

            for file_path in files:
                if (
                    file_path is not None
                    and isinstance(file_path, str)
                    and os.path.exists(file_path)
                ):
                    img = read_dicom(file_path)
                    if (
                        img.shape[0] != Config.IMG_SIZE
                        or img.shape[1] != Config.IMG_SIZE
                    ):
                        try:
                            img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
                        except:
                            img = np.zeros(
                                (Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8
                            )
                else:
                    img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)

                img = img.astype(np.float32)
                if img.max() > 0:
                    img = (img - img.min()) / (img.max() - img.min())

                channel_imgs.append(img)

        # Stack channels: (H, W, 9)
        img_vol = np.stack(channel_imgs, axis=-1)

        # Apply Augmentations
        if self.transform:
            transformed = self.transform(image=img_vol)
            img_vol = transformed["image"]

        # Transpose to (3, H, W) for PyTorch
        img_vol = np.transpose(img_vol, (2, 0, 1))

        final_tensor = torch.from_numpy(img_vol).float()
        target = torch.tensor(row["MGMT_value"], dtype=torch.float32)

        return final_tensor, target


def get_dataloader(
    metadata_df,
    split,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Creates and returns a DataLoader for the BraTSDataset.

    Args:
        metadata_df: DataFrame with metadata.
        split: 'train', 'val', or 'test'.
        batch_size: Batch size.
        num_workers: Number of workers.
        load_cached_data: Whether to use caching.
        debug: If True, subsets the data for quick debugging.
    """
    if debug:
        metadata_df = metadata_df.head(Config.DEBUG_SAMPLE_SIZE)

    dataset = BraTSDataset(metadata_df, split=split, load_cached_data=load_cached_data)

    shuffle = split == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=shuffle and len(dataset) > batch_size,
    )

    return loader
