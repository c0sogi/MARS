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
        self.cache_path = os.path.join(
            self.cache_dir, f"cached_file_lists_{split}.parquet"
        )

        # Load or generate file lists (Geometric Volumetric Sampling)
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
        Implements Geometric Volumetric Sampling:
        1. Sort files by instance number.
        2. Discard top/bottom 15%.
        3. Uniformly sample N slices.
        """
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                # Load cached dataframe
                cached_df = pd.read_parquet(self.cache_path)

                # Cite debug_lesson_1: Verify Cache Consistency Before Loading
                if len(cached_df) == len(self.metadata_df):
                    return cached_df
                else:
                    print(
                        f"Cache mismatch for {self.split}: Expected {len(self.metadata_df)} samples, found {len(cached_df)}. Regenerating..."
                    )
            except Exception:
                pass  # Regenerate if load fails

        records = []

        for _, row in self.metadata_df.iterrows():
            sid = row["BraTS21ID"]
            record = {
                "BraTS21ID": sid,
                "MGMT_value": row.get("MGMT_value", -1),  # Default to -1 for test set
            }

            # Process each modality
            for mod in Config.SELECTED_MODALITIES:
                # Metadata columns are like 'flair_path', 't1wce_path'
                col_name = f"{mod.lower()}_path"
                rel_path = row[col_name]
                full_dir_path = os.path.join(Config.INPUT_DIR, rel_path)

                selected_files = []
                if os.path.exists(full_dir_path):
                    # List all DICOMs
                    files = [f for f in os.listdir(full_dir_path) if f.endswith(".dcm")]

                    if files:
                        # Sort by instance number (Image-X.dcm)
                        # Extract number: Image-123.dcm -> 123
                        try:
                            files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
                        except:
                            files.sort()  # Fallback lexical sort

                        # Geometric Sampling Logic
                        n_files = len(files)
                        start_idx = int(n_files * Config.SLICE_DISCARD_PCT)
                        end_idx = int(n_files * (1 - Config.SLICE_DISCARD_PCT))

                        # Safety check for very small volumes
                        if end_idx <= start_idx:
                            start_idx = 0
                            end_idx = n_files

                        # Uniform sampling of N indices
                        # linspace returns evenly spaced numbers over a specified interval
                        indices = np.linspace(
                            start_idx, end_idx - 1, Config.NUM_SLICES
                        ).astype(int)
                        selected_files = [
                            os.path.join(full_dir_path, files[i]) for i in indices
                        ]

                # Handle missing/empty directories by filling with None
                if len(selected_files) < Config.NUM_SLICES:
                    # Pad with None
                    selected_files = selected_files + [None] * (
                        Config.NUM_SLICES - len(selected_files)
                    )

                record[f"{mod}_files"] = selected_files

            records.append(record)

        df_processed = pd.DataFrame(records)

        # Save to cache for future runs
        # Parquet supports columns containing lists
        try:
            df_processed.to_parquet(self.cache_path)
        except Exception as e:
            print(f"Warning: Failed to save cache to {self.cache_path}: {e}")

        return df_processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        slices = []

        # Iterate over sequence step (0 to N-1)
        for seq_idx in range(Config.NUM_SLICES):
            # For each time step, we have C channels (FLAIR, T1wCE, T2w)
            channel_imgs = []
            for mod in Config.SELECTED_MODALITIES:
                file_path = row[f"{mod}_files"][seq_idx]

                if file_path is not None and os.path.exists(file_path):
                    img = read_dicom(file_path)

                    # Ensure correct size if reading raw
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

                # Normalize to [0, 1] per slice
                img = img.astype(np.float32)
                if img.max() > 0:
                    img = (img - img.min()) / (img.max() - img.min())

                channel_imgs.append(img)

            # Stack channels for this time step: (H, W, C)
            slice_vol = np.stack(channel_imgs, axis=-1)
            slices.append(slice_vol)

        # Stack all slices along channel dimension for spatial augmentation
        # Result: (H, W, Sequence * Channels)
        # This ensures that if we rotate the image, the entire 3D volume rotates coherently
        combined_vol = np.concatenate(slices, axis=-1)

        # Apply Augmentations
        if self.transform:
            transformed = self.transform(image=combined_vol)
            combined_vol = transformed["image"]

        # Reshape back to (Channels, Height, Width) for PyTorch
        # Current shape: (H, W, S*C)
        # Transpose to (S*C, H, W) where S*C is the total channel depth
        combined_vol = np.transpose(combined_vol, (2, 0, 1))

        # Convert to torch tensor: (Total_Channels, H, W)
        final_tensor = torch.from_numpy(combined_vol).float()

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
