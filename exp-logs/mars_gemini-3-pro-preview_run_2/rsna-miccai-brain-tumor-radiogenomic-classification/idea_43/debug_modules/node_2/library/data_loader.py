import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from library.config import Config
from library.utils import load_dicom_image, resize_image, normalize_image


class BraTSDataset(Dataset):
    """
    Dataset class for BraTS21 data.
    Wraps pre-processed numpy arrays and applies geometric augmentations during training.
    """

    def __init__(self, data, labels, is_train=False):
        self.data = data
        self.labels = labels
        self.is_train = is_train

        # Define augmentations for training
        if self.is_train:
            self.transform = A.Compose(
                [
                    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ]
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # data is (C, H, W) -> (12, 224, 224)
        image = self.data[idx]
        label = self.labels[idx]

        if self.is_train and self.transform:
            # Albumentations expects HWC format
            image_hwc = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image_hwc)["image"]
            image = np.transpose(augmented, (2, 0, 1))

        return torch.tensor(image, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )


def get_file_id(filename):
    """Parses the integer ID from a DICOM filename (e.g., 'Image-123.dcm' -> 123)."""
    try:
        return int(filename.split("-")[1].split(".")[0])
    except:
        return -1


def process_subject(row, input_dir):
    """
    Processes a single subject to create a 12-channel input tensor.
    Implements Modality-Adaptive Stride and Fidelity-Aligned ROI Selection.
    """
    # 1. Identify Anchor from FLAIR (Fidelity-Aligned ROI Selection)
    flair_path = os.path.join(input_dir, row["path_FLAIR"])
    if not os.path.exists(flair_path):
        return None

    flair_files = [f for f in os.listdir(flair_path) if f.endswith(".dcm")]
    if not flair_files:
        return None

    # Map ID -> Filename
    flair_map = {}
    for f in flair_files:
        fid = get_file_id(f)
        if fid != -1:
            flair_map[fid] = f

    sorted_ids = sorted(flair_map.keys())
    if not sorted_ids:
        return None

    # Restrict search to 15%-85% depth
    n_files = len(sorted_ids)
    start_idx = int(n_files * Config.ANCHOR_RANGE[0])
    end_idx = int(n_files * Config.ANCHOR_RANGE[1])

    if start_idx >= end_idx:
        search_ids = sorted_ids  # Fallback to all if too few slices
    else:
        search_ids = sorted_ids[start_idx:end_idx]

    # Find Anchor based on Max Intensity (Raw Pixel Sum)
    max_intensity = -1
    anchor_id = -1

    for fid in search_ids:
        fpath = os.path.join(flair_path, flair_map[fid])
        img = load_dicom_image(fpath)
        if img is None:
            continue

        # Calculate integral on raw pixels (no smoothing)
        current_intensity = np.sum(img)
        if current_intensity > max_intensity:
            max_intensity = current_intensity
            anchor_id = fid

    if anchor_id == -1:
        # Fallback: middle of sorted_ids
        anchor_id = sorted_ids[len(sorted_ids) // 2]

    # 2. Load Modalities with Adaptive Stride
    channels = []

    # Order: FLAIR, T2w, T1w, T1wCE
    for mod in Config.MODALITIES:
        mod_path = os.path.join(input_dir, row[f"path_{mod}"])
        stride = Config.STRIDES[mod]
        # Target IDs: [Anchor-Stride, Anchor, Anchor+Stride]
        target_ids = [anchor_id - stride, anchor_id, anchor_id + stride]

        # Scan available files for this modality
        mod_files = []
        if os.path.exists(mod_path):
            mod_files = [f for f in os.listdir(mod_path) if f.endswith(".dcm")]

        mod_map = {}
        for f in mod_files:
            fid = get_file_id(f)
            if fid != -1:
                mod_map[fid] = f

        mod_ids = sorted(mod_map.keys())

        if not mod_ids:
            # Empty modality folder -> 3 blank channels (Spectral Padding)
            for _ in range(3):
                channels.append(
                    np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
                )
            continue

        min_id = mod_ids[0]
        max_id = mod_ids[-1]

        for tid in target_ids:
            load_id = tid
            is_out_of_bounds = (tid < min_id) or (tid > max_id)
            final_img = None

            if is_out_of_bounds:
                # Spatial Clamping: Replicate nearest valid slice
                if tid < min_id:
                    load_id = min_id
                else:
                    load_id = max_id

                if load_id in mod_map:
                    fpath = os.path.join(mod_path, mod_map[load_id])
                    img = load_dicom_image(fpath)
                    if img is not None:
                        final_img = img
            else:
                # In bounds
                if tid in mod_map:
                    fpath = os.path.join(mod_path, mod_map[tid])
                    img = load_dicom_image(fpath)
                    if img is not None:
                        final_img = img
                else:
                    # Missing (gap) -> Zero Padding (Spectral Padding)
                    pass

            # Preprocessing
            if final_img is not None:
                final_img = resize_image(final_img, Config.IMG_SIZE)
                final_img = normalize_image(final_img)
            else:
                final_img = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
                )

            channels.append(final_img)

    # Stack to (12, 224, 224)
    volume = np.stack(channels, axis=0)
    return volume


def get_dataloader(
    split="train", batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=False
):
    """
    Creates a DataLoader for the specified split.
    Handles caching of processed numpy arrays to speed up subsequent runs.
    """

    # Define cache paths
    cache_data_path = os.path.join(Config.CACHE_DIR, f"{split}_data.npy")
    cache_labels_path = os.path.join(Config.CACHE_DIR, f"{split}_labels.npy")

    data = None
    labels = None

    # 1. Attempt to load from cache
    if (
        load_cached_data
        and os.path.exists(cache_data_path)
        and os.path.exists(cache_labels_path)
    ):
        print(f"Loading cached data for {split} from {Config.CACHE_DIR}...")
        try:
            data = np.load(cache_data_path)
            labels = np.load(cache_labels_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")
            data = None

    # 2. Compute data from scratch if needed
    if data is None:
        print(f"Processing data for {split}...")

        # Load Metadata
        if split == "train":
            df = pd.read_csv(Config.TRAIN_METADATA)
        elif split == "val":
            df = pd.read_csv(Config.VAL_METADATA)
        elif split == "test":
            df = pd.read_csv(Config.TEST_METADATA)
        else:
            raise ValueError(f"Unknown split: {split}")

        if debug:
            df = df.head(Config.DEBUG_SIZE)

        data_list = []
        label_list = []
        failed_count = 0

        for idx, row in df.iterrows():
            vol = process_subject(row, Config.INPUT_DIR)

            if vol is None:
                failed_count += 1
                # For test set, we must provide a prediction, so we substitute a zero volume.
                # For train/val, we skip corrupt samples to maintain training stability.
                if split == "test":
                    vol = np.zeros(
                        (Config.NUM_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE),
                        dtype=np.float32,
                    )
                else:
                    continue

            data_list.append(vol)

            if "MGMT_value" in row:
                label_list.append(row["MGMT_value"])
            else:
                label_list.append(0.5)  # Placeholder for test set

        # Circuit Breaker: Abort if too many failures
        total_subjects = len(df)
        if total_subjects > 0:
            failure_rate = failed_count / total_subjects
            if failure_rate > 0.01 and split != "test":
                raise RuntimeError(
                    f"Data Corruption Error: {failure_rate*100:.2f}% of subjects failed to load (Threshold: 1%). Aborting."
                )

        data = np.array(data_list, dtype=np.float32)
        labels = np.array(label_list, dtype=np.float32)

        # Save to cache
        if not debug:
            np.save(cache_data_path, data)
            np.save(cache_labels_path, labels)
            print(f"Saved cache to {Config.CACHE_DIR}")

    # 3. Create Dataset and DataLoader
    is_train = split == "train"
    dataset = BraTSDataset(data, labels, is_train=is_train)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return loader
