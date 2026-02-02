import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from library.config import Config
from library.utils import load_dicom
from library.transforms import get_transforms


class CervicalSpineDataset(Dataset):
    def __init__(
        self,
        mode="train",
        transform=None,
        load_cached_data=True,
        tta_offset=0.0,
        seq_length=64,
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Augmentation pipeline. If None, uses default from library.
            load_cached_data (bool): If True, attempts to load sorted paths from disk.
            tta_offset (float): Offset fraction for sampling (e.g., -0.3, 0.0, 0.3) for TTA.
            seq_length (int): Number of slices to sample per volume.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.tta_offset = tta_offset
        self.seq_length = seq_length

        # Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_METADATA_PATH)
            self.image_dir = Config.TRAIN_IMAGES_DIR
            self.transform = transform or get_transforms("train")
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_METADATA_PATH)
            self.image_dir = Config.TRAIN_IMAGES_DIR
            self.transform = transform or get_transforms("valid")
        elif mode == "test":
            self.df = pd.read_csv(Config.TEST_METADATA_PATH)
            self.image_dir = Config.TEST_IMAGES_DIR
            self.transform = transform or get_transforms("test")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Debug Mode: Slice dataset if configured
        if Config.DEBUG_DATA_SIZE is not None:
            self.df = self.df.iloc[: Config.DEBUG_DATA_SIZE].reset_index(drop=True)
            print(
                f"[{mode.upper()}] Debug mode: Reduced dataset to {len(self.df)} samples."
            )

        # Cache Logic for Sorted File Paths
        self.cache_path = os.path.join(Config.CACHE_DIR, f"sorted_paths_{mode}.npy")
        self.sorted_paths_map = self._load_or_generate_cache()

    def _load_or_generate_cache(self):
        """
        Loads the dictionary of sorted file paths from cache or generates it.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        if self.load_cached_data and os.path.exists(self.cache_path):
            print(
                f"[{self.mode.upper()}] Loading sorted paths from cache: {self.cache_path}"
            )
            try:
                return np.load(self.cache_path, allow_pickle=True).item()
            except Exception as e:
                print(f"[{self.mode.upper()}] Cache load failed ({e}). Regenerating...")

        print(f"[{self.mode.upper()}] Generating sorted file paths cache...")

        # Get unique studies and their paths
        study_uids = self.df["StudyInstanceUID"].unique()

        # Prepare arguments for parallel processing
        # We need the full path to the study directory
        study_dirs = []
        for uid in study_uids:
            # Metadata image_path is relative, e.g., "train_images/UID"
            # We need to construct the full path based on the input dir structure
            # The metadata generator script put relative paths like "train_images/..."
            # Config.INPUT_DIR is "./input".

            # Find the row for this UID to get the relative path
            # Optimized: Assume metadata 'image_path' column is correct relative to input dir
            rel_path = self.df[self.df["StudyInstanceUID"] == uid]["image_path"].iloc[0]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            study_dirs.append((uid, full_path))

        # Parallel Sort
        results = {}
        with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
            # Map returns iterator, convert to list to trigger execution
            futures = [
                executor.submit(self._process_study_files, uid, path)
                for uid, path in study_dirs
            ]

            for future in tqdm(
                futures, total=len(study_dirs), desc=f"Sorting {self.mode}"
            ):
                uid, sorted_files = future.result()
                if sorted_files:
                    results[uid] = sorted_files

        # Save to cache
        np.save(self.cache_path, results)
        print(f"[{self.mode.upper()}] Cache saved to {self.cache_path}")
        return results

    @staticmethod
    def _process_study_files(uid, directory):
        """
        Helper to list and sort DICOM files by Z-position.
        Returns (uid, list_of_filenames).
        """
        if not os.path.exists(directory):
            return uid, []

        files = glob.glob(os.path.join(directory, "*.dcm"))
        if not files:
            return uid, []

        # Read Z-position for sorting
        # We store tuples (z_pos, filename)
        file_data = []

        for f_path in files:
            try:
                # Read only specific tag to be fast
                ds = pydicom.dcmread(f_path, stop_before_pixels=True)
                # ImagePositionPatient is tag (0020, 0032). Z is index 2.
                z_pos = float(ds.ImagePositionPatient[2])
                file_data.append((z_pos, os.path.basename(f_path)))
            except Exception:
                # Fallback: try sorting by filename number if DICOM header fails
                # (Though competition data usually has headers)
                try:
                    # e.g. "100.dcm" -> 100
                    fname = os.path.basename(f_path)
                    num = int(os.path.splitext(fname)[0])
                    file_data.append((num, fname))
                except:
                    pass

        # Sort by Z position (or fallback number)
        file_data.sort(key=lambda x: x[0])

        # Return only filenames
        sorted_filenames = [x[1] for x in file_data]
        return uid, sorted_filenames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row["StudyInstanceUID"]

        # Retrieve sorted filenames
        filenames = self.sorted_paths_map.get(uid, [])
        num_files = len(filenames)

        # Handle empty or missing studies safely
        if num_files == 0:
            # Return zero tensor if data missing (should not happen with valid metadata)
            # Shape: (Seq, 3, H, W)
            return torch.zeros(
                (self.seq_length, 3, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])
            ), torch.zeros(Config.NUM_CLASSES)

        # --- Sampling Logic ---
        indices = []
        stride = num_files / self.seq_length

        if self.mode == "train":
            # Dynamic Z-Jitter
            for i in range(self.seq_length):
                # Center of the bin
                center = i * stride + (stride / 2)
                # Add random jitter within the bin [-0.5*stride, 0.5*stride]
                jitter = np.random.uniform(-0.5 * stride, 0.5 * stride)
                sample_idx = int(center + jitter)
                indices.append(sample_idx)
        else:
            # Deterministic Sampling (with optional TTA offset)
            # tta_offset is fraction of stride, e.g., 0.0 is center
            for i in range(self.seq_length):
                center = (i + 0.5 + self.tta_offset) * stride
                indices.append(int(center))

        # Clamp indices to valid range
        indices = np.clip(indices, 0, num_files - 1)

        # --- 2.5D Stack Loading ---
        # We need to construct the full path to the image directory
        # Metadata 'image_path' is relative to input dir
        rel_dir = row["image_path"]
        full_dir = os.path.join(Config.INPUT_DIR, rel_dir)

        images_list = []

        for center_idx in indices:
            # 2.5D: Load [z-1, z, z+1]
            # Clamp neighbors
            neighbors = [center_idx - 1, center_idx, center_idx + 1]
            neighbors = np.clip(neighbors, 0, num_files - 1).astype(int)

            slice_channels = []
            for n_idx in neighbors:
                fname = filenames[n_idx]
                fpath = os.path.join(full_dir, fname)

                try:
                    # Load and preprocess (windowing, resizing)
                    # load_dicom returns (H, W) uint8
                    img = load_dicom(
                        fpath,
                        size=Config.IMAGE_SIZE,
                        window_level=Config.WINDOW_LEVEL,
                        window_width=Config.WINDOW_WIDTH,
                    )
                except Exception:
                    # Fallback for corrupt file: black image
                    img = np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)

                slice_channels.append(img)

            # Stack to (H, W, 3)
            # Note: load_dicom returns (H, W), stack adds channel dim
            stacked_img = np.stack(slice_channels, axis=-1)  # (H, W, 3)
            images_list.append(stacked_img)

        # Convert to numpy array: (Seq, H, W, 3)
        volume = np.array(images_list, dtype=np.uint8)

        # --- Augmentation ---
        # VolumetricReplayWrapper expects (Seq, H, W, C)
        # Returns Tensor (Seq, C, H, W)
        if self.transform:
            volume_tensor = self.transform(volume)
        else:
            # Fallback to simple tensor conversion
            volume_tensor = torch.from_numpy(volume).permute(0, 3, 1, 2).float() / 255.0

        # --- Labels ---
        if self.mode == "test":
            # Dummy labels for test set
            labels = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
        else:
            # Extract targets
            # Columns: C1, C2, C3, C4, C5, C6, C7, patient_overall
            target_vals = row[Config.TARGET_COLS].values.astype(np.float32)
            labels = torch.tensor(target_vals, dtype=torch.float32)

        return volume_tensor, labels
