import os
import re
import glob
import numpy as np
import cv2
import pandas as pd
from library.config import Config

# Attempt to import pydicom, handle if missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


class ImageProcessor:
    def __init__(self, debug=False):
        """
        Initializes the ImageProcessor.

        Args:
            debug (bool): If True, processes only a small subset of data.
        """
        self.debug = debug
        self.img_size = Config.IMG_SIZE
        self.modalities = Config.MODALITIES
        self.cache_dir = Config.CACHE_DIR

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _natural_sort_key(self, s):
        """
        Sorts strings containing numbers naturally (e.g., Image-1, Image-2, Image-10).
        """
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)
        ]

    def load_dicom_volume(self, folder_path):
        """
        Reads a directory of DICOM files and constructs a 3D volume.
        Sorts files by the number in the filename (e.g., Image-10.dcm) to ensure correct Z-ordering.
        Returns a 3D numpy array (Depth, Height, Width).
        """
        if not os.path.exists(folder_path):
            return np.zeros((10, self.img_size, self.img_size), dtype=np.float32)

        # Sort files naturally to maintain anatomical order
        files = sorted(
            glob.glob(os.path.join(folder_path, "*.dcm")),
            key=lambda f: self._natural_sort_key(os.path.basename(f)),
        )

        if not files:
            return np.zeros((10, self.img_size, self.img_size), dtype=np.float32)

        slices = []
        for f in files:
            try:
                if HAS_PYDICOM:
                    ds = pydicom.dcmread(f)
                    img = ds.pixel_array
                else:
                    # Fallback to OpenCV if pydicom is missing
                    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)

                if img is not None:
                    slices.append(img)
            except Exception:
                continue

        if not slices:
            return np.zeros((10, self.img_size, self.img_size), dtype=np.float32)

        volume = np.array(slices)
        return volume

    def get_brain_roi(self, volume):
        """
        Calculates the 3D bounding box of the non-zero content (brain tissue).
        Returns (z_min, z_max, y_min, y_max, x_min, x_max).
        """
        # Threshold to identify brain tissue (simple non-zero check)
        mask = volume > 0
        coords = np.argwhere(mask)

        if coords.size == 0:
            # If volume is empty, return full range
            d, h, w = volume.shape
            return 0, d, 0, h, 0, w

        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        return z_min, z_max + 1, y_min, y_max + 1, x_min, x_max + 1

    def normalize_image(self, image):
        """
        Applies instance-level min-max scaling to [0, 1].
        """
        image = image.astype(np.float32)
        min_val = np.min(image)
        max_val = np.max(image)

        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)

        return image

    def extract_orthogonal_views(self, volume):
        """
        Extracts the middle slice from Axial, Coronal, and Sagittal planes
        based on the geometric center of the brain ROI.
        Returns 3 2D arrays: (axial, coronal, sagittal).
        """
        z_min, z_max, y_min, y_max, x_min, x_max = self.get_brain_roi(volume)

        # Calculate centers
        cz = (z_min + z_max) // 2
        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2

        # Ensure indices are within bounds
        d, h, w = volume.shape
        cz = np.clip(cz, 0, d - 1)
        cy = np.clip(cy, 0, h - 1)
        cx = np.clip(cx, 0, w - 1)

        # Extract slices
        # Axial: (Y, X) at fixed Z
        axial = volume[cz, :, :]

        # Coronal: (Z, X) at fixed Y -> Resize to (H, W)
        # Note: We treat Z as 'height' in the 2D image for coronal/sagittal views
        coronal_raw = volume[:, cy, :]

        # Sagittal: (Z, Y) at fixed X -> Resize to (H, W)
        sagittal_raw = volume[:, :, cx]

        # Resize all to target IMG_SIZE
        # cv2.resize expects (width, height)
        axial_resized = cv2.resize(
            axial.astype(np.float32), (self.img_size, self.img_size)
        )
        coronal_resized = cv2.resize(
            coronal_raw.astype(np.float32), (self.img_size, self.img_size)
        )
        sagittal_resized = cv2.resize(
            sagittal_raw.astype(np.float32), (self.img_size, self.img_size)
        )

        return axial_resized, coronal_resized, sagittal_resized

    def process_dataset(self, df, split_name, load_cached_data=True):
        """
        Main function to process a dataset (train/val/test).
        Generates or loads 3 views for every subject.

        Args:
            df (pd.DataFrame): Metadata dataframe.
            split_name (str): 'train', 'val', or 'test' (used for cache naming).
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: {
                'ids': np.array,
                'axial': np.array (N, H, W, 3),
                'coronal': np.array (N, H, W, 3),
                'sagittal': np.array (N, H, W, 3),
                'targets': np.array (N,) (if available)
            }
        """
        # Define cache paths
        cache_ids = os.path.join(self.cache_dir, f"{split_name}_ids.npy")
        cache_ax = os.path.join(self.cache_dir, f"{split_name}_axial.npy")
        cache_cor = os.path.join(self.cache_dir, f"{split_name}_coronal.npy")
        cache_sag = os.path.join(self.cache_dir, f"{split_name}_sagittal.npy")
        cache_tgt = os.path.join(self.cache_dir, f"{split_name}_targets.npy")

        has_targets = "MGMT_value" in df.columns

        # Try loading cache
        if load_cached_data:
            try:
                if os.path.exists(cache_ids) and os.path.exists(cache_ax):
                    print(f"Loading cached data for {split_name}...")
                    ids = np.load(cache_ids)
                    X_ax = np.load(cache_ax)
                    X_cor = np.load(cache_cor)
                    X_sag = np.load(cache_sag)

                    targets = None
                    if has_targets and os.path.exists(cache_tgt):
                        targets = np.load(cache_tgt)
                    elif has_targets:
                        # Cache inconsistency, recompute
                        raise FileNotFoundError

                    return {
                        "ids": ids,
                        "axial": X_ax,
                        "coronal": X_cor,
                        "sagittal": X_sag,
                        "targets": targets,
                    }
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing...")

        # Recompute
        print(f"Processing {len(df)} subjects for {split_name}...")

        if self.debug:
            df = df.iloc[: Config.DEBUG_SAMPLE_SIZE]
            print(f"DEBUG MODE: Reduced dataset to {len(df)}")

        ids_list = []
        X_ax_list = []
        X_cor_list = []
        X_sag_list = []
        y_list = []

        for idx, row in df.iterrows():
            subject_id = row["BraTS21ID"]

            # Prepare channels for this subject
            # Shape: (H, W, 3) for each view
            subj_ax = np.zeros(
                (self.img_size, self.img_size, len(self.modalities)), dtype=np.float32
            )
            subj_cor = np.zeros(
                (self.img_size, self.img_size, len(self.modalities)), dtype=np.float32
            )
            subj_sag = np.zeros(
                (self.img_size, self.img_size, len(self.modalities)), dtype=np.float32
            )

            # Iterate over modalities (FLAIR, T1wCE, T2w)
            for ch_idx, mod in enumerate(self.modalities):
                # Construct path
                # Metadata contains relative path, e.g., "train/00000/FLAIR"
                # Column names in metadata are lowercase: "flair_path", "t1wce_path", etc.
                rel_path = row[f"{mod.lower()}_path"]
                full_path = os.path.join(Config.INPUT_DIR, rel_path)

                # Load Volume
                vol = self.load_dicom_volume(full_path)

                # Extract Orthogonal Views (ROI Centered)
                ax, cor, sag = self.extract_orthogonal_views(vol)

                # Normalize and assign to channel
                subj_ax[:, :, ch_idx] = self.normalize_image(ax)
                subj_cor[:, :, ch_idx] = self.normalize_image(cor)
                subj_sag[:, :, ch_idx] = self.normalize_image(sag)

            ids_list.append(subject_id)
            X_ax_list.append(subj_ax)
            X_cor_list.append(subj_cor)
            X_sag_list.append(subj_sag)

            if has_targets:
                y_list.append(row["MGMT_value"])

        # Convert to numpy
        ids = np.array(ids_list)
        X_ax = np.array(X_ax_list, dtype=np.float32)
        X_cor = np.array(X_cor_list, dtype=np.float32)
        X_sag = np.array(X_sag_list, dtype=np.float32)

        targets = np.array(y_list, dtype=np.float32) if has_targets else None

        # Save to cache
        print(f"Saving cache to {self.cache_dir}...")
        np.save(cache_ids, ids)
        np.save(cache_ax, X_ax)
        np.save(cache_cor, X_cor)
        np.save(cache_sag, X_sag)
        if has_targets:
            np.save(cache_tgt, targets)

        return {
            "ids": ids,
            "axial": X_ax,
            "coronal": X_cor,
            "sagittal": X_sag,
            "targets": targets,
        }
