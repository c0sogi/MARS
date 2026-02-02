import os
import glob
import numpy as np
import cv2
import torch
from library.config import Config


class DicomHandler:
    """
    Handles loading, selecting, and preprocessing of DICOM CT scans.
    Includes a fallback raw binary reader for environments without pydicom.
    """

    @staticmethod
    def read_dicom_raw(path):
        """
        Reads a DICOM file. Attempts to use pydicom if available.
        Falls back to raw binary reading assuming 512x512 int16 uncompressed data.
        """
        # 1. Try pydicom (Standard way)
        try:
            import pydicom

            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)

            # Apply rescale if available to get HU
            if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
                slope = float(dcm.RescaleSlope)
                intercept = float(dcm.RescaleIntercept)
                img = img * slope + intercept
            return img
        except ImportError:
            pass
        except Exception:
            pass

        # 2. Fallback: Raw binary read
        # CT scans in this dataset are typically 512x512 int16
        img_dim = 512
        expected_bytes = img_dim * img_dim * 2

        try:
            with open(path, "rb") as f:
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()

                if file_size < expected_bytes:
                    return np.zeros((img_dim, img_dim), dtype=np.float32)

                f.seek(file_size - expected_bytes)
                buffer = f.read(expected_bytes)

            img = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
            img = img.reshape((img_dim, img_dim))

            # Heuristic for HU correction:
            # Air is ~ -1000 HU. If raw min is > -100 (e.g. 0), it's likely shifted by +1024.
            if img.min() > -100:
                img -= 1024

            return img
        except Exception:
            # Return zero array on failure
            return np.zeros((512, 512), dtype=np.float32)

    @staticmethod
    def load_scan(patient_dir):
        """
        Loads all DICOM slices for a patient, sorts them by instance number.
        Returns a list of 2D numpy arrays.
        """
        files = glob.glob(os.path.join(patient_dir, "*.dcm"))

        # Sort by file number (filename is usually '1.dcm', '10.dcm')
        def get_file_num(fname):
            base = os.path.basename(fname)
            name, _ = os.path.splitext(base)
            try:
                return int(name)
            except ValueError:
                return 0

        files.sort(key=get_file_num)

        slices = []
        for f in files:
            img = DicomHandler.read_dicom_raw(f)
            slices.append(img)

        if not slices:
            return [np.zeros((512, 512), dtype=np.float32)]

        return slices

    @staticmethod
    def select_variance_slices(slices, n_slices=Config.N_SLICES):
        """
        Selects the top N slices with the highest pixel variance in the center crop.
        High variance typically indicates lung tissue/fibrosis vs empty space.
        """
        if len(slices) <= n_slices:
            return slices

        variances = []
        h, w = slices[0].shape

        # Center crop parameters (50% of image)
        crop_size = min(h, w) // 2
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2

        for i, s in enumerate(slices):
            crop = s[start_h : start_h + crop_size, start_w : start_w + crop_size]
            var = np.var(crop)
            variances.append((var, i))

        # Sort by variance descending
        variances.sort(key=lambda x: x[0], reverse=True)

        # Pick top indices and resort them spatially
        top_indices = [x[1] for x in variances[:n_slices]]
        top_indices.sort()

        selected_slices = [slices[i] for i in top_indices]
        return selected_slices

    @staticmethod
    def preprocess_slice(slice_arr, img_size=Config.IMG_SIZE):
        """
        Preprocesses a single slice:
        1. Windowing (Lung Window: -1000 to 400 HU)
        2. Normalization [0, 1]
        3. Resize
        4. Channel Replication (1 -> 3)
        5. ImageNet Normalization
        """
        # 1. Windowing
        min_hu = -1000
        max_hu = 400
        img = np.clip(slice_arr, min_hu, max_hu)

        # 2. Normalize to [0, 1]
        img = (img - min_hu) / (max_hu - min_hu)

        # 3. Resize
        img_resized = cv2.resize(
            img, (img_size, img_size), interpolation=cv2.INTER_LINEAR
        )

        # 4. Channel Replication (H, W) -> (H, W, 3)
        img_rgb = np.stack([img_resized] * 3, axis=-1)

        # 5. Normalize with ImageNet Mean/Std
        mean = np.array(Config.IMAGENET_MEAN, dtype=np.float32)
        std = np.array(Config.IMAGENET_STD, dtype=np.float32)
        img_norm = (img_rgb - mean) / std

        # 6. Transpose to (C, H, W) for PyTorch
        img_tensor = np.transpose(img_norm, (2, 0, 1))

        return img_tensor

    @staticmethod
    def process_patient(patient_id, subset="train", load_cached_data=True):
        """
        Orchestrates the loading and processing pipeline with caching.

        Args:
            patient_id (str): Patient ID.
            subset (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: Processed tensor of shape (N_SLICES, 3, H, W).
        """
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Corrupt file, proceed to recompute

        # 2. Determine source directory
        if subset in ["train", "val"]:
            base_dir = Config.TRAIN_DCM_DIR
        else:
            base_dir = Config.TEST_DCM_DIR

        patient_dir = os.path.join(base_dir, patient_id)

        # Handle missing directories (possible in test set structure)
        if not os.path.exists(patient_dir):
            return np.zeros(
                (Config.N_SLICES, 3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        # 3. Load and Select Slices
        slices = DicomHandler.load_scan(patient_dir)
        selected = DicomHandler.select_variance_slices(slices, Config.N_SLICES)

        # Pad if insufficient slices
        while len(selected) < Config.N_SLICES:
            selected.append(np.zeros_like(selected[0]))

        # 4. Preprocess
        processed_stack = []
        for s in selected:
            p = DicomHandler.preprocess_slice(s, Config.IMG_SIZE)
            processed_stack.append(p)

        # Stack -> (N_SLICES, 3, H, W)
        result = np.stack(processed_stack, axis=0).astype(np.float32)

        # 5. Save to cache
        try:
            np.save(cache_path, result)
        except Exception:
            pass

        return result
