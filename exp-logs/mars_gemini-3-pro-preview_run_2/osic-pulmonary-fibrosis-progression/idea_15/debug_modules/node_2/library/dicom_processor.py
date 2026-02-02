import os
import glob
import numpy as np
import cv2
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    IMAGE_SIZE,
    SLICES_PER_AXIS,
    HU_MIN,
    HU_MAX,
)


class DicomProcessor:
    """
    Handles DICOM loading, preprocessing, Multi-Axis Variance Sampling,
    and Volumetric Histogram computation.
    """

    def __init__(self):
        # Check for pydicom availability
        try:
            import pydicom

            self.has_pydicom = True
        except ImportError:
            self.has_pydicom = False

    def process_patient(self, patient_id, dcm_rel_path, load_cached_data=True):
        """
        Main pipeline to process a patient's CT scan.

        Args:
            patient_id (str): Unique patient identifier.
            dcm_rel_path (str): Relative path to the DICOM directory (e.g., 'train/ID...').
            load_cached_data (bool): If True, attempts to load from disk cache.

        Returns:
            tuple: (images, histogram)
                images (np.ndarray): Shape (2*SLICES_PER_AXIS, IMAGE_SIZE, IMAGE_SIZE), float32 normalized.
                histogram (np.ndarray): Shape (4,), float32 density distribution.
        """
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, f"{patient_id}.npz")

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                return data["images"], data["histogram"]
            except Exception:
                # If cache is corrupt or unreadable, proceed to process from scratch
                pass

        # 2. Process Data from Scratch
        full_path = os.path.join(INPUT_DIR, dcm_rel_path)

        # Load and convert to HU
        volume = self.load_scan(full_path)

        if volume is None or volume.size == 0:
            # Handle missing data or errors gracefully
            images = np.zeros(
                (2 * SLICES_PER_AXIS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
            )
            histogram = np.zeros(4, dtype=np.float32)
        else:
            # Clean HU values
            volume = self.get_pixels_hu(volume)

            # Generate Multi-Axis Images (Axial + Coronal)
            images = self.get_multi_axis_slices(volume)

            # Generate Volumetric Histogram
            # We create a broad mask to isolate body tissue/lung from background air
            mask = self.segment_lung_mask(volume)
            histogram = self.get_density_histogram(volume, mask)

        # 3. Save to Cache
        np.savez(cache_path, images=images, histogram=histogram)

        return images, histogram

    def load_scan(self, path):
        """
        Loads DICOM files from a directory into a 3D numpy array (Z, Y, X).
        Handles sorting by instance number and conversion to Hounsfield Units.
        """
        if not os.path.exists(path):
            return None

        files = glob.glob(os.path.join(path, "*.dcm"))
        if not files:
            return None

        # Sort files by Instance Number (derived from filename)
        # Filenames are typically '1.dcm', '10.dcm', etc.
        try:
            files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except ValueError:
            files.sort()

        slices = []

        # Strategy A: Use pydicom if available (Preferred)
        if self.has_pydicom:
            import pydicom

            try:
                scans = [pydicom.dcmread(f) for f in files]
                # Sort by ImagePositionPatient Z if available, else retain filename sort
                # (Skipping complex sort for robustness, filename sort is usually sufficient here)

                for s in scans:
                    img = s.pixel_array.astype(np.float32)
                    slope = getattr(s, "RescaleSlope", 1)
                    intercept = getattr(s, "RescaleIntercept", -1024)
                    img = img * slope + intercept
                    slices.append(img)
            except Exception:
                slices = []  # Fallback to Strategy B if any error occurs

        # Strategy B: Use OpenCV if pydicom is missing or fails
        if not slices:
            for f in files:
                # cv2.IMREAD_UNCHANGED (-1) loads the raw depth info (usually uint16)
                img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    img = img.astype(np.float32)
                    # Heuristic: Standard DICOM pixel values are often shifted by +1024
                    # so that air (-1000 HU) becomes 24.
                    # HU = Pixel - 1024
                    img = img - 1024
                    slices.append(img)

        if not slices:
            return None

        return np.stack(slices)

    def get_pixels_hu(self, volume):
        """
        Cleans and clips the Hounsfield Unit volume to remove artifacts.
        """
        # Clip to a reasonable physical range to remove metal artifacts or scanning errors
        # Air is -1000, Bone is ~400-1000. We allow a bit more range.
        return np.clip(volume, -2000, 2000)

    def segment_lung_mask(self, volume):
        """
        Creates a binary mask isolating the lung/body area.
        Uses a broad threshold to exclude background air and high-density bone.
        """
        # Background air is usually -1000. We threshold slightly above that.
        # We also cap at a high value to exclude dense bone if desired,
        # but for histogram we want to capture consolidation (> -400).
        # Range: > -1000 (Air) and < 1000 (Bone/Metal)
        return (volume > -1000) & (volume < 1000)

    def get_multi_axis_slices(self, volume):
        """
        Selects slices with highest variance from Axial and Coronal planes.
        Returns: (2*SLICES_PER_AXIS, IMAGE_SIZE, IMAGE_SIZE)
        """
        # 1. Axial Selection (Z-axis)
        # volume shape: (Z, Y, X)
        axial_slices = self._select_slices_from_axis(volume)

        # 2. Coronal Selection (Y-axis)
        # Transpose to (Y, Z, X)
        coronal_volume = np.transpose(volume, (1, 0, 2))
        coronal_slices = self._select_slices_from_axis(coronal_volume)

        # Concatenate along the slice dimension
        return np.concatenate([axial_slices, coronal_slices], axis=0)

    def _select_slices_from_axis(self, volume):
        """
        Helper to calculate variance, select top slices, and resize.
        """
        # Calculate variance across the spatial dimensions (H, W) for each slice
        # This captures the amount of texture/information in the slice
        variances = np.var(volume, axis=(1, 2))

        n_slices = volume.shape[0]
        if n_slices < SLICES_PER_AXIS:
            # If volume is too shallow, take all indices
            indices = np.arange(n_slices)
        else:
            # Select indices of top N variances
            # argsort is ascending, so we take the tail
            top_indices = np.argsort(variances)[-SLICES_PER_AXIS:]
            # Sort indices to maintain anatomical order (Top-to-Bottom or Front-to-Back)
            indices = np.sort(top_indices)

        selected_slices = []
        for idx in indices:
            img = volume[idx]

            # Normalization for CNN Input (Lung Window)
            # We clip to the specific lung window defined in config (e.g. -1000 to -400)
            # This contrasts the lung tissue against air.
            img_windowed = np.clip(img, HU_MIN, HU_MAX)

            # Min-Max Scale to [0, 1]
            denom = HU_MAX - HU_MIN
            if denom != 0:
                img_norm = (img_windowed - HU_MIN) / denom
            else:
                img_norm = np.zeros_like(img_windowed)

            # Resize to target size for EfficientNet
            img_resized = cv2.resize(img_norm, (IMAGE_SIZE, IMAGE_SIZE))
            selected_slices.append(img_resized)

        # Pad with zeros if we didn't have enough slices
        while len(selected_slices) < SLICES_PER_AXIS:
            selected_slices.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))

        return np.array(selected_slices, dtype=np.float32)

    def get_density_histogram(self, volume, mask):
        """
        Computes a 4-bin density histogram on the masked volume.
        Bins:
            1. Emphysema:       < -950 HU
            2. Healthy:         [-950, -700) HU
            3. Fibrosis:        [-700, -400) HU
            4. Consolidation:   >= -400 HU
        """
        # Apply mask and flatten
        pixels = volume[mask]

        if pixels.size == 0:
            return np.zeros(4, dtype=np.float32)

        # Define thresholds
        t1, t2, t3 = -950, -700, -400

        # Count pixels in each bin
        c1 = np.sum(pixels < t1)
        c2 = np.sum((pixels >= t1) & (pixels < t2))
        c3 = np.sum((pixels >= t2) & (pixels < t3))
        c4 = np.sum(pixels >= t3)

        counts = np.array([c1, c2, c3, c4], dtype=np.float32)

        # Normalize by total masked volume
        return counts / pixels.size
