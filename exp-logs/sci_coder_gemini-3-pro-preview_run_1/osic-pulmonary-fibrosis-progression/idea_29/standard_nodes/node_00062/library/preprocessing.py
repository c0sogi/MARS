import os
import glob
import numpy as np
import pandas as pd
import cv2
import logging

# Attempt to import pydicom.
# While strictly required for .dcm files, we wrap it to prevent immediate import errors
# if the environment verification script runs without it.
try:
    import pydicom
except ImportError:
    pydicom = None

from library.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Handles the conversion of raw DICOM series into Fixed Overlapping Orthogonal Tri-Slabs.
    """

    def __init__(self, img_size=Config.IMG_SIZE):
        self.img_size = img_size

    def load_scan(self, path):
        """
        Loads a CT scan from a directory of DICOM files.
        Returns a numpy array of shape (D, H, W) in Hounsfield Units.
        """
        if not os.path.exists(path):
            logger.warning(f"Directory not found: {path}")
            return None

        # Find all DICOM files
        files = glob.glob(os.path.join(path, "*.dcm"))
        if not files:
            return None

        if pydicom is None:
            logger.error("pydicom is not installed. Cannot process DICOM files.")
            return None

        # Read slices
        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(f)
                slices.append(ds)
            except Exception:
                continue

        if not slices:
            return None

        # Sort slices: prefer ImagePositionPatient[2] (Z-coord), fallback to InstanceNumber
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except (AttributeError, ValueError):
            try:
                slices.sort(key=lambda x: int(x.InstanceNumber))
            except (AttributeError, ValueError):
                # If sorting fails, assume filename order or random (suboptimal but prevents crash)
                pass

        # Extract pixel data and convert to Hounsfield Units (HU)
        images = []
        for s in slices:
            try:
                # Convert to float
                img = s.pixel_array.astype(np.float32)

                # Apply Rescale Slope and Intercept
                slope = getattr(s, "RescaleSlope", 1)
                intercept = getattr(s, "RescaleIntercept", -1024)

                if slope != 1:
                    img = slope * img.astype(np.float64)
                    img = img.astype(np.float32)

                img += intercept
                images.append(img)
            except Exception:
                continue

        if not images:
            return None

        return np.array(images)

    def normalize_hu(self, volume):
        """
        Normalizes Hounsfield Units to [0, 1] range.
        Uses a broad window [-1000, 400] covering Air to Bone.
        """
        min_hu = -1000
        max_hu = 400

        # Clip and Normalize
        volume = np.clip(volume, min_hu, max_hu)
        volume = (volume - min_hu) / (max_hu - min_hu)
        return volume

    def generate_tri_slabs(self, volume):
        """
        Generates Axial and Coronal Tri-Slab MIPs.

        Logic:
        1. Axial: Split Depth (D) into 3 overlapping slabs. MIP along D.
        2. Coronal: Split Height (H) into 3 overlapping slabs. MIP along H.

        Slab boundaries: 0-40%, 30-70%, 60-100% (approx 15% overlap relative to slab size).

        Returns:
            axial_img: (H, W, 3)
            coronal_img: (D, W, 3)
        """
        D, H, W = volume.shape

        # --- 1. Axial View (MIP along Axis 0 / Depth) ---
        # Define indices
        p1_end = int(D * 0.40)
        p2_start = int(D * 0.30)
        p2_end = int(D * 0.70)
        p3_start = int(D * 0.60)

        # Handle cases with very few slices
        if D < 3:
            mip_ax_1 = np.max(volume, axis=0)
            mip_ax_2 = mip_ax_1
            mip_ax_3 = mip_ax_1
        else:
            # Extract slabs
            s1 = volume[:p1_end] if p1_end > 0 else volume
            s2 = volume[p2_start:p2_end] if p2_end > p2_start else volume
            s3 = volume[p3_start:] if p3_start < D else volume

            # Compute MIPs
            mip_ax_1 = np.max(s1, axis=0) if s1.shape[0] > 0 else np.max(volume, axis=0)
            mip_ax_2 = np.max(s2, axis=0) if s2.shape[0] > 0 else np.max(volume, axis=0)
            mip_ax_3 = np.max(s3, axis=0) if s3.shape[0] > 0 else np.max(volume, axis=0)

        axial_img = np.stack([mip_ax_1, mip_ax_2, mip_ax_3], axis=-1)

        # --- 2. Coronal View (MIP along Axis 1 / Height) ---
        # Coronal plane is usually X-Z (looking from Y).
        # Volume is (D, H, W). We collapse H.

        p1_end_c = int(H * 0.40)
        p2_start_c = int(H * 0.30)
        p2_end_c = int(H * 0.70)
        p3_start_c = int(H * 0.60)

        if H < 3:
            mip_cor_1 = np.max(volume, axis=1)
            mip_cor_2 = mip_cor_1
            mip_cor_3 = mip_cor_1
        else:
            s1_c = volume[:, :p1_end_c, :]
            s2_c = volume[:, p2_start_c:p2_end_c, :]
            s3_c = volume[:, p3_start_c:, :]

            mip_cor_1 = (
                np.max(s1_c, axis=1) if s1_c.shape[1] > 0 else np.max(volume, axis=1)
            )
            mip_cor_2 = (
                np.max(s2_c, axis=1) if s2_c.shape[1] > 0 else np.max(volume, axis=1)
            )
            mip_cor_3 = (
                np.max(s3_c, axis=1) if s3_c.shape[1] > 0 else np.max(volume, axis=1)
            )

        coronal_img = np.stack([mip_cor_1, mip_cor_2, mip_cor_3], axis=-1)

        return axial_img, coronal_img

    def resize_and_format(self, img):
        """
        Resizes image to target size and converts to uint8 [0, 255].
        """
        # Resize
        img_resized = cv2.resize(
            img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
        )

        # Scale to 0-255 and cast
        img_uint8 = (img_resized * 255).astype(np.uint8)
        return img_uint8

    def process_patient(self, dicom_dir):
        """
        Orchestrates the processing for a single patient directory.
        Returns: (Axial_Image, Coronal_Image)
        """
        full_path = os.path.join(Config.INPUT_ROOT, dicom_dir)

        # 1. Load Volume
        volume = self.load_scan(full_path)

        # Handle failure
        if volume is None:
            empty = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            return empty, empty

        # 2. Normalize
        volume = self.normalize_hu(volume)

        # 3. Generate Tri-Slabs
        axial, coronal = self.generate_tri_slabs(volume)

        # 4. Resize and Format
        axial_out = self.resize_and_format(axial)
        coronal_out = self.resize_and_format(coronal)

        return axial_out, coronal_out


def preprocess_dataset(df, load_cached_data=True):
    """
    Main entry point for preprocessing.
    Iterates over the dataframe, processes DICOMs for each patient, and caches the results.

    Args:
        df (pd.DataFrame): Metadata dataframe containing 'Patient' and 'dicom_dir'.
        load_cached_data (bool): If True, skips processing if file exists in cache.
    """
    preprocessor = Preprocessor()

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    unique_patients = df["Patient"].unique()
    print(f"Preprocessing {len(unique_patients)} patients...")

    count_processed = 0
    count_cached = 0
    count_failed = 0

    for pid in unique_patients:
        # Define cache file paths
        axial_path = os.path.join(Config.CACHE_DIR, f"{pid}_axial.npy")
        coronal_path = os.path.join(Config.CACHE_DIR, f"{pid}_coronal.npy")

        # Check cache
        if (
            load_cached_data
            and os.path.exists(axial_path)
            and os.path.exists(coronal_path)
        ):
            count_cached += 1
            continue

        # Get DICOM directory (use the first entry for this patient)
        patient_rows = df[df["Patient"] == pid]
        if len(patient_rows) == 0:
            continue

        dicom_dir = patient_rows.iloc[0]["dicom_dir"]

        try:
            # Process
            axial, coronal = preprocessor.process_patient(dicom_dir)

            # Save to cache
            np.save(axial_path, axial)
            np.save(coronal_path, coronal)
            count_processed += 1

        except Exception as e:
            logger.error(f"Error processing patient {pid}: {e}")
            # Save zeros to prevent pipeline crash
            zeros = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            np.save(axial_path, zeros)
            np.save(coronal_path, zeros)
            count_failed += 1

    print(
        f"Preprocessing Complete. Processed: {count_processed}, Cached: {count_cached}, Failed: {count_failed}"
    )
