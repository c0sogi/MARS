import os
import glob
import numpy as np
import cv2
import warnings
from library.config import Config

# Attempt to import pydicom (standard for DICOM) or rasterio (fallback from installed list)
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class DicomProcessor:
    """
    Handles loading of DICOM CT scans and generation of Fixed Overlapping Orthogonal Tri-Slab inputs.
    """

    def __init__(self):
        self.img_size = Config.IMG_SIZE
        self.cache_dir = Config.CACHE_DIR
        self.slab_overlap = Config.SLAB_OVERLAP

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _read_dicom_file(self, path):
        """
        Reads a single DICOM file and returns the pixel array in Hounsfield Units.
        """
        if HAS_PYDICOM:
            try:
                dcm = pydicom.dcmread(path)
                img = dcm.pixel_array.astype(np.float32)

                # Apply Rescale Slope and Intercept to convert to HU
                intercept = getattr(dcm, "RescaleIntercept", 0)
                slope = getattr(dcm, "RescaleSlope", 1)

                # Handle cases where tags are arrays
                if isinstance(slope, (list, tuple, np.ndarray)):
                    slope = slope[0]
                if isinstance(intercept, (list, tuple, np.ndarray)):
                    intercept = intercept[0]

                img = img * float(slope) + float(intercept)
                return img
            except Exception as e:
                # If pydicom fails on a specific file
                return None

        elif HAS_RASTERIO:
            # Fallback using Rasterio (GDAL)
            try:
                with rasterio.open(path) as src:
                    img = src.read(1).astype(np.float32)
                    # Rasterio might return raw values.
                    # Without explicit metadata parsing, we assume raw or pre-processed.
                    # This is a best-effort fallback.
                    return img
            except Exception:
                return None
        else:
            raise ImportError(
                "No suitable library found to read DICOM files (pydicom or rasterio)."
            )

    def load_scan(self, dicom_dir):
        """
        Loads all DICOM files from a directory into a 3D numpy array (D, H, W).
        """
        files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
        if not files:
            return None

        # Sort by instance number. Filenames are usually '1.dcm', '10.dcm', etc.
        # We sort by the integer value of the filename.
        try:
            files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except ValueError:
            files.sort()  # Fallback to lexical sort if filenames are non-numeric

        slices = []
        for f in files:
            img = self._read_dicom_file(f)
            if img is not None:
                slices.append(img)

        if not slices:
            return None

        # Stack into 3D volume (Depth, Height, Width)
        try:
            volume = np.stack(slices)
            return volume
        except ValueError:
            # Handle mismatch in image dimensions
            return None

    def _process_plane(self, volume, plane="axial"):
        """
        Generates a Tri-Slab MIP image for the specified plane.
        """
        # Volume is (Z, Y, X) -> (Depth, Height, Width)
        if plane == "coronal":
            # Coronal view is looking from the front (Y-axis becomes depth)
            # Transpose to (Y, Z, X) so we slice along the first dimension
            vol_perm = volume.transpose(1, 0, 2)
        else:
            # Axial view is looking from top/bottom (Z-axis is depth)
            vol_perm = volume

        depth = vol_perm.shape[0]

        # Calculate slab boundaries with overlap
        # 3 slabs: 0-33%, 33-66%, 66-100% (centers) with overlap
        one_third = depth / 3.0
        overlap_px = int(depth * self.slab_overlap)

        # Slab 1
        s1_start = 0
        s1_end = int(one_third + overlap_px)

        # Slab 2
        s2_start = int(one_third - overlap_px)
        s2_end = int(2 * one_third + overlap_px)

        # Slab 3
        s3_start = int(2 * one_third - overlap_px)
        s3_end = depth

        # Clip indices
        slabs_indices = [
            (max(0, s1_start), min(depth, s1_end)),
            (max(0, s2_start), min(depth, s2_end)),
            (max(0, s3_start), min(depth, s3_end)),
        ]

        channels = []
        for start, end in slabs_indices:
            if start >= end:
                # Handle edge case of very small volume
                slab_mip = np.max(vol_perm, axis=0)
            else:
                slab = vol_perm[start:end, :, :]
                if slab.shape[0] == 0:
                    slab_mip = np.zeros(
                        (vol_perm.shape[1], vol_perm.shape[2]), dtype=np.float32
                    )
                else:
                    # Maximum Intensity Projection
                    slab_mip = np.max(slab, axis=0)
            channels.append(slab_mip)

        # Stack into RGB (H, W, 3)
        img = np.stack(channels, axis=-1)

        # Windowing and Normalization
        # Clip to [-1000, 1000] HU (Air to Bone)
        min_hu, max_hu = -1000.0, 1000.0
        img = np.clip(img, min_hu, max_hu)

        # Normalize to 0-255
        img = (img - min_hu) / (max_hu - min_hu)
        img = (img * 255).astype(np.uint8)

        # Resize to target resolution
        # cv2.resize expects (width, height)
        img = cv2.resize(
            img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
        )

        return img

    def generate_dual_view_mips(self, patient_id, dicom_dir, load_cached_data=True):
        """
        Generates (or loads) the Axial and Coronal Tri-Slab MIPs for a patient.

        Args:
            patient_id (str): Unique patient identifier.
            dicom_dir (str): Path to the directory containing DICOM files.
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            tuple: (axial_img, coronal_img) as numpy arrays of shape (H, W, 3).
        """
        axial_path = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        coronal_path = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")

        # 1. Try Loading from Cache
        if load_cached_data:
            if os.path.exists(axial_path) and os.path.exists(coronal_path):
                try:
                    axial = np.load(axial_path)
                    coronal = np.load(coronal_path)
                    return axial, coronal
                except Exception:
                    # If load fails (corrupt file), proceed to recompute
                    pass

        # 2. Compute from Scratch
        volume = self.load_scan(dicom_dir)

        if volume is None:
            # Return black images if scan loading fails
            # This prevents pipeline crash on bad data
            axial = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            coronal = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            axial = self._process_plane(volume, plane="axial")
            coronal = self._process_plane(volume, plane="coronal")

        # 3. Save to Cache
        try:
            np.save(axial_path, axial)
            np.save(coronal_path, coronal)
        except Exception as e:
            print(f"Warning: Failed to cache data for {patient_id}: {e}")

        return axial, coronal
