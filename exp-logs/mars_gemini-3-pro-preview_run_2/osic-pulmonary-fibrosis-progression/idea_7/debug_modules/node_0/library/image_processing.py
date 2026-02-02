import os
import glob
import numpy as np
import pandas as pd
import cv2
import torch
import timm
import struct
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

from library.config import (
    INPUT_DIR,
    IDEA_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    IMG_SIZE,
    SLICE_PERCENTILES,
    HU_MIN,
    HU_MAX,
    PCA_COMPONENTS,
    DEVICE,
    BATCH_SIZE,
)
from library.utils import save_numpy, load_numpy


# ====================================================
# Custom DICOM Parser (No pydicom)
# ====================================================
class DicomParser:
    """
    A lightweight DICOM parser using standard libraries and numpy.
    Optimized for extracting Pixel Data, Rescale Intercept/Slope, and Geometry.
    Assumes Little Endian Explicit VR for most metadata.
    """

    # Tags (Group, Element)
    TAG_ROWS = b"\x28\x00\x10\x00"
    TAG_COLS = b"\x28\x00\x11\x00"
    TAG_INTERCEPT = b"\x28\x00\x52\x10"
    TAG_SLOPE = b"\x28\x00\x53\x10"
    TAG_PIXEL_DATA = b"\xe0\x7f\x10\x00"

    def __init__(self):
        pass

    @staticmethod
    def get_tag_value(data, tag_bytes, vr_type="US"):
        """
        Searches for a tag and returns its value.
        vr_type: 'US' (Unsigned Short), 'DS' (Decimal String)
        """
        idx = data.find(tag_bytes)
        if idx == -1:
            return None

        # Standard Explicit VR structure: Tag(4) + VR(2) + Length(2) + Value(Length)
        # Note: For some VRs (OB, OW, OF, SQ, UT, UN), structure is Tag(4) + VR(2) + Reserved(2) + Length(4)
        # We assume standard short VRs for Rows/Cols/Intercept/Slope in common CT DICOMs.

        try:
            # Skip Tag (4 bytes)
            current_idx = idx + 4

            # Read VR (2 bytes) - not strictly needed if we assume, but good to skip
            current_idx += 2

            # Read Length (2 bytes)
            length = struct.unpack("<H", data[current_idx : current_idx + 2])[0]
            current_idx += 2

            value_bytes = data[current_idx : current_idx + length]

            if vr_type == "US":
                return struct.unpack("<H", value_bytes)[0]
            elif vr_type == "DS":
                return float(value_bytes.decode("utf-8", errors="ignore").strip())

        except Exception:
            return None
        return None

    @staticmethod
    def read_pixel_data(data, rows, cols):
        """
        Extracts pixel data array.
        """
        idx = data.find(DicomParser.TAG_PIXEL_DATA)
        if idx == -1:
            return np.zeros((rows, cols), dtype=np.int16)

        try:
            # Pixel Data Structure: Tag(4) + VR(2) + Reserved(2) + Length(4) + Value
            # Or Tag(4) + Length(4) + Value (Implicit VR)
            # We use a heuristic: Look for the start of the bulk data.
            # Usually for OW (Other Word) VR:

            # Check VR at idx+4
            vr = data[idx + 4 : idx + 6]

            if vr in [b"OW", b"OB"]:
                # Explicit VR with 4-byte length
                offset = 12  # 4(Tag) + 2(VR) + 2(Res) + 4(Len)
            else:
                # Implicit VR or other structure, assume 4 byte length immediately after tag
                offset = 8  # 4(Tag) + 4(Len)

            start_idx = idx + offset
            expected_bytes = rows * cols * 2

            # Safety check on buffer size
            if start_idx + expected_bytes > len(data):
                # Fallback: try to read from end
                start_idx = len(data) - expected_bytes

            if start_idx < 0:
                return np.zeros((rows, cols), dtype=np.int16)

            pixel_data = np.frombuffer(
                data, dtype=np.int16, count=rows * cols, offset=start_idx
            )
            return pixel_data.reshape((rows, cols))

        except Exception:
            return np.zeros((rows, cols), dtype=np.int16)

    def read_dicom(self, path):
        """
        Parses a DICOM file and returns the HU-scaled numpy array.
        """
        try:
            with open(path, "rb") as f:
                data = f.read()

            # Parse Metadata
            rows = self.get_tag_value(data, self.TAG_ROWS, "US")
            cols = self.get_tag_value(data, self.TAG_COLS, "US")
            intercept = self.get_tag_value(data, self.TAG_INTERCEPT, "DS")
            slope = self.get_tag_value(data, self.TAG_SLOPE, "DS")

            # Defaults if missing
            if rows is None:
                rows = 512
            if cols is None:
                cols = 512
            if intercept is None:
                intercept = -1024
            if slope is None:
                slope = 1.0

            # Parse Pixels
            img = self.read_pixel_data(data, rows, cols)

            # Apply Rescale
            img = img.astype(np.float32) * slope + intercept

            return img

        except Exception as e:
            # Return a blank image on failure to avoid crashing the pipeline
            return np.full((512, 512), -1000.0, dtype=np.float32)


# ====================================================
# Image Processing & Feature Extraction
# ====================================================


class FeatureExtractor:
    def __init__(self, device=DEVICE):
        self.device = device
        # Load EfficientNet B0
        self.model = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )
        self.model.to(device)
        self.model.eval()

        # Normalization for ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        self.parser = DicomParser()

    def preprocess_image(self, img):
        """
        Resizes, clips HU, normalizes to 0-1, and converts to tensor.
        """
        # Clip HU
        img = np.clip(img, HU_MIN, HU_MAX)

        # Normalize to 0-1
        img = (img - HU_MIN) / (HU_MAX - HU_MIN)

        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Convert to 3 channel (grayscale repeated)
        img = np.stack([img, img, img], axis=-1)

        # To Tensor (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return img_tensor

    def get_anatomical_slices(self, patient_dir):
        """
        Selects slices at configured percentiles (Apex, Mid, Base).
        """
        files = sorted(
            glob.glob(os.path.join(patient_dir, "*.dcm")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )

        if not files:
            return None

        num_files = len(files)
        selected_slices = []

        for p in SLICE_PERCENTILES:
            idx = int(num_files * p)
            idx = min(idx, num_files - 1)

            img = self.parser.read_dicom(files[idx])
            selected_slices.append(self.preprocess_image(img))

        # Stack into batch: (3, 3, H, W)
        return torch.stack(selected_slices)

    def compute_lung_volume(self, patient_dir):
        """
        Approximates lung volume via voxel counting on strided slices.
        """
        files = sorted(
            glob.glob(os.path.join(patient_dir, "*.dcm")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
        )

        if not files:
            return 0.0

        # Stride for efficiency (e.g., every 5th slice)
        stride = 5
        selected_files = files[::stride]

        voxel_count = 0

        for f in selected_files:
            img = self.parser.read_dicom(f)
            # Threshold for lung tissue (approx -1000 to -400 HU)
            mask = (img > -1000) & (img < -400)
            voxel_count += np.sum(mask)

        # Adjust for stride to estimate total count
        total_estimated_voxels = voxel_count * stride

        # Normalize arbitrarily to keep values manageable (e.g., / 1e6)
        # or return raw. Raw is fine for linear models/PCA if scaled later.
        return float(total_estimated_voxels)

    @torch.no_grad()
    def extract_features(self, patient_id, dcm_path):
        """
        Extracts concatenated texture features and volume.
        Returns: numpy array of shape (Feature_Dim + 1,)
        """
        full_path = os.path.join(INPUT_DIR, dcm_path)

        # 1. Texture Features
        slices_tensor = self.get_anatomical_slices(full_path)
        if slices_tensor is None:
            # Fallback for missing data
            return np.zeros(1280 * 3 + 1)

        slices_tensor = slices_tensor.to(self.device)

        # Normalize with ImageNet stats
        slices_tensor = (slices_tensor - self.mean) / self.std

        # Extract features (Batch of 3)
        features = self.model(slices_tensor)  # (3, 1280)

        # Flatten/Concatenate: [Top, Mid, Bot]
        texture_vector = features.flatten().cpu().numpy()  # (3840,)

        # 2. Volume
        volume = self.compute_lung_volume(full_path)

        # 3. Combine
        # We return texture and volume separately or combined?
        # We will combine them later after PCA on texture.
        # For now, return tuple
        return texture_vector, volume


def generate_dataset_features(metadata_path, subset_name, load_cached_data=True):
    """
    Main function to generate or load features for a dataset.

    Args:
        metadata_path: Path to the metadata CSV.
        subset_name: 'train', 'val', or 'test'.
        load_cached_data: Boolean to use cache.

    Returns:
        Dictionary {PatientID: Feature_Vector}
        Where Feature_Vector is PCA-reduced Texture + Volume.
    """

    # Paths
    cache_path_ids = os.path.join(IDEA_DIR, f"{subset_name}_ids.npy")
    cache_path_feats = os.path.join(IDEA_DIR, f"{subset_name}_features.npy")

    # 1. Check Cache
    if load_cached_data:
        ids = load_numpy(cache_path_ids)
        feats = load_numpy(cache_path_feats)
        if ids is not None and feats is not None:
            print(f"Loaded {subset_name} features from cache.")
            # Reconstruct dictionary
            return {pid: feat for pid, feat in zip(ids, feats)}

    print(f"Generating {subset_name} features from scratch...")

    # 2. Load Metadata
    df = pd.read_csv(metadata_path)
    unique_patients = df[["Patient", "dcm_path"]].drop_duplicates()

    # 3. Extract Raw Features
    extractor = FeatureExtractor()

    patient_ids = []
    raw_textures = []
    volumes = []

    total = len(unique_patients)
    for i, row in unique_patients.iterrows():
        pid = row["Patient"]
        path = row["dcm_path"]

        tex, vol = extractor.extract_features(pid, path)

        patient_ids.append(pid)
        raw_textures.append(tex)
        volumes.append(vol)

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{total} patients...")

    raw_textures = np.array(raw_textures)  # (N, 3840)
    volumes = np.array(volumes).reshape(-1, 1)  # (N, 1)

    # 4. PCA Processing
    # Logic: If Train, fit PCA. If Val/Test, load PCA.
    pca_path = os.path.join(IDEA_DIR, "pca_model.joblib")

    if subset_name == "train":
        print("Fitting PCA on training data...")
        # Standardize before PCA? Yes, usually good for deep features.
        # But for simplicity and robustness in this pipeline, we apply PCA directly
        # as EfficientNet features are roughly same scale (ReLU outputs).
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        texture_pca = pca.fit_transform(raw_textures)
        joblib.dump(pca, pca_path)
    else:
        if os.path.exists(pca_path):
            pca = joblib.load(pca_path)
            texture_pca = pca.transform(raw_textures)
        else:
            # Fallback if PCA model missing (should not happen if train runs first)
            print(
                "Warning: PCA model not found. Initializing new PCA (suboptimal for val/test)."
            )
            pca = PCA(
                n_components=min(PCA_COMPONENTS, len(raw_textures)), random_state=42
            )
            texture_pca = pca.fit_transform(raw_textures)

    # 5. Combine Features
    # Scale Volume?
    # Volume is large (e.g. 1e6). PCA features are small.
    # We apply a log1p transform to volume to stabilize it.
    volumes_log = np.log1p(volumes)

    final_features = np.hstack([texture_pca, volumes_log])

    # 6. Save Cache
    save_numpy(cache_path_ids, np.array(patient_ids))
    save_numpy(cache_path_feats, final_features)

    print(f"Saved {subset_name} features to {IDEA_DIR}")

    return {pid: feat for pid, feat in zip(patient_ids, final_features)}
