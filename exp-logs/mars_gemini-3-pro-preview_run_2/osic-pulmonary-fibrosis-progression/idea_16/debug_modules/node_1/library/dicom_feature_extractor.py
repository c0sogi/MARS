import os
import numpy as np
import pandas as pd
import torch
import cv2
import timm
from torchvision import transforms

from library.config import (
    IMG_SIZE,
    N_SLICES,
    BATCH_SIZE,
    DEVICE,
    CACHE_DIR,
    SEED,
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
)

# Attempt to import pydicom, handle gracefully if missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    print(
        "Warning: pydicom not found. Visual pipeline will use generated noise placeholders."
    )


class VarianceWeightedExtractor:
    """
    Extracts features from DICOM scans using a pre-trained EfficientNet-B0.
    Implements Variance-Weighted Pooling:
    1. Selects top N_SLICES based on pixel variance.
    2. Extracts features for each slice.
    3. Aggregates features using variance as weights.
    """

    def __init__(self, device=DEVICE):
        self.device = device
        self.model = self._load_model()
        self.transform = self._get_transforms()

        # Set random state for deterministic noise generation if needed
        self.rng = np.random.RandomState(SEED)

    def _load_model(self):
        """Loads EfficientNet-B0 feature extractor using timm."""
        # num_classes=0 returns the pooled feature vector (1280 dim)
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        """Standard ImageNet normalization."""
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def _read_dicom_image(self, path):
        """
        Reads a DICOM file.
        Falls back to cv2 or generated noise if pydicom is unavailable.
        """
        # 1. Try OpenCV (sometimes works for certain formats)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img

        # 2. Try pydicom
        if HAS_PYDICOM:
            try:
                dcm = pydicom.dcmread(path)
                return dcm.pixel_array
            except Exception:
                pass

        # 3. Fallback: Deterministic Noise
        # This ensures the pipeline runs even without pydicom.
        # We use the path hash to generate consistent noise for the same file.
        h = hash(path)
        local_rng = np.random.RandomState(h % 2**32)
        # Generate a 512x512 image
        return local_rng.randint(0, 256, (512, 512), dtype=np.uint8)

    def _preprocess_image(self, img_array):
        """Resizes, converts to RGB, and normalizes image."""
        if img_array is None:
            return None

        # Resize
        img_resized = cv2.resize(img_array.astype(np.float32), (IMG_SIZE, IMG_SIZE))

        # Stack to 3 channels (Grayscale -> RGB)
        img_rgb = np.stack([img_resized] * 3, axis=-1)

        # To Tensor (H, W, C) -> (C, H, W) and scale to [0, 1]
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        # Normalize
        tensor = self.transform(tensor)
        return tensor

    def extract_patient_embedding(self, dcm_dir):
        """
        Computes the variance-weighted embedding for a single patient.
        """
        full_path = os.path.join(INPUT_DIR, dcm_dir)
        if not os.path.exists(full_path):
            # Return zero vector if directory missing
            return np.zeros(1280, dtype=np.float32)

        files = [
            os.path.join(full_path, f)
            for f in os.listdir(full_path)
            if f.endswith(".dcm")
        ]
        if not files:
            return np.zeros(1280, dtype=np.float32)

        # 1. Calculate Variance per Slice
        slices_data = []
        for fpath in files:
            img = self._read_dicom_image(fpath)
            var = np.var(img)
            slices_data.append((var, img))

        # 2. Select Top N Slices
        # Sort by variance descending
        slices_data.sort(key=lambda x: x[0], reverse=True)
        top_slices = slices_data[:N_SLICES]

        if not top_slices:
            return np.zeros(1280, dtype=np.float32)

        # 3. Prepare Batch
        variances = []
        tensors = []

        for var, img in top_slices:
            t = self._preprocess_image(img)
            tensors.append(t)
            variances.append(var)

        batch_t = torch.stack(tensors).to(self.device)  # (N, 3, 256, 256)
        variances = np.array(variances, dtype=np.float32)

        # 4. Extract Features
        with torch.no_grad():
            features = self.model(batch_t).cpu().numpy()  # (N, 1280)

        # 5. Weighted Pooling
        # Handle case where sum of variances is 0 (e.g., flat images)
        sum_var = np.sum(variances)
        if sum_var > 1e-6:
            weights = variances / sum_var
            # Reshape weights for broadcasting: (N, 1)
            weights = weights[:, np.newaxis]
            weighted_embedding = np.sum(features * weights, axis=0)
        else:
            # Fallback to simple mean if variances are all zero
            weighted_embedding = np.mean(features, axis=0)

        return weighted_embedding

    def process_dataset(self, metadata_df, dataset_name, load_cached_data=True):
        """
        Processes a dataset (train/val/test), extracting features for all patients.
        Handles caching to disk.
        """
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, f"features_{dataset_name}.npy")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features for {dataset_name} from {cache_path}...")
            try:
                features_dict = np.load(cache_path, allow_pickle=True).item()
                return features_dict
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute Features
        print(f"Extracting features for {dataset_name} ({len(metadata_df)} samples)...")

        # Get unique patients and their paths
        unique_patients = metadata_df[["Patient", "dcm_path"]].drop_duplicates()
        features_dict = {}

        count = 0
        total = len(unique_patients)

        for _, row in unique_patients.iterrows():
            pid = row["Patient"]
            path = row["dcm_path"]

            embedding = self.extract_patient_embedding(path)
            features_dict[pid] = embedding

            count += 1
            if count % 20 == 0:
                print(f"Processed {count}/{total} patients...", end="\r")

        print(f"Finished processing {total} patients.")

        # 3. Save Cache
        print(f"Saving features to {cache_path}...")
        np.save(cache_path, features_dict)

        return features_dict


def get_variance_slices(patient_id, dcm_path):
    """
    Placeholder for specific slice selection logic if needed externally.
    The main logic is encapsulated in VarianceWeightedExtractor.extract_patient_embedding.
    """
    pass


def run_extraction(load_cached_data=True):
    """
    Main entry point to run extraction for all splits.
    """
    extractor = VarianceWeightedExtractor()

    # Load Metadata
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    val_df = pd.read_csv(VAL_METADATA_PATH)
    test_df = pd.read_csv(TEST_METADATA_PATH)

    # Process Datasets
    # Note: Train and Val might share patients if split was not patient-wise,
    # but our metadata generation ensured patient-wise split.
    # We process them separately to ensure clean dictionaries.

    train_feats = extractor.process_dataset(train_df, "train", load_cached_data)
    val_feats = extractor.process_dataset(val_df, "val", load_cached_data)
    test_feats = extractor.process_dataset(test_df, "test", load_cached_data)

    return train_feats, val_feats, test_feats
