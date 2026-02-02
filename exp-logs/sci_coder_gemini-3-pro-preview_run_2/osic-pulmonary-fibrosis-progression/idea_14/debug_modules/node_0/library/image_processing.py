import os
import numpy as np
import cv2
import torch
import timm
from library.config import Config

# Attempt to import pydicom; handle graceful fallback if not installed
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


class ImageProcessor:
    def __init__(self, device=None):
        """
        Initializes the ImageProcessor with a pre-trained EfficientNet model.
        """
        self.device = device if device else Config.DEVICE

        # Initialize EfficientNet-B0
        # We use a try-except block to handle potential internet access issues for weights
        try:
            self.model = timm.create_model(
                "efficientnet_b0", pretrained=True, num_classes=0
            )
        except Exception:
            print(
                "Warning: Could not download pretrained weights. Using random initialization."
            )
            self.model = timm.create_model(
                "efficientnet_b0", pretrained=False, num_classes=0
            )

        self.model.to(self.device)
        self.model.eval()

        # ImageNet normalization statistics
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def load_scan(self, path):
        """
        Loads CT scan slices from a directory and converts them to Hounsfield Units (HU).
        Supports pydicom (preferred) and cv2 (fallback).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")

        files = [
            f
            for f in os.listdir(path)
            if f.endswith(".dcm") or f.endswith(".png") or f.endswith(".jpg")
        ]
        if not files:
            # Return empty volume if no files found
            return np.zeros((0, 512, 512), dtype=np.float32)

        # Sort files by instance number (filename)
        # Assuming filenames are like '1.dcm', '10.dcm'
        try:
            files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        except ValueError:
            files.sort()

        slices = []

        if HAS_PYDICOM:
            # Method 1: Use pydicom (Standard)
            scans = []
            for f in files:
                try:
                    ds = pydicom.dcmread(os.path.join(path, f))
                    scans.append(ds)
                except Exception:
                    continue

            if not scans:
                return np.zeros((0, 512, 512), dtype=np.float32)

            # Convert to HU
            intercept = (
                scans[0].RescaleIntercept
                if hasattr(scans[0], "RescaleIntercept")
                else -1024
            )
            slope = scans[0].RescaleSlope if hasattr(scans[0], "RescaleSlope") else 1

            image_stack = np.stack([s.pixel_array for s in scans]).astype(np.float32)
            image_stack = image_stack * slope + intercept

            # Handle potential padding values (e.g., -2000)
            image_stack[image_stack < -1024] = -1024
            return image_stack

        else:
            # Method 2: Use OpenCV (Fallback)
            # Assumes 16-bit images where pixel value ~ HU + 1024
            for f in files:
                img_path = os.path.join(path, f)
                # Read as unchanged (16-bit if available)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                slices.append(img)

            if not slices:
                return np.zeros((0, 512, 512), dtype=np.float32)

            image_stack = np.stack(slices).astype(np.float32)

            # Heuristic conversion to HU for CT scans
            # Standard offset is often 1024 or 32768 depending on encoding
            # Here we assume standard offset 1024
            image_stack = image_stack - 1024
            return image_stack

    def segment_lung(self, volume):
        """
        Creates a binary mask for lung tissue based on HU thresholding.
        """
        mask = (volume >= Config.HU_MIN) & (volume <= Config.HU_MAX)
        return mask

    def extract_morphological_profile(self, volume):
        """
        Computes 1D profiles (Area and Density) along the Z-axis and fits polynomials.
        Returns 8 coefficients (4 for Area, 4 for Density).
        """
        if volume.shape[0] == 0:
            return np.zeros(8, dtype=np.float32)

        # Calculate Area (pixel count) and Density (mean HU) per slice
        # We only consider pixels within the lung window
        mask = self.segment_lung(volume)

        area_profile = np.sum(mask, axis=(1, 2))

        # Avoid division by zero for density
        density_profile = np.zeros_like(area_profile, dtype=np.float32)
        for i in range(len(volume)):
            if area_profile[i] > 0:
                density_profile[i] = np.mean(volume[i][mask[i]])
            else:
                density_profile[i] = (
                    Config.HU_MIN
                )  # Default to min density if no lung found

        # Normalize Z-axis to [0, 1]
        z = np.linspace(0, 1, len(volume))

        # Fit polynomials (Degree 3)
        # If volume is too small for degree 3, pad or reduce degree logic handled by polyfit (rank warning)
        # but we ensure output size is fixed.
        if len(z) > Config.POLY_ORDER:
            area_coeffs = np.polyfit(z, area_profile, Config.POLY_ORDER)
            density_coeffs = np.polyfit(z, density_profile, Config.POLY_ORDER)
        else:
            # Fallback for very few slices
            area_coeffs = np.zeros(Config.POLY_ORDER + 1, dtype=np.float32)
            density_coeffs = np.zeros(Config.POLY_ORDER + 1, dtype=np.float32)
            # Simple mean fill if possible
            if len(z) > 0:
                area_coeffs[-1] = np.mean(area_profile)
                density_coeffs[-1] = np.mean(density_profile)

        return np.concatenate([area_coeffs, density_coeffs]).astype(np.float32)

    def extract_stratified_texture(self, volume):
        """
        Extracts deep texture features from 3 stratified slices (Top, Middle, Bottom).
        Selects the slice with max variance in each zone.
        """
        if volume.shape[0] == 0:
            # Return zero vector matching EfficientNet-B0 output * 3
            return np.zeros(1280 * 3, dtype=np.float32)

        num_slices = volume.shape[0]
        # Define zones
        splits = np.array_split(np.arange(num_slices), 3)

        selected_slices = []

        for split_indices in splits:
            if len(split_indices) == 0:
                # Handle empty zone (e.g. very small volume)
                selected_slices.append(
                    np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
                )
                continue

            # Extract sub-volume
            zone_vol = volume[split_indices]

            # Calculate variance per slice to find the most informative one
            variances = np.var(zone_vol, axis=(1, 2))
            best_idx = np.argmax(variances)
            best_slice = zone_vol[best_idx]

            # Resize to model input size
            resized_slice = cv2.resize(best_slice, (Config.IMG_SIZE, Config.IMG_SIZE))
            selected_slices.append(resized_slice)

        # Ensure we have 3 slices
        while len(selected_slices) < 3:
            selected_slices.append(
                np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
            )

        # Preprocess for EfficientNet
        features_list = []

        with torch.no_grad():
            for slc in selected_slices:
                # Clip to lung window and normalize to [0, 1]
                slc = np.clip(slc, Config.HU_MIN, Config.HU_MAX)
                slc = (slc - Config.HU_MIN) / (Config.HU_MAX - Config.HU_MIN)

                # Convert to tensor (H, W) -> (1, 3, H, W)
                # We replicate the grayscale channel 3 times for RGB model
                tensor = torch.tensor(slc, dtype=torch.float32).to(self.device)
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                tensor = tensor.repeat(1, 3, 1, 1)  # (1, 3, H, W)

                # Normalize with ImageNet stats
                tensor = (tensor - self.mean) / self.std

                # Forward pass
                # EfficientNet-B0 output is (Batch, 1280)
                feat = self.model.forward_features(tensor)
                # Global Average Pooling
                feat = self.model.global_pool(feat)
                if isinstance(feat, tuple):  # Handle some timm versions
                    feat = feat[0]
                feat = feat.flatten().cpu().numpy()

                features_list.append(feat)

        return np.concatenate(features_list).astype(np.float32)

    def process_patient(self, patient_id, dcm_path, load_cached_data=True):
        """
        Orchestrates the feature extraction pipeline for a single patient.
        Implements caching to disk.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True).item()
                return data
            except Exception:
                pass  # Corrupt cache, recompute

        # 2. Load and Process Scan
        # Construct full path. Metadata path is relative to INPUT_DIR
        full_path = os.path.join(Config.INPUT_DIR, dcm_path)

        try:
            volume = self.load_scan(full_path)

            # 3. Extract Features
            morph_features = self.extract_morphological_profile(volume)
            texture_features = self.extract_stratified_texture(volume)

            result = {"morph": morph_features, "texture": texture_features}

            # 4. Save to Cache
            # Ensure directory exists (redundant if Config.setup called, but safe)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, result)

            return result

        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            # Return zero features on failure to allow pipeline to continue
            return {
                "morph": np.zeros(8, dtype=np.float32),
                "texture": np.zeros(1280 * 3, dtype=np.float32),
            }
