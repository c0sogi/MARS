import os
import re
import glob
import numpy as np
import pandas as pd
from library.config import Config
from library.data_processing import read_dicom_robust
from library.utils import get_logger

# Initialize Logger
logger = get_logger("roi_selection")


class ROIGenerator:
    """
    Implements the Logical-Consensus ROI Selection strategy.
    """

    def __init__(self):
        self.roi_modalities = Config.ROI_MODALITIES  # ["FLAIR", "T1wCE"]
        self.depth_range = Config.ROI_DEPTH_RANGE  # (0.15, 0.85)
        self.stack_offsets = Config.STACK_OFFSETS  # [-5, 0, 5]
        self.input_modalities = (
            Config.INPUT_MODALITIES
        )  # ["FLAIR", "T1w", "T1wCE", "T2w"]

    def _extract_id(self, filename):
        """Extracts the integer ID from 'Image-XXX.dcm'."""
        match = re.search(r"Image-(\d+)\.dcm", filename)
        if match:
            return int(match.group(1))
        return -1

    def _get_sorted_files(self, directory):
        """
        Returns a sorted list of (id, filename, full_path) for a directory.
        """
        if not os.path.isdir(directory):
            return []

        files = glob.glob(os.path.join(directory, "*.dcm"))
        # Parse IDs
        file_data = []
        for f in files:
            fname = os.path.basename(f)
            fid = self._extract_id(fname)
            if fid != -1:
                file_data.append((fid, fname, f))

        # Sort by ID
        file_data.sort(key=lambda x: x[0])
        return file_data

    def _compute_intensity_profile(self, file_data):
        """
        Computes sum of pixel intensities for a list of files.
        Returns numpy array of sums.
        """
        intensities = []
        for _, _, path in file_data:
            try:
                img = read_dicom_robust(path)
                intensities.append(np.sum(img))
            except Exception:
                intensities.append(0.0)
        return np.array(intensities, dtype=np.float32)

    def _normalize_profile(self, profile):
        """Min-Max normalization to [0, 1]."""
        if len(profile) == 0:
            return profile
        p_min = np.min(profile)
        p_max = np.max(profile)
        if p_max > p_min:
            return (profile - p_min) / (p_max - p_min)
        return np.zeros_like(profile)

    def _get_nearest_file(self, target_id, modality_files):
        """
        Finds the file with ID closest to target_id.
        modality_files: list of (id, fname, path)
        """
        if not modality_files:
            return None

        # Extract IDs
        ids = np.array([x[0] for x in modality_files])
        # Find index of nearest
        idx = (np.abs(ids - target_id)).argmin()
        return modality_files[idx][2]  # Return full path

    def process_subject(self, subject_row):
        """
        Determines the 12 file paths for a single subject.
        """
        # 1. Gather file lists for all modalities
        modality_files = {}
        for mod in self.input_modalities:
            # Construct path: input_dir + relative_path
            # The metadata contains relative paths like "train/00000/FLAIR"
            # We need to prepend Config.INPUT_DIR
            rel_path = subject_row.get(f"path_{mod}")
            if rel_path:
                full_dir = os.path.join(Config.INPUT_DIR, rel_path)
                modality_files[mod] = self._get_sorted_files(full_dir)
            else:
                modality_files[mod] = []

        # 2. Check if we have the necessary modalities for ROI selection
        flair_files = modality_files.get("FLAIR", [])
        t1wce_files = modality_files.get("T1wCE", [])

        if not flair_files:
            # Fallback if FLAIR is missing: return empty strings
            return {
                f"{mod}_{i}": ""
                for mod in self.input_modalities
                for i in range(len(self.stack_offsets))
            }

        # 3. Compute Profiles
        flair_profile = self._compute_intensity_profile(flair_files)

        # If T1wCE is available, use consensus. If not, fallback to FLAIR only.
        if t1wce_files:
            t1wce_profile = self._compute_intensity_profile(t1wce_files)

            # Resample T1wCE to match FLAIR length
            if len(flair_profile) > 1 and len(t1wce_profile) > 1:
                x_t1wce = np.linspace(0, 1, len(t1wce_profile))
                x_flair = np.linspace(0, 1, len(flair_profile))
                t1wce_resampled = np.interp(x_flair, x_t1wce, t1wce_profile)
            elif len(flair_profile) > 0:
                # If single slice, just broadcast mean
                t1wce_resampled = np.full_like(flair_profile, np.mean(t1wce_profile))
            else:
                t1wce_resampled = np.zeros_like(flair_profile)

            # Normalize and Sum
            consensus = self._normalize_profile(
                flair_profile
            ) + self._normalize_profile(t1wce_resampled)
        else:
            consensus = self._normalize_profile(flair_profile)

        # 4. Select Peak in Range
        n_slices = len(consensus)
        if n_slices == 0:
            return {
                f"{mod}_{i}": ""
                for mod in self.input_modalities
                for i in range(len(self.stack_offsets))
            }

        start_idx = int(n_slices * self.depth_range[0])
        end_idx = int(n_slices * self.depth_range[1])

        # Handle edge case where range is empty or too small
        if start_idx >= end_idx:
            start_idx = 0
            end_idx = n_slices

        # Extract valid region
        valid_region = consensus[start_idx:end_idx]
        if len(valid_region) == 0:
            peak_relative_idx = 0
        else:
            peak_relative_idx = np.argmax(valid_region)

        peak_idx = start_idx + peak_relative_idx

        # 5. Determine Logical Indices and Explicit IDs
        # We use FLAIR as the anchor for logical indexing
        anchor_id = flair_files[peak_idx][0]

        # We want logical neighbors: peak-5, peak, peak+5
        # Convert these logical neighbors to Explicit IDs based on FLAIR's list
        # We clamp logical indices to [0, n_slices-1]

        paths_dict = {}

        for offset_idx, offset in enumerate(self.stack_offsets):
            target_logical_idx = np.clip(peak_idx + offset, 0, n_slices - 1)
            target_explicit_id = flair_files[target_logical_idx][0]

            # 6. Resolve paths for all modalities
            for mod in self.input_modalities:
                # Find file with target_explicit_id in this modality
                # If not found, find nearest
                files = modality_files.get(mod, [])
                if not files:
                    paths_dict[f"{mod}_{offset_idx}"] = ""
                else:
                    path = self._get_nearest_file(target_explicit_id, files)
                    paths_dict[f"{mod}_{offset_idx}"] = path if path else ""

        return paths_dict


def generate_roi_cache(metadata_df, load_cached_data=True):
    """
    Generates or loads the ROI cache.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing subject metadata.
        load_cached_data (bool): Whether to attempt loading from disk.

    Returns:
        pd.DataFrame: DataFrame with BraTS21ID and columns for each file path.
                      Format: BraTS21ID, FLAIR_0, FLAIR_1, FLAIR_2, T1w_0, ...
    """
    cache_file = os.path.join(Config.CACHE_DIR, "roi_cache.parquet")

    # 1. Try Load
    if load_cached_data and os.path.exists(cache_file):
        logger.info(f"Loading ROI cache from {cache_file}")
        try:
            return pd.read_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute
    logger.info("Generating ROI cache (Logical-Consensus)...")

    roi_gen = ROIGenerator()
    results = []

    total = len(metadata_df)
    for idx, row in metadata_df.iterrows():
        if idx % 50 == 0:
            logger.info(f"Processing subject {idx}/{total}")

        subject_id = row["BraTS21ID"]
        paths = roi_gen.process_subject(row)
        paths["BraTS21ID"] = subject_id
        results.append(paths)

    # 3. Create DataFrame
    df_cache = pd.DataFrame(results)

    # Ensure BraTS21ID is int
    if "BraTS21ID" in df_cache.columns:
        df_cache["BraTS21ID"] = df_cache["BraTS21ID"].astype(int)

    # 4. Save
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df_cache.to_parquet(cache_file, index=False)
    logger.info(f"ROI cache saved to {cache_file}")

    return df_cache
