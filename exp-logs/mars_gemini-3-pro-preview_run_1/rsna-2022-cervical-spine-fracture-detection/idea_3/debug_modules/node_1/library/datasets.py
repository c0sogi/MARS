import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import pydicom

from library.config import Config
from library.utils import (
    process_dicom,
    crop_image,
    load_dicom,
    save_cache,
    load_cache,
)


class SegmentationDataset(Dataset):
    """
    Dataset for training the Stage 1 Localizer (U-Net).
    Provides (image, mask) pairs resized to LOCALIZER_IMG_SIZE.
    Uses Bounding Boxes to generate binary masks.
    """

    def __init__(self, metadata_df, load_cached_data=True, transform=None):
        self.transform = transform
        self.cache_dir = os.path.join(Config.CACHE_DIR, "segmentation")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load Bounding Boxes
        self.bbox_df = pd.read_csv(Config.TRAIN_BBOXES_PATH)

        # Filter for studies in the provided metadata
        valid_uids = set(metadata_df["StudyInstanceUID"].unique())
        self.bbox_df = self.bbox_df[self.bbox_df["StudyInstanceUID"].isin(valid_uids)]

        # Store metadata for path lookups
        self.metadata_df = metadata_df.set_index("StudyInstanceUID")

        # Prepare or load cache index
        self.samples = self._prepare_data(load_cached_data)

    def _prepare_data(self, load_cached_data):
        index_path = "segmentation_index_bbox.parquet"

        # 1. Try loading cached index
        if load_cached_data:
            cached_df = load_cache(index_path, use_parquet=True)
            if cached_df is not None:
                if len(cached_df) > 0:
                    sample_path = os.path.join(
                        self.cache_dir, cached_df.iloc[0]["filename"]
                    )
                    if os.path.exists(sample_path):
                        return cached_df.to_dict("records")

        # 2. Generate data
        print("Generating Segmentation Dataset Cache from BBoxes...")
        samples = []

        # Group bboxes by Study and Slice
        grouped = self.bbox_df.groupby(["StudyInstanceUID", "slice_number"])

        for (study_uid, slice_num), group in grouped:
            try:
                # Get Image Path
                if study_uid not in self.metadata_df.index:
                    continue

                rel_path = self.metadata_df.loc[study_uid, "image_path"]
                img_dir = os.path.join(Config.INPUT_DIR, rel_path)

                # Construct DICOM path (assuming slice_number.dcm)
                dcm_path = os.path.join(img_dir, f"{slice_num}.dcm")
                if not os.path.exists(dcm_path):
                    continue

                save_name = f"{study_uid}_{slice_num}.npz"
                save_path = os.path.join(self.cache_dir, save_name)

                if load_cached_data and os.path.exists(save_path):
                    samples.append({"filename": save_name})
                    continue

                # Load Image
                image = process_dicom(dcm_path)  # 512x512

                # Create Mask from BBoxes
                mask = np.zeros((512, 512), dtype=np.float32)

                for _, row in group.iterrows():
                    x, y = int(row["x"]), int(row["y"])
                    w, h = int(row["width"]), int(row["height"])

                    # Clip to image boundaries
                    x = max(0, min(x, 512))
                    y = max(0, min(y, 512))
                    w = max(0, min(w, 512 - x))
                    h = max(0, min(h, 512 - y))

                    if w > 0 and h > 0:
                        mask[y : y + h, x : x + w] = 1.0

                # Resize to Localizer size
                tgt_h, tgt_w = Config.LOCALIZER_IMG_SIZE
                image_resized = cv2.resize(
                    image, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR
                )
                mask_resized = cv2.resize(
                    mask, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST
                )

                np.savez_compressed(save_path, image=image_resized, mask=mask_resized)
                samples.append({"filename": save_name})

            except Exception as e:
                print(f"Error processing {study_uid} slice {slice_num}: {e}")
                continue

        # Save index
        df_samples = pd.DataFrame(samples)
        save_cache(df_samples, index_path, use_parquet=True)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename = self.samples[idx]["filename"]
        path = os.path.join(self.cache_dir, filename)

        # Load data
        data = np.load(path)
        image = data["image"]
        mask = data["mask"]

        # To Tensor
        # Image: (H, W) -> (1, H, W)
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        # Mask: (H, W) -> (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        # Binarize mask (segmentation labels are 1-7 for C1-C7, we just want "Spine" vs "Background" for localization)
        # Or we can keep multi-class. The Localizer description says "predict center of mass", implying binary or heatmap.
        # Let's treat > 0 as spine.
        mask_tensor = (mask_tensor > 0).float()

        return image_tensor, mask_tensor


class SliceClassificationDataset(Dataset):
    """
    Dataset for training the Stage 2 Encoder.
    Provides 2.5D cropped slice stacks and binary labels.
    """

    def __init__(self, metadata_df, load_cached_data=True, transform=None):
        self.transform = transform
        self.metadata_df = metadata_df

        # Load Bounding Boxes
        bbox_df = pd.read_csv(Config.TRAIN_BBOXES_PATH)
        self.bbox_df = bbox_df

        # Prepare metadata mapping
        self.slice_data = self._prepare_data(load_cached_data)

    def _prepare_data(self, load_cached_data):
        cache_filename = "slice_classification_meta.parquet"

        # 1. Try loading cache
        if load_cached_data:
            df = load_cache(cache_filename, use_parquet=True)
            if df is not None:
                return df

        # 2. Generate Data
        print("Generating Slice Classification Metadata...")

        # Filter studies that have bounding boxes (we need reliable labels)
        valid_uids = set(self.bbox_df["StudyInstanceUID"].unique())
        # Intersect with provided metadata (train/val split)
        split_uids = set(self.metadata_df["StudyInstanceUID"].unique())
        target_uids = list(valid_uids.intersection(split_uids))

        records = []

        for uid in target_uids:
            # Get image directory
            row = self.metadata_df[self.metadata_df["StudyInstanceUID"] == uid].iloc[0]
            img_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

            # Get all slices
            dcm_files = glob.glob(os.path.join(img_dir, "*.dcm"))
            # Sort by slice number (assuming filename is number.dcm)
            # Some filenames might be complex, but typically in this dataset they are 'int.dcm'
            try:
                dcm_files.sort(
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                )
            except:
                dcm_files.sort()

            # Get BBoxes for this study
            study_bboxes = self.bbox_df[self.bbox_df["StudyInstanceUID"] == uid]

            # Create a lookup for bboxes: slice_number -> (x, y, w, h)
            bbox_lookup = {}
            for _, bbox in study_bboxes.iterrows():
                bbox_lookup[bbox["slice_number"]] = (
                    bbox["x"],
                    bbox["y"],
                    bbox["width"],
                    bbox["height"],
                )

            for i, dcm_path in enumerate(dcm_files):
                # Extract slice number from filename
                try:
                    slice_num = int(os.path.splitext(os.path.basename(dcm_path))[0])
                except:
                    slice_num = -1  # Should not happen based on dataset desc

                label = 0
                crop_center = (256, 256)  # Default center

                if slice_num in bbox_lookup:
                    label = 1
                    x, y, w, h = bbox_lookup[slice_num]
                    crop_center = (y + h / 2, x + w / 2)  # y, x format for numpy

                # Add to records
                # We store index i to easily retrieve prev/next slices
                records.append(
                    {
                        "StudyInstanceUID": uid,
                        "slice_path": dcm_path,
                        "slice_index": i,
                        "total_slices": len(dcm_files),
                        "label": label,
                        "center_y": crop_center[0],
                        "center_x": crop_center[1],
                    }
                )

        df = pd.DataFrame(records)
        save_cache(df, cache_filename, use_parquet=True)
        return df

    def __len__(self):
        return len(self.slice_data)

    def __getitem__(self, idx):
        row = self.slice_data.iloc[idx]

        # Determine paths for 2.5D stack (Current, -1, +1)
        # We need to find the paths for idx-1, idx, idx+1 within the same study
        # Since slice_data is a flat list, we can't just do idx-1 unless sorted.
        # But we stored 'slice_path' and 'slice_index'.
        # We can infer neighbors if we assume the directory structure is consistent.

        curr_path = row["slice_path"]
        dirname = os.path.dirname(curr_path)
        filename = os.path.basename(curr_path)
        try:
            slice_num = int(os.path.splitext(filename)[0])
            prev_path = os.path.join(dirname, f"{slice_num - 1}.dcm")
            next_path = os.path.join(dirname, f"{slice_num + 1}.dcm")
        except:
            # Fallback if naming is weird
            prev_path = curr_path
            next_path = curr_path

        # Load images
        # If neighbor doesn't exist (boundary), replicate current
        img_c = process_dicom(curr_path)

        if os.path.exists(prev_path):
            img_p = process_dicom(prev_path)
        else:
            img_p = img_c.copy()

        if os.path.exists(next_path):
            img_n = process_dicom(next_path)
        else:
            img_n = img_c.copy()

        # Crop
        cy, cx = row["center_y"], row["center_x"]
        crop_h, crop_w = Config.ENCODER_CROP_SIZE

        crop_c = crop_image(img_c, (cy, cx), (crop_h, crop_w))
        crop_p = crop_image(img_p, (cy, cx), (crop_h, crop_w))
        crop_n = crop_image(img_n, (cy, cx), (crop_h, crop_w))

        # Stack: (3, H, W)
        stack = np.stack([crop_p, crop_c, crop_n], axis=0)

        # To Tensor
        stack_tensor = torch.from_numpy(stack).float()
        label_tensor = torch.tensor(row["label"], dtype=torch.float)

        return stack_tensor, label_tensor


class FeatureSequenceDataset(Dataset):
    """
    Dataset for training the Stage 3 Aggregator (RNN).
    Provides sequences of features and patient-level labels.
    """

    def __init__(self, features_dict, metadata_df):
        """
        Args:
            features_dict (dict): {StudyUID: torch.Tensor(Seq_Len, Feature_Dim)}
            metadata_df (pd.DataFrame): Contains StudyInstanceUID and targets.
        """
        self.features_dict = features_dict
        self.metadata_df = metadata_df
        self.uids = metadata_df["StudyInstanceUID"].tolist()

        # Target columns: patient_overall, C1..C7
        self.target_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]

        # Load Features
        if uid in self.features_dict:
            features = self.features_dict[uid]  # (Seq_Len, Dim)
        else:
            # Fallback for missing features (e.g. during debug)
            features = torch.zeros((10, 2048))

        # Load Labels
        row = self.metadata_df.iloc[idx]
        labels = row[self.target_cols].values.astype(np.float32)
        labels_tensor = torch.from_numpy(labels)

        return features, labels_tensor


class InferenceDataset(Dataset):
    """
    Dataset for Inference.
    Loads full volumes for processing.
    """

    def __init__(self, metadata_df):
        self.metadata_df = metadata_df

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        uid = row["StudyInstanceUID"]
        img_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Get all slices sorted
        dcm_files = glob.glob(os.path.join(img_dir, "*.dcm"))
        try:
            dcm_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except:
            dcm_files.sort()

        # Load all images
        # Note: This might be memory intensive. If OOM occurs, we must switch to yielding chunks.
        # Given 220GB RAM, 1 scan is fine.
        images = []
        for f in dcm_files:
            # Load and Window (keep original 512x512 resolution)
            img = process_dicom(f)
            images.append(img)

        if len(images) == 0:
            volume = torch.zeros((1, 512, 512))
        else:
            volume = np.stack(images, axis=0)  # (Depth, H, W)
            volume = torch.from_numpy(volume).float()

        return volume, uid
