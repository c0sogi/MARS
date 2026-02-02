import os
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

from library.config import Config
from library.utils import (
    load_dicom_array,
    load_reoriented_segmentation,
    load_scan_volume,
)


class SegmentationDataset(Dataset):
    """
    Stage 1 Dataset: 2D Semantic Segmentation.
    Loads DICOM slices and corresponding NIFTI masks.
    """

    def __init__(self, split="train", transform=None, load_cached_data=True):
        self.split = split
        self.transform = transform

        # Load Metadata
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = Config.VAL_METADATA_PATH
        else:
            raise ValueError("Split must be 'train' or 'val'")

        self.df = pd.read_csv(meta_path)
        # Filter for studies with segmentation
        self.df = self.df[self.df["has_segmentation"] == True].reset_index(drop=True)

        # Cache directory setup
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Build or Load Index
        self.index_data = self._build_index(load_cached_data)

        if Config.DEBUG:
            self.index_data = self.index_data[: Config.DEBUG_SAMPLE_SIZE]

    def _build_index(self, load_cached_data):
        cache_file = os.path.join(
            self.cache_dir, f"segmentation_index_{self.split}.parquet"
        )

        if load_cached_data and os.path.exists(cache_file):
            return pd.read_parquet(cache_file).to_dict("records")

        print(f"Building Segmentation Index for {self.split}...")
        index_list = []

        for _, row in self.df.iterrows():
            study_uid = row["StudyInstanceUID"]
            # Path to NIFTI
            seg_path = os.path.join(Config.INPUT_DIR, row["segmentation_path"])

            # Load NIFTI to check valid slices
            # We cache the NIFTI volume as .npy for faster random access later
            vol_cache_path = os.path.join(self.cache_dir, f"{study_uid}_mask.npy")

            if load_cached_data and os.path.exists(vol_cache_path):
                mask_vol = np.load(vol_cache_path)
            else:
                mask_vol = load_reoriented_segmentation(seg_path)
                if mask_vol is None:
                    continue
                # Save for future fast access
                np.save(vol_cache_path, mask_vol)

            # Identify slices with annotations (C1-C7)
            # mask_vol shape: (D, H, W)
            # We keep slices that have any label > 0
            # To reduce dataset size, we might skip empty slices or subsample them

            num_slices = mask_vol.shape[0]

            # Get list of DICOM files to match indices
            study_img_dir = os.path.join(Config.INPUT_DIR, row["image_path"])
            dcm_files = sorted(
                glob.glob(os.path.join(study_img_dir, "*.dcm")),
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
            )

            # Ensure mask and dcm count match (basic check)
            # If mismatch, we take the minimum length
            limit = min(num_slices, len(dcm_files))

            for z in range(limit):
                # Check if slice has content
                if np.any(mask_vol[z] > 0):
                    index_list.append(
                        {
                            "StudyInstanceUID": study_uid,
                            "slice_idx": z,
                            "dcm_path": dcm_files[z],
                            "mask_cache_path": vol_cache_path,
                        }
                    )
                # Optionally add some background slices?
                # For now, focus on anatomical regions to stabilize training.

        df_index = pd.DataFrame(index_list)
        df_index.to_parquet(cache_file)
        return index_list

    def __len__(self):
        return len(self.index_data)

    def __getitem__(self, idx):
        item = self.index_data[idx]

        # 1. Load Image
        img = load_dicom_array(
            item["dcm_path"],
            size=Config.STAGE1_IMAGE_SIZE,
            window_center=Config.WINDOW_CENTER,
            window_width=Config.WINDOW_WIDTH,
        )
        # Add channel dim: (1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        # 2. Load Mask
        # Load volume from npy cache (memmap mode is faster for single slice)
        mask_vol = np.load(item["mask_cache_path"], mmap_mode="r")
        mask_slice = mask_vol[item["slice_idx"]].copy()  # (H, W)

        # Resize mask to match image
        if mask_slice.shape != Config.STAGE1_IMAGE_SIZE:
            mask_slice = cv2.resize(
                mask_slice,
                (Config.STAGE1_IMAGE_SIZE[1], Config.STAGE1_IMAGE_SIZE[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        mask_tensor = torch.from_numpy(mask_slice).long()

        return img_tensor, mask_tensor


class SliceClassificationDataset(Dataset):
    """
    Stage 2 Dataset: 2.5D Binary Classification (Fracture vs No Fracture).
    Input: Stack of 3 slices + Mask.
    """

    def __init__(self, split="train", load_cached_data=True):
        self.split = split

        # Load Metadata
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = Config.VAL_METADATA_PATH
        else:
            # For inference, we'd use a different loader or logic
            raise ValueError(
                "SliceClassificationDataset is for training/validation only."
            )

        self.df_meta = pd.read_csv(meta_path)
        self.df_bbox = (
            pd.read_csv(Config.BBOX_PATH)
            if os.path.exists(Config.BBOX_PATH)
            else pd.DataFrame()
        )

        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.samples = self._prepare_samples(load_cached_data)

        if Config.DEBUG:
            self.samples = self.samples[: Config.DEBUG_SAMPLE_SIZE]

    def _prepare_samples(self, load_cached_data):
        cache_file = os.path.join(
            self.cache_dir, f"slice_classification_meta_{self.split}.parquet"
        )

        if load_cached_data and os.path.exists(cache_file):
            return pd.read_parquet(cache_file).to_dict("records")

        print(f"Preparing Slice Classification Samples for {self.split}...")

        samples = []
        valid_uids = set(self.df_meta["StudyInstanceUID"].unique())

        # 1. Positive Samples (From Bounding Boxes)
        # Filter bboxes for studies in this split
        if not self.df_bbox.empty:
            df_pos = self.df_bbox[self.df_bbox["StudyInstanceUID"].isin(valid_uids)]

            for _, row in df_pos.iterrows():
                samples.append(
                    {
                        "StudyInstanceUID": row["StudyInstanceUID"],
                        "slice_idx": row["slice_number"],
                        "label": 1.0,
                        "x": row["x"],
                        "y": row["y"],
                        "w": row["width"],
                        "h": row["height"],
                        "is_bbox": True,
                    }
                )

        # 2. Negative Samples (Random sampling)
        # We want roughly 1:1 or 1:2 ratio
        n_pos = len(samples)
        n_neg = max(n_pos, 100)  # Ensure at least some negatives

        # Collect all possible slices (simplified: just random valid slices)
        # We don't have a master list of all slices, so we iterate studies and pick random slices
        neg_samples = []
        studies = self.df_meta["StudyInstanceUID"].tolist()

        while len(neg_samples) < n_neg:
            uid = random.choice(studies)
            # Estimate slice count (avg ~300) or check file count if needed.
            # For speed, assume 0-200 is safe or check directory.
            # Let's check directory to be safe.
            row = self.df_meta[self.df_meta["StudyInstanceUID"] == uid].iloc[0]
            study_path = os.path.join(Config.INPUT_DIR, row["image_path"])
            if not os.path.exists(study_path):
                continue

            # Just count files once per study if we were optimizing, but here random try
            # We'll just pick a random number between 1 and 100 (most scans have these)
            # Better: list files
            files = os.listdir(study_path)
            if not files:
                continue
            slice_idx = random.randint(1, len(files))

            # Check if this slice is in positives
            is_pos = False
            for p in samples:
                if p["StudyInstanceUID"] == uid and p["slice_idx"] == slice_idx:
                    is_pos = True
                    break

            if not is_pos:
                neg_samples.append(
                    {
                        "StudyInstanceUID": uid,
                        "slice_idx": slice_idx,
                        "label": 0.0,
                        "x": 0,
                        "y": 0,
                        "w": 0,
                        "h": 0,  # Dummy
                        "is_bbox": False,
                    }
                )

        all_samples = samples + neg_samples
        random.shuffle(all_samples)

        df_samples = pd.DataFrame(all_samples)
        df_samples.to_parquet(cache_file)
        return all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        uid = item["StudyInstanceUID"]
        center_slice_idx = int(item["slice_idx"])

        # Get paths
        # We need to find the image path from metadata
        # Optimization: Create a UID->Path map in __init__?
        # For now, reconstruct path
        # Assuming train_images structure
        study_dir = os.path.join(Config.TRAIN_IMAGES_DIR, uid)

        # Load 3 slices: z-1, z, z+1
        imgs = []
        for offset in [-1, 0, 1]:
            s_idx = center_slice_idx + offset
            # Construct filename (e.g., "10.dcm")
            # Note: filenames are not always padded. "1.dcm", "100.dcm"
            path = os.path.join(study_dir, f"{s_idx}.dcm")

            # Load full slice
            img = load_dicom_array(
                path,
                size=Config.ORIGINAL_SIZE,  # Load original 512x512
                window_center=Config.WINDOW_CENTER,
                window_width=Config.WINDOW_WIDTH,
            )
            imgs.append(img)

        # Stack: (H, W, 3)
        img_stack = np.stack(imgs, axis=-1)

        # Load Mask (4th channel)
        # If segmentation exists in cache, use it. Else zeros.
        mask_cache_path = os.path.join(self.cache_dir, f"{uid}_mask.npy")
        mask_layer = np.zeros(Config.ORIGINAL_SIZE, dtype=np.float32)

        if os.path.exists(mask_cache_path):
            try:
                # Load mask volume
                mask_vol = np.load(mask_cache_path, mmap_mode="r")
                # Check bounds
                if 0 <= center_slice_idx < mask_vol.shape[0]:
                    m = mask_vol[center_slice_idx]
                    # Binarize: Any vertebra = 1
                    m = (m > 0).astype(np.float32)
                    # Resize if needed (mask_vol might be original size or not?
                    # load_reoriented_segmentation returns original dims usually)
                    if m.shape != Config.ORIGINAL_SIZE:
                        m = cv2.resize(
                            m,
                            (Config.ORIGINAL_SIZE[1], Config.ORIGINAL_SIZE[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    mask_layer = m
            except:
                pass

        # Combine: (H, W, 4)
        input_vol = np.dstack([img_stack, mask_layer])

        # Crop
        # If bbox, crop around bbox. Else center crop.
        h, w = Config.ORIGINAL_SIZE
        crop_h, crop_w = Config.STAGE2_CROP_SIZE

        if item["is_bbox"]:
            # BBox center
            cx = item["x"] + item["w"] / 2
            cy = item["y"] + item["h"] / 2
        else:
            # Image center
            cx = w / 2
            cy = h / 2

        # Calculate top-left
        x1 = int(cx - crop_w / 2)
        y1 = int(cy - crop_h / 2)

        # Clamp
        x1 = max(0, min(x1, w - crop_w))
        y1 = max(0, min(y1, h - crop_h))

        crop = input_vol[y1 : y1 + crop_h, x1 : x1 + crop_w, :]

        # To Tensor: (C, H, W)
        # Transpose from (H, W, C) -> (C, H, W)
        crop_tensor = torch.from_numpy(crop.transpose(2, 0, 1)).float()

        label_tensor = torch.tensor([item["label"]], dtype=torch.float32)

        return crop_tensor, label_tensor


class PatientSequenceDataset(Dataset):
    """
    Stage 3 Dataset: Sequence Aggregation.
    Loads pre-computed features (Stage 1 + Stage 2 outputs) and Patient Labels.
    """

    def __init__(self, split="train", load_cached_data=True):
        self.split = split

        # Metadata
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = Config.VAL_METADATA_PATH
        elif split == "test":
            meta_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError("Invalid split")

        self.df = pd.read_csv(meta_path)
        self.feature_dir = os.path.join(Config.WORKING_DIR, "cache", "features")

        # Filter studies that have features generated
        # If features don't exist, we can't train/predict on them
        # In a real pipeline, we'd ensure they exist.
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            uid = row["StudyInstanceUID"]
            fpath = os.path.join(self.feature_dir, f"{uid}.npy")
            if os.path.exists(fpath):
                self.valid_indices.append(idx)

        if len(self.valid_indices) == 0 and split != "test":
            print("Warning: No feature files found. Dataset is empty.")

        if Config.DEBUG:
            self.valid_indices = self.valid_indices[: Config.DEBUG_SAMPLE_SIZE]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        uid = row["StudyInstanceUID"]

        # Load Features: (SeqLen, FeatureDim)
        # FeatureDim = 512 (Local) + 512 (Global) + 8 (Probs) = 1032
        feature_path = os.path.join(self.feature_dir, f"{uid}.npy")
        features = np.load(feature_path)

        # Convert to Tensor
        features_tensor = torch.from_numpy(features).float()

        # Split features for the model inputs
        # Assuming concatenation order: [Local(512), Global(512), Probs(8)]
        local_emb = features_tensor[:, : Config.STAGE2_EMBEDDING_DIM]
        global_ctx = features_tensor[
            :, Config.STAGE2_EMBEDDING_DIM : Config.STAGE2_EMBEDDING_DIM + 512
        ]
        anat_probs = features_tensor[:, -8:]

        # Load Labels
        if self.split in ["train", "val"]:
            # C1-C7, patient_overall
            # Columns in metadata: C1, C2, ..., C7, patient_overall
            # We need to stack them in order
            labels = []
            for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]:
                labels.append(row[c])
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
            return local_emb, global_ctx, anat_probs, labels_tensor
        else:
            # Test set - no labels, return dummy or just features
            return local_emb, global_ctx, anat_probs, row["StudyInstanceUID"]

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable sequence lengths.
        Returns packed sequences or padded sequences.
        For simplicity with Bi-GRU, we can just return a list or pad here.
        Since batch_size might be > 1, padding is required.
        """
        # Sort by sequence length (descending) for pack_padded_sequence if used
        batch.sort(key=lambda x: x[0].shape[0], reverse=True)

        local_embs, global_ctxs, anat_probs, targets = zip(*batch)

        # Pad sequences
        lengths = [x.shape[0] for x in local_embs]
        max_len = max(lengths)

        # Helper to pad
        def pad_tensor(tensors, dim):
            padded = torch.zeros(len(tensors), max_len, dim)
            for i, t in enumerate(tensors):
                end = t.shape[0]
                padded[i, :end, :] = t
            return padded

        local_padded = pad_tensor(local_embs, Config.STAGE2_EMBEDDING_DIM)
        global_padded = pad_tensor(global_ctxs, 512)
        anat_padded = pad_tensor(anat_probs, 8)

        if isinstance(targets[0], torch.Tensor):
            targets_stacked = torch.stack(targets)
            return local_padded, global_padded, anat_padded, targets_stacked, lengths
        else:
            return local_padded, global_padded, anat_padded, list(targets), lengths
