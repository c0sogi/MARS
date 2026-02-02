import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from library.config import Config
from library.utils import read_dicom, seed_everything


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
def get_transforms(stage, mode="train"):
    """
    Returns the albumentations transforms for the specific stage and mode.
    """
    if stage == "segmentation":
        # Stage 1: 256x256 Resize, simple augs
        if mode == "train":
            return A.Compose(
                [
                    A.Resize(Config.SEG_IMAGE_SIZE[0], Config.SEG_IMAGE_SIZE[1]),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                    ),
                    A.Normalize(mean=(Config.PIXEL_MEAN,), std=(Config.PIXEL_STD,)),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(Config.SEG_IMAGE_SIZE[0], Config.SEG_IMAGE_SIZE[1]),
                    A.Normalize(mean=(Config.PIXEL_MEAN,), std=(Config.PIXEL_STD,)),
                    ToTensorV2(),
                ]
            )

    elif stage == "classifier":
        # Stage 2: 256x256 Crop. Input is 4 channels (3 RGB + 1 Mask).
        # We assume input to transform is already cropped.
        if mode == "train":
            return A.Compose(
                [
                    A.Resize(Config.ENC_IMAGE_SIZE[0], Config.ENC_IMAGE_SIZE[1]),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5),
                    A.Normalize(
                        mean=[Config.PIXEL_MEAN] * Config.ENC_IN_CHANNELS,
                        std=[Config.PIXEL_STD] * Config.ENC_IN_CHANNELS,
                    ),
                    ToTensorV2(),
                ],
                additional_targets={"image": "image"},
            )
        else:
            return A.Compose(
                [
                    A.Resize(Config.ENC_IMAGE_SIZE[0], Config.ENC_IMAGE_SIZE[1]),
                    A.Normalize(
                        mean=[Config.PIXEL_MEAN] * Config.ENC_IN_CHANNELS,
                        std=[Config.PIXEL_STD] * Config.ENC_IN_CHANNELS,
                    ),
                    ToTensorV2(),
                ],
                additional_targets={"image": "image"},
            )

    return None


# -----------------------------------------------------------------------------
# Stage 1: Segmentation Dataset
# -----------------------------------------------------------------------------
class SegmentationDataset(Dataset):
    def __init__(self, df, transform=None, cache_masks=True):
        """
        df: DataFrame containing 'StudyInstanceUID', 'image_path'
        Only rows with has_bounding_box=True should be passed.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.samples = []

        # Load Bounding Boxes
        if os.path.exists(Config.BBOX_CSV):
            self.bbox_df = pd.read_csv(Config.BBOX_CSV)
        else:
            self.bbox_df = pd.DataFrame(
                columns=[
                    "StudyInstanceUID",
                    "slice_number",
                    "x",
                    "y",
                    "width",
                    "height",
                ]
            )

        self._prepare_samples()

    def _prepare_samples(self):
        """
        Creates a list of valid slices based on DICOM counts.
        """
        print(f"Preparing Segmentation Dataset with {len(self.df)} studies...")

        for idx, row in self.df.iterrows():
            uid = row["StudyInstanceUID"]
            img_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

            # Count slices by listing files
            try:
                # Fast count
                files = [f for f in os.listdir(img_dir) if f.endswith(".dcm")]
                num_slices = len(files)
            except Exception:
                num_slices = 0

            if num_slices == 0:
                continue

            # Add all slices for this volume
            # We add all slices, but only those with BBoxes will have positive masks.
            for z in range(num_slices):
                self.samples.append(
                    {
                        "StudyInstanceUID": uid,
                        "slice_idx": z,
                        "image_dir": img_dir,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        uid = sample["StudyInstanceUID"]
        z = sample["slice_idx"]
        img_dir = sample["image_dir"]

        # 1. Load Image
        dcm_path = os.path.join(img_dir, f"{z+1}.dcm")

        # Fallback if file naming isn't perfect
        if not os.path.exists(dcm_path):
            files = sorted(
                glob.glob(os.path.join(img_dir, "*.dcm")),
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
            )
            if z < len(files):
                dcm_path = files[z]

        img = read_dicom(dcm_path, Config.WINDOW_CENTER, Config.WINDOW_WIDTH)
        h, w = img.shape

        # 2. Generate Mask from BBox
        mask = np.zeros((h, w), dtype=np.float32)

        # Filter bboxes for this slice (z+1 because slice_number is 1-based)
        slice_bboxes = self.bbox_df[
            (self.bbox_df["StudyInstanceUID"] == uid)
            & (self.bbox_df["slice_number"] == z + 1)
        ]

        for _, row in slice_bboxes.iterrows():
            bx, by, bw, bh = row["x"], row["y"], row["width"], row["height"]
            x1, y1 = int(bx), int(by)
            x2, y2 = int(bx + bw), int(by + bh)
            # Clip
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Assign class 1 (we treat all fractures as generic "ROI")
            mask[y1:y2, x1:x2] = 1.0

        # 3. Transform
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            img = torch.from_numpy(img).unsqueeze(0)
            mask = torch.from_numpy(mask)

        return img, mask.long()


# -----------------------------------------------------------------------------
# Stage 2: Slice Classifier Dataset
# -----------------------------------------------------------------------------
class SliceClassifierDataset(Dataset):
    def __init__(self, samples, transform=None, mask_volume_dict=None):
        """
        samples: List of dicts with keys: 'StudyInstanceUID', 'slice_idx', 'label', 'image_dir', 'center_xy'
        mask_volume_dict: Optional dict {uid: 3d_array} for GT masks during training.
        """
        self.samples = samples
        self.transform = transform
        self.mask_volume_dict = mask_volume_dict or {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        uid = sample["StudyInstanceUID"]
        z = sample["slice_idx"]
        img_dir = sample["image_dir"]
        label = sample["label"]

        # 1. Load 3 Slices (z-1, z, z+1)
        slices = []
        for offset in [-1, 0, 1]:
            curr_z = z + offset
            dcm_path = os.path.join(img_dir, f"{curr_z+1}.dcm")

            if os.path.exists(dcm_path):
                img = read_dicom(dcm_path, Config.WINDOW_CENTER, Config.WINDOW_WIDTH)
            else:
                # Padding with black if out of bounds
                img = np.zeros((512, 512), dtype=np.float32)
            slices.append(img)

        # Stack: (H, W, 3)
        img_rgb = np.stack(slices, axis=-1)

        # 2. Get Mask (Alpha Channel)
        mask_slice = np.zeros((512, 512), dtype=np.float32)
        if uid in self.mask_volume_dict:
            vol = self.mask_volume_dict[uid]
            if 0 <= z < vol.shape[0]:
                # Binary bone mask (anything > 0)
                mask_slice = (vol[z] > 0).astype(np.float32)

        # 3. Determine Crop Center
        center_x, center_y = 256, 256
        if "center_xy" in sample and sample["center_xy"] is not None:
            center_x, center_y = sample["center_xy"]
        elif np.sum(mask_slice) > 0:
            ys, xs = np.where(mask_slice > 0)
            center_y, center_x = int(np.mean(ys)), int(np.mean(xs))

        # 4. Crop
        crop_h, crop_w = Config.ENC_IMAGE_SIZE
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(512, x1 + crop_w)
        y2 = min(512, y1 + crop_h)

        # Adjust boundary
        if x2 - x1 < crop_w:
            if x1 == 0:
                x2 = min(512, crop_w)
            else:
                x1 = max(0, 512 - crop_w)
        if y2 - y1 < crop_h:
            if y1 == 0:
                y2 = min(512, crop_h)
            else:
                y1 = max(0, 512 - crop_h)

        img_crop = img_rgb[y1:y2, x1:x2, :]
        mask_crop = mask_slice[y1:y2, x1:x2]

        # Concatenate: (H, W, 4)
        input_tensor = np.concatenate([img_crop, mask_crop[:, :, np.newaxis]], axis=-1)

        # 5. Transform
        if self.transform:
            augmented = self.transform(image=input_tensor)
            input_tensor = augmented["image"]  # (4, H, W)
        else:
            input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1)  # (4, H, W)

        return input_tensor.float(), torch.tensor(label, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Stage 3: Sequence Dataset
# -----------------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, df, features_dict, anatomical_ids_dict=None):
        """
        df: DataFrame with StudyInstanceUID and patient-level targets.
        features_dict: {uid: np.array(N, Feature_Dim)}
        anatomical_ids_dict: {uid: np.array(N,)} - integers 0-7
        """
        self.df = df
        self.features_dict = features_dict
        self.anatomical_ids_dict = anatomical_ids_dict

        self.targets = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row["StudyInstanceUID"]

        labels = row[self.targets].values.astype(np.float32)

        if uid in self.features_dict:
            feats = self.features_dict[uid]
        else:
            feats = np.zeros((10, Config.ENC_FEATURE_DIM), dtype=np.float32)

        if self.anatomical_ids_dict and uid in self.anatomical_ids_dict:
            anat_ids = self.anatomical_ids_dict[uid]
        else:
            anat_ids = np.zeros(feats.shape[0], dtype=np.longlong)

        # Pad / Truncate
        seq_len = feats.shape[0]
        max_len = Config.AGG_MAX_SEQ_LEN

        if seq_len > max_len:
            start = (seq_len - max_len) // 2
            feats = feats[start : start + max_len]
            anat_ids = anat_ids[start : start + max_len]
            mask = np.ones(max_len, dtype=np.float32)
        else:
            pad_len = max_len - seq_len
            feats = np.pad(feats, ((0, pad_len), (0, 0)), mode="constant")
            anat_ids = np.pad(anat_ids, (0, pad_len), mode="constant")
            mask = np.concatenate([np.ones(seq_len), np.zeros(pad_len)]).astype(
                np.float32
            )

        return (
            torch.from_numpy(feats).float(),
            torch.from_numpy(anat_ids).long(),
            torch.from_numpy(mask).float(),
            torch.from_numpy(labels).float(),
        )


# -----------------------------------------------------------------------------
# Data Loading Factory
# -----------------------------------------------------------------------------
def get_dataloaders(stage, df_train, df_val, load_cached_data=True):
    """
    Factory function to create dataloaders for specific stages.
    """

    if stage == "segmentation":
        # Use bounding boxes instead of segmentations due to missing nibabel
        train_seg_df = df_train[df_train["has_bounding_box"]].copy()
        val_seg_df = df_val[df_val["has_bounding_box"]].copy()

        train_ds = SegmentationDataset(
            train_seg_df,
            transform=get_transforms("segmentation", "train"),
            cache_masks=False,
        )
        val_ds = SegmentationDataset(
            val_seg_df,
            transform=get_transforms("segmentation", "val"),
            cache_masks=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.SEG_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.SEG_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        return train_loader, val_loader

    elif stage == "classifier":
        # Load Bounding Boxes
        bbox_df = pd.read_csv(Config.BBOX_CSV)

        # Filter train/val to those with segmentation for high-quality training
        train_uids = set(df_train[df_train["has_segmentation"]]["StudyInstanceUID"])
        val_uids = set(df_val[df_val["has_segmentation"]]["StudyInstanceUID"])

        def build_samples(uids, df_meta):
            samples = []
            # 1. Positive Samples
            pos_bbox = bbox_df[bbox_df["StudyInstanceUID"].isin(uids)]
            for _, row in pos_bbox.iterrows():
                samples.append(
                    {
                        "StudyInstanceUID": row["StudyInstanceUID"],
                        "slice_idx": row["slice_number"] - 1,
                        "label": 1,
                        "image_dir": os.path.join(
                            Config.INPUT_DIR, f"train_images/{row['StudyInstanceUID']}"
                        ),
                    }
                )
            # 2. Negative Samples (Simple sampling: 2 negatives per patient)
            for uid in uids:
                img_dir = os.path.join(Config.INPUT_DIR, f"train_images/{uid}")
                try:
                    n_slices = len(glob.glob(os.path.join(img_dir, "*.dcm")))
                    if n_slices == 0:
                        continue
                except:
                    continue

                frac_slices = set(
                    pos_bbox[pos_bbox["StudyInstanceUID"] == uid]["slice_number"] - 1
                )
                count = 0
                for _ in range(20):  # try 20 times to find 2 negatives
                    if count >= 2:
                        break
                    z = np.random.randint(0, n_slices)
                    if z not in frac_slices:
                        samples.append(
                            {
                                "StudyInstanceUID": uid,
                                "slice_idx": z,
                                "label": 0,
                                "image_dir": img_dir,
                            }
                        )
                        count += 1
            return samples

        train_samples = build_samples(train_uids, df_train)
        val_samples = build_samples(val_uids, df_val)

        # GT masks loading removed due to missing nibabel.
        # SliceClassifierDataset will default to center crop.
        mask_dict = {}

        train_ds = SliceClassifierDataset(
            train_samples,
            transform=get_transforms("classifier", "train"),
            mask_volume_dict=mask_dict,
        )
        val_ds = SliceClassifierDataset(
            val_samples,
            transform=get_transforms("classifier", "val"),
            mask_volume_dict=mask_dict,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.ENC_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.ENC_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )
        return train_loader, val_loader

    elif stage == "aggregator":
        if not os.path.exists(Config.TRAIN_FEATURES_CACHE):
            print("Cached features not found. Cannot create Aggregator DataLoader.")
            return None, None

        train_feats = np.load(Config.TRAIN_FEATURES_CACHE, allow_pickle=True).item()
        val_feats = np.load(Config.VAL_FEATURES_CACHE, allow_pickle=True).item()

        # Assuming anatomical IDs are available or placeholder
        train_anat = {}
        val_anat = {}

        train_ds = SequenceDataset(df_train, train_feats, train_anat)
        val_ds = SequenceDataset(df_val, val_feats, val_anat)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.AGG_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.AGG_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )
        return train_loader, val_loader

    return None, None
