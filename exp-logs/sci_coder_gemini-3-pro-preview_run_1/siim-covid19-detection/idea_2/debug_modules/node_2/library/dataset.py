import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
import ast
from library.config import Config


def get_transforms(data):
    """
    Returns the Albumentations transformation pipeline for a given data split.
    """
    if data == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                # Explicitly setting mask_fill_value=0 ensures the mask is also zeroed out
                A.CoarseDropout(mask_fill_value=0, **Config.COARSE_DROPOUT_PARAMS),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class ChestXrayDataset(Dataset):
    def __init__(self, split="train", transform=None, load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Custom transforms.
            load_cached_data (bool): If True, attempts to load pre-processed .npy files.
        """
        self.split = split
        self.transform = transform or get_transforms(split)
        self.load_cached_data = load_cached_data

        # Define paths based on split
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
            self.cache_img_path = Config.CACHE_TRAIN_IMAGES
            self.cache_mask_path = Config.CACHE_TRAIN_MASKS
            self.cache_label_path = Config.CACHE_TRAIN_LABELS
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA_PATH
            self.cache_img_path = Config.CACHE_VAL_IMAGES
            self.cache_mask_path = Config.CACHE_VAL_MASKS
            self.cache_label_path = Config.CACHE_VAL_LABELS
        elif split == "test":
            self.metadata_path = Config.TEST_METADATA_PATH
            self.cache_img_path = Config.CACHE_TEST_IMAGES
            self.cache_dims_path = Config.CACHE_TEST_DIMS
            self.cache_mask_path = None
            self.cache_label_path = None
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load Metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

        if Config.DEBUG:
            self.df = self.df.head(Config.DEBUG_SAMPLE_SIZE)

        # Process and Load Data
        self._process_and_cache()

    def _process_and_cache(self):
        """
        Checks for cached .npy files. If found and allowed, loads them.
        Otherwise, reads DICOMs, processes them, saves to cache, and loads them.
        """
        # Check if all required cache files exist
        cache_exists = os.path.exists(self.cache_img_path)
        if self.split != "test":
            cache_exists = (
                cache_exists
                and os.path.exists(self.cache_mask_path)
                and os.path.exists(self.cache_label_path)
            )
        else:
            cache_exists = cache_exists and os.path.exists(self.cache_dims_path)

        # Load Cached Data
        if self.load_cached_data and cache_exists:
            print(f"[{self.split}] Loading cached data from {Config.WORKING_DIR}...")
            self.images = np.load(self.cache_img_path)

            if self.split != "test":
                self.masks = np.load(self.cache_mask_path)
                self.labels = np.load(self.cache_label_path)
            else:
                self.original_dims = pd.read_parquet(self.cache_dims_path).values

        # Process from Scratch
        else:
            print(
                f"[{self.split}] Processing raw DICOM data (Cache not found or disabled)..."
            )
            img_list = []
            mask_list = []
            label_list = []
            dim_list = []

            # Ensure output directory exists
            os.makedirs(Config.WORKING_DIR, exist_ok=True)

            for idx, row in self.df.iterrows():
                # 1. Read DICOM
                path = os.path.join(Config.INPUT_DIR, row["file_path"])
                try:
                    dcm = pydicom.dcmread(path)
                    img = dcm.pixel_array

                    # Fix Photometric Interpretation (ensure bone is white)
                    if (
                        hasattr(dcm, "PhotometricInterpretation")
                        and dcm.PhotometricInterpretation == "MONOCHROME1"
                    ):
                        img = np.max(img) - img
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    # Create empty placeholder if read fails
                    img = np.zeros(Config.IMG_SIZE, dtype=np.uint8)

                # 2. Normalize to 0-255 uint8
                if img.max() > img.min():
                    img = (img - img.min()) / (img.max() - img.min()) * 255.0
                else:
                    img = img * 0
                img = img.astype(np.uint8)

                h_orig, w_orig = img.shape

                # 3. Resize Image
                # cv2.resize uses (width, height)
                img_resized = cv2.resize(
                    img,
                    (Config.IMG_SIZE[1], Config.IMG_SIZE[0]),
                    interpolation=cv2.INTER_AREA,
                )
                img_list.append(img_resized)

                if self.split != "test":
                    # 4. Create Mask (Train/Val)
                    mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
                    if pd.notna(row["boxes"]):
                        try:
                            boxes = ast.literal_eval(row["boxes"])
                            for box in boxes:
                                x, y, w, h = (
                                    box["x"],
                                    box["y"],
                                    box["width"],
                                    box["height"],
                                )
                                x, y, w, h = int(x), int(y), int(w), int(h)
                                cv2.rectangle(mask, (x, y), (x + w, y + h), 1, -1)
                        except:
                            pass  # Keep mask as zeros if parsing fails

                    # Resize Mask (Nearest Neighbor to keep binary)
                    mask_resized = cv2.resize(
                        mask,
                        (Config.IMG_SIZE[1], Config.IMG_SIZE[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    mask_list.append(mask_resized)

                    # 5. Extract Labels
                    # Columns: Negative, Typical, Indeterminate, Atypical
                    cols = [
                        "Negative for Pneumonia",
                        "Typical Appearance",
                        "Indeterminate Appearance",
                        "Atypical Appearance",
                    ]
                    label_vec = row[cols].values.astype(np.float32)
                    label_list.append(label_vec)
                else:
                    # Store original dimensions for Test
                    dim_list.append([h_orig, w_orig])

            # Convert to numpy arrays and Save
            self.images = np.array(img_list)
            np.save(self.cache_img_path, self.images)

            if self.split != "test":
                self.masks = np.array(mask_list)
                self.labels = np.array(label_list)
                np.save(self.cache_mask_path, self.masks)
                np.save(self.cache_label_path, self.labels)
            else:
                self.original_dims = np.array(dim_list)
                pd.DataFrame(self.original_dims, columns=["h", "w"]).to_parquet(
                    self.cache_dims_path
                )

            print(
                f"[{self.split}] Processing complete. Cache saved to {Config.WORKING_DIR}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image
        img = self.images[idx]

        # Convert grayscale to RGB (3 channels) for backbone compatibility
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.split != "test":
            mask = self.masks[idx]
            label = self.labels[idx]

            if self.transform:
                # Albumentations handles image and mask simultaneously
                transformed = self.transform(image=img, mask=mask)
                img = transformed["image"]
                mask = transformed["mask"]

            # Mask preparation: (H, W) -> (1, H, W) and float for BCE Loss
            mask = mask.float().unsqueeze(0)

            return {
                "image": img,
                "mask": mask,
                "label": torch.tensor(label, dtype=torch.float32),
                "study_id": self.df.iloc[idx]["study_id"],
            }
        else:
            if self.transform:
                transformed = self.transform(image=img)
                img = transformed["image"]

            return {
                "image": img,
                "study_id": self.df.iloc[idx]["study_id"],
                "image_id": self.df.iloc[idx]["image_id"],
                "orig_dim": self.original_dims[idx],  # [height, width]
            }
