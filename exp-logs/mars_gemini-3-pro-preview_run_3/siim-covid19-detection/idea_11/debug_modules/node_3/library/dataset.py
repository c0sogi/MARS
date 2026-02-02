import os
import torch
import numpy as np
import albumentations as A
import cv2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_dicom, letterbox_resize, preprocess_metadata, collate_fn


class SIIMDataset(Dataset):
    """
    PyTorch Dataset for SIIM-FISABIO-RSNA COVID-19 Detection.
    Handles loading DICOMs, applying augmentations (LSJ), and formatting for DINO.
    """

    def __init__(self, split="train", load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached metadata.
        """
        self.split = split

        # Select metadata file based on split
        if split == "train":
            meta_path = Config.TRAIN_METADATA
        elif split == "val":
            meta_path = Config.VAL_METADATA
        else:
            meta_path = Config.TEST_METADATA

        # Load and process metadata (handles caching internally via utils)
        self.df = preprocess_metadata(
            meta_path, load_cached_data=load_cached_data, split_name=split
        )

        # Debugging: Sample subset
        if Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Define Augmentations
        # Train: Large Scale Jittering (LSJ) simulation + Flip
        # Test/Val: None (Resize is handled in __getitem__)
        if split == "train":
            self.aug = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    # LSJ: Resize 0.1 to 2.0.
                    # A.RandomScale(scale_limit=(-0.9, 1.0)) -> 0.1 to 2.0 relative scale
                    A.RandomScale(scale_limit=(-0.5, 1.0), p=0.5),
                    # ShiftScaleRotate combines scaling and cropping (shifting) to simulate jitter
                    A.ShiftScaleRotate(
                        shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5
                    ),
                    A.RandomBrightnessContrast(p=0.2),
                ],
                bbox_params=A.BboxParams(
                    format="coco", min_visibility=0.1, label_fields=["class_labels"]
                ),
            )
        else:
            self.aug = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = load_dicom(img_path)  # Returns (H, W, 3)

        # 2. Get Study Label
        # Default to -1 for test
        study_label = -1
        if self.split != "test":
            # Extract one-hot labels and convert to index
            # Config.STUDY_LABELS matches the column names in metadata
            labels = row[Config.STUDY_LABELS].values.astype(float)
            study_label = np.argmax(labels)

        # 3. Get Bounding Boxes
        boxes = []
        if self.split != "test":
            raw_boxes = row["boxes"]
            # preprocess_metadata ensures raw_boxes is a list of [x, y, w, h] or empty list
            if isinstance(raw_boxes, list):
                boxes = list(raw_boxes)
            elif hasattr(raw_boxes, "tolist"):
                boxes = raw_boxes.tolist()

        # 4. Apply Augmentations (Albumentations)
        # We need dummy class labels for A.Compose
        n_boxes = len(boxes)
        class_labels = [0] * n_boxes  # 0 is 'opacity'

        if self.aug:
            # Albumentations expects boxes in [x, y, w, h] (coco format)
            transformed = self.aug(image=img, bboxes=boxes, class_labels=class_labels)
            img = transformed["image"]
            boxes = transformed["bboxes"]

        # 5. Letterbox Resize (Fit to Config.IMG_SIZE)
        # Returns resized image, scale ratio, and padding (dw, dh)
        img_lb, ratio, (pad_w, pad_h) = letterbox_resize(
            img, target_size=Config.IMG_SIZE
        )

        # 6. Adjust Boxes for Letterbox and Normalize
        final_boxes = []
        for box in boxes:
            x, y, w, h = box

            # Apply scaling
            x *= ratio
            y *= ratio
            w *= ratio
            h *= ratio

            # Apply padding
            x += pad_w
            y += pad_h

            # Convert to [cx, cy, w, h] normalized (0-1)
            cx = (x + w / 2) / Config.IMG_SIZE
            cy = (y + h / 2) / Config.IMG_SIZE
            nw = w / Config.IMG_SIZE
            nh = h / Config.IMG_SIZE

            final_boxes.append([cx, cy, nw, nh])

        # 7. Prepare Tensors
        # Image: (H, W, 3) -> (3, H, W), Normalize 0-1
        img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0

        target = {}
        target["boxes"] = torch.tensor(final_boxes, dtype=torch.float32)

        if len(final_boxes) > 0:
            target["labels"] = torch.zeros(
                (len(final_boxes),), dtype=torch.int64
            )  # Class 0 for opacity
        else:
            # Empty tensors for no findings
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        target["study_label"] = torch.tensor(study_label, dtype=torch.int64)

        # Metadata for inference/eval
        target["image_id"] = row["image_id"]
        target["study_id"] = row["study_id"]
        # Original size (before letterbox) is needed to rescale boxes back to original image
        # Note: If augmentation was applied, this is the augmented size.
        # For valid/test (no aug), this is the DICOM size.
        target["orig_size"] = torch.tensor([img.shape[0], img.shape[1]])
        target["img_size"] = torch.tensor([Config.IMG_SIZE, Config.IMG_SIZE])

        return img_tensor, target
