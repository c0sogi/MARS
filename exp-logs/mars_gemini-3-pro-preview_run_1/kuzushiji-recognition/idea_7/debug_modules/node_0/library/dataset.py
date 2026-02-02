import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import LabelEncoder, gaussian_radius, draw_umich_gaussian


class KuzushijiDataset(Dataset):
    def __init__(
        self,
        mode="train",
        load_cached_data=True,
        transform=None,
        max_objs=None,
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached annotation data.
            transform (A.Compose): Optional custom augmentation pipeline.
            max_objs (int): Maximum number of objects per image (for tensor padding).
        """
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.max_objs = max_objs if max_objs is not None else Config.MAX_DETECTIONS
        self.down_ratio = 4
        self.output_size = Config.IMG_SIZE // self.down_ratio

        # Select Metadata File
        if self.mode == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
        elif self.mode == "val":
            self.metadata_path = Config.VAL_METADATA_PATH
        else:
            self.metadata_path = Config.TEST_METADATA_PATH

        # Load Metadata
        self.df = pd.read_csv(self.metadata_path)

        # Initialize Label Encoder
        self.le = LabelEncoder().fit(load_cached_data=load_cached_data)

        # Pre-process Annotations (with Caching)
        if self.mode in ["train", "val"]:
            self.annotations, self.img_indices = self._process_and_cache_annotations()

        # Define Transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_transforms()

    def _get_transforms(self):
        """
        Returns the Albumentations transform pipeline based on the mode.
        """
        transforms_list = [
            A.LongestMaxSize(max_size=Config.IMG_SIZE),
            A.PadIfNeeded(
                min_height=Config.IMG_SIZE,
                min_width=Config.IMG_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
        ]

        # Geometric Augmentations for Training
        if self.mode == "train":
            transforms_list.extend(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    )
                ]
            )

        # Normalization and Tensor Conversion
        transforms_list.extend(
            [A.Normalize(mean=Config.MEAN, std=Config.STD), ToTensorV2()]
        )

        return A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(
                format="coco", label_fields=["class_labels"], min_visibility=0.25
            ),
        )

    def _process_and_cache_annotations(self):
        """
        Parses annotations and implements caching using .npy files.
        Returns:
            all_anns (np.ndarray): Array of shape (N_total, 6) -> [img_idx, cls_idx, x, y, w, h]
            img_indices (np.ndarray): Array of shape (N_images + 1) -> [start_idx, ...]
        """
        cache_dir = Config.WORK_DIR
        os.makedirs(cache_dir, exist_ok=True)

        anns_path = os.path.join(cache_dir, f"{self.mode}_anns.npy")
        indices_path = os.path.join(cache_dir, f"{self.mode}_indices.npy")

        # 1. Try Loading Cache
        if (
            self.load_cached_data
            and os.path.exists(anns_path)
            and os.path.exists(indices_path)
        ):
            try:
                all_anns = np.load(anns_path)
                img_indices = np.load(indices_path)
                # Verify consistency
                if len(img_indices) == len(self.df) + 1:
                    return all_anns, img_indices
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing...")

        # 2. Compute from Scratch
        all_anns_list = []
        img_indices_list = [0]
        current_idx = 0

        for idx, row in self.df.iterrows():
            label_str = row.get("labels", "")
            if (
                pd.isna(label_str)
                or not isinstance(label_str, str)
                or not label_str.strip()
            ):
                img_indices_list.append(current_idx)
                continue

            parts = label_str.strip().split(" ")
            # Format: Unicode X Y W H ...
            num_anns = len(parts) // 5

            for i in range(num_anns):
                base = i * 5
                char = parts[base]
                try:
                    x = float(parts[base + 1])
                    y = float(parts[base + 2])
                    w = float(parts[base + 3])
                    h = float(parts[base + 4])

                    cls_idx = self.le.transform(char)
                    if cls_idx != -1:
                        all_anns_list.append([idx, cls_idx, x, y, w, h])
                        current_idx += 1
                except ValueError:
                    continue

            img_indices_list.append(current_idx)

        if all_anns_list:
            all_anns = np.array(all_anns_list, dtype=np.float32)
        else:
            all_anns = np.zeros((0, 6), dtype=np.float32)

        img_indices = np.array(img_indices_list, dtype=np.int32)

        # 3. Save to Cache
        np.save(anns_path, all_anns)
        np.save(indices_path, img_indices)

        return all_anns, img_indices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"]

        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(img_path)
        if img is None:
            # Fallback for missing images (should not happen based on EDA)
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]

        # Get Annotations
        bboxes = []
        class_labels = []

        if self.mode in ["train", "val"]:
            start_pos = self.img_indices[idx]
            end_pos = self.img_indices[idx + 1]
            if end_pos > start_pos:
                # Extract [cls_idx, x, y, w, h]
                # Cache stores [img_idx, cls_idx, x, y, w, h]
                anns = self.annotations[start_pos:end_pos]
                for ann in anns:
                    # ann: img_idx, cls_idx, x, y, w, h
                    cls_idx = int(ann[1])
                    x, y, w, h = ann[2], ann[3], ann[4], ann[5]
                    bboxes.append([x, y, w, h])
                    class_labels.append(cls_idx)

        # Apply Augmentations
        # Albumentations expects [x, y, w, h] for coco format
        transformed = self.transform(
            image=img, bboxes=bboxes, class_labels=class_labels
        )
        img_tensor = transformed["image"]
        aug_bboxes = transformed["bboxes"]
        aug_labels = transformed["class_labels"]

        # Initialize Targets
        # hm: (1, H, W) - Class Agnostic Heatmap for detection
        hm = np.zeros((1, self.output_size, self.output_size), dtype=np.float32)

        # Dense targets for CenterNet
        ind = np.zeros((self.max_objs), dtype=np.int64)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        cls_ids = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        num_objs = min(len(aug_bboxes), self.max_objs)

        for k in range(num_objs):
            bbox = aug_bboxes[k]
            cls_id = aug_labels[k]

            x, y, w, h = bbox

            # Map to output feature map scale
            # We use the center of the bounding box
            ct = np.array([(x + w / 2), (y + h / 2)], dtype=np.float32)
            ct = ct / self.down_ratio

            ct_int = ct.astype(np.int32)

            # Check bounds
            if (0 <= ct_int[0] < self.output_size) and (
                0 <= ct_int[1] < self.output_size
            ):
                # Draw Gaussian on heatmap (class agnostic -> channel 0)
                # Calculate radius based on object size in feature map
                h_out, w_out = h / self.down_ratio, w / self.down_ratio
                radius = gaussian_radius((math_h(h_out), math_h(w_out)))
                radius = max(0, int(radius))

                draw_umich_gaussian(hm[0], ct_int, radius)

                # Indices for gathering
                ind[k] = ct_int[1] * self.output_size + ct_int[0]

                # Regression targets
                reg[k] = ct - ct_int
                wh[k] = [w / self.down_ratio, h / self.down_ratio]

                # Classification target
                cls_ids[k] = cls_id
                reg_mask[k] = 1

        # Return Dictionary
        ret = {
            "image": img_tensor,
            "hm": torch.from_numpy(hm),
            "ind": torch.from_numpy(ind),
            "wh": torch.from_numpy(wh),
            "reg": torch.from_numpy(reg),
            "cls_id": torch.from_numpy(cls_ids),
            "reg_mask": torch.from_numpy(reg_mask),
            "image_id": img_id,
            "orig_shape": torch.tensor([orig_h, orig_w]),
            "label_str": row.get("labels", ""),
        }

        return ret


def math_h(x):
    # Helper to avoid math.ceil/floor import issues inside loop if not imported
    # Using int + 1 for ceil approximation or just passing float to gaussian_radius which handles it
    return x
