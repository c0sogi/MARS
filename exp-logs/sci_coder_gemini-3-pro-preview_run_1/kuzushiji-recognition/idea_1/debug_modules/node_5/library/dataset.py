import math
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from library.config import Config
from library.utils import (
    load_unicode_map,
    preprocess_metadata,
    get_affine_transform,
    affine_transform,
    gaussian_radius,
    draw_gaussian,
)


class KuzushijiDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True, transform=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load processed metadata from cache.
            transform: Optional transforms (not used in this implementation as we do manual affine).
        """
        self.split = split
        self.transform = transform

        # Load Unicode Mapping
        self.char_to_id, self.id_to_char = load_unicode_map()

        # Determine Metadata Path
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = Config.VAL_METADATA_PATH
        elif split == "test":
            meta_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load and Preprocess Metadata (Handles caching internally via utils)
        self.data = preprocess_metadata(
            meta_path,
            self.char_to_id,
            load_cached_data=load_cached_data,
            split_name=split,
        )

        # Hyperparameters
        self.input_size = Config.IMG_SIZE
        self.output_size = self.input_size // 4
        self.num_classes = Config.NUM_CLASSES

        # Normalization Stats (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(Config.INPUT_DIR, item["file_path"])

        # 1. Load Image
        img = cv2.imread(img_path)
        if img is None:
            # Fallback for robustness
            img = np.zeros((512, 512, 3), dtype=np.uint8)

        height, width = img.shape[:2]

        # 2. Prepare Affine Transform (Letterbox Resize)
        c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        s = max(height, width) * 1.0

        # Input transform (to 1024x1024)
        trans_input = get_affine_transform(c, s, 0, [self.input_size, self.input_size])

        # Apply transform
        inp = cv2.warpAffine(
            img,
            trans_input,
            (self.input_size, self.input_size),
            flags=cv2.INTER_LINEAR,
        )

        # 3. Normalize
        inp = inp.astype(np.float32) / 255.0
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)  # HWC -> CHW

        # Base Return Dictionary
        ret = {
            "input": torch.from_numpy(inp),
            "image_id": item["image_id"],
        }

        # If Test, return meta for inverse transform and exit
        if self.split == "test":
            ret["center"] = torch.from_numpy(c)
            ret["scale"] = torch.tensor(s, dtype=torch.float32)
            return ret

        # 4. Generate Targets (Train/Val)

        # Output transform (to 256x256)
        trans_output = get_affine_transform(
            c, s, 0, [self.output_size, self.output_size]
        )

        # Initialize Targets
        # Heatmap: [1, H, W]
        hm = np.zeros((1, self.output_size, self.output_size), dtype=np.float32)
        # Regression: [4, H, W] -> [ox, oy, log(w), log(h)]
        reg = np.zeros((4, self.output_size, self.output_size), dtype=np.float32)
        # Regression Mask: [1, H, W]
        reg_mask = np.zeros((1, self.output_size, self.output_size), dtype=np.float32)
        # Classification: [H, W] -> Long (Class IDs)
        cls_target = np.full((self.output_size, self.output_size), -1, dtype=np.int64)

        anns = item["annotations"]  # [class_id, x, y, w, h]

        for ann in anns:
            cls_id = int(ann[0])
            bbox = ann[1:]  # x, y, w, h
            x, y, w, h = bbox

            # Transform BBox to Output Space
            # Transform Top-Left and Bottom-Right
            rect = np.array([[x, y], [x + w, y + h]], dtype=np.float32)
            pt1 = affine_transform(rect[0], trans_output)
            pt2 = affine_transform(rect[1], trans_output)

            x_out, y_out = pt1
            w_out = pt2[0] - x_out
            h_out = pt2[1] - y_out

            if h_out > 0 and w_out > 0:
                # Gaussian Radius
                radius = gaussian_radius((math.ceil(h_out), math.ceil(w_out)))
                radius = max(0, int(radius))

                # Center Point
                ct = np.array([x_out + w_out / 2, y_out + h_out / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                # Bounds Check
                if (
                    ct_int[0] >= 0
                    and ct_int[0] < self.output_size
                    and ct_int[1] >= 0
                    and ct_int[1] < self.output_size
                ):

                    # 1. Draw Gaussian on Heatmap
                    draw_gaussian(hm[0], ct_int, radius)

                    # 2. Classification Target
                    cls_target[ct_int[1], ct_int[0]] = cls_id

                    # 3. Regression Target
                    # Offsets
                    reg[0, ct_int[1], ct_int[0]] = ct[0] - ct_int[0]
                    reg[1, ct_int[1], ct_int[0]] = ct[1] - ct_int[1]
                    # Dimensions (Log Scale)
                    reg[2, ct_int[1], ct_int[0]] = np.log(w_out)
                    reg[3, ct_int[1], ct_int[0]] = np.log(h_out)

                    # 4. Regression Mask
                    reg_mask[0, ct_int[1], ct_int[0]] = 1

        ret.update(
            {
                "hm": torch.from_numpy(hm),
                "cls_target": torch.from_numpy(cls_target),
                "reg_target": torch.from_numpy(reg),
                "reg_mask": torch.from_numpy(reg_mask),
            }
        )

        return ret
