import os
import cv2
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library import config, utils


class KuzushijiDataset(Dataset):
    def __init__(self, split, load_cached_data=True, debug=False):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load processed metadata from cache.
            debug (bool): If True, limits dataset size for debugging.
        """
        self.split = split
        self.debug = debug
        self.img_size = config.IMG_SIZE
        self.input_dir = config.INPUT_DIR
        self.max_objs = config.MAX_DETECTIONS

        # Define metadata paths
        if split == "train":
            self.meta_path = config.TRAIN_METADATA_PATH
        elif split == "val":
            self.meta_path = config.VAL_METADATA_PATH
        elif split == "test":
            self.meta_path = config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # Cache configuration
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.data_cache_path = os.path.join(self.cache_dir, f"data_{split}.npy")
        self.class_map_path = os.path.join(self.cache_dir, "class_map.npy")

        # Load data and class mapping
        self.data, self.class_to_idx = self._load_processed_data(load_cached_data)

        if self.debug:
            self.data = self.data[: config.DEBUG_SAMPLE_SIZE]

    def _load_processed_data(self, load_cached_data):
        """
        Loads processed metadata and class mapping, using cache if available.
        """
        # 1. Load or Build Class Mapping (Must be consistent across splits)
        class_to_idx = {}
        if load_cached_data and os.path.exists(self.class_map_path):
            class_to_idx = np.load(self.class_map_path, allow_pickle=True).item()
        else:
            # Build mapping from Training Metadata
            train_df = pd.read_csv(config.TRAIN_METADATA_PATH, keep_default_na=False)
            unique_chars = set()
            for labels in train_df["labels"]:
                if labels:
                    parts = labels.split()
                    # Labels format: Char X Y W H ...
                    for i in range(0, len(parts), 5):
                        unique_chars.add(parts[i])

            # Sort for determinism
            sorted_chars = sorted(list(unique_chars))
            class_to_idx = {c: i for i, c in enumerate(sorted_chars)}

            # Save to cache
            np.save(self.class_map_path, class_to_idx)

        # 2. Load or Process Dataset Metadata
        if load_cached_data and os.path.exists(self.data_cache_path):
            data = np.load(self.data_cache_path, allow_pickle=True).tolist()
        else:
            df = pd.read_csv(self.meta_path, keep_default_na=False)
            data = []

            for _, row in df.iterrows():
                entry = {
                    "image_id": row["image_id"],
                    "file_path": row["file_path"],
                    "labels": [],
                }

                # Parse labels if available and not test set
                if self.split != "test" and row["labels"]:
                    parts = row["labels"].split()
                    for i in range(0, len(parts), 5):
                        char = parts[i]
                        if char in class_to_idx:
                            try:
                                x = int(parts[i + 1])
                                y = int(parts[i + 2])
                                w = int(parts[i + 3])
                                h = int(parts[i + 4])
                                entry["labels"].append(
                                    {
                                        "char": char,
                                        "class_id": class_to_idx[char],
                                        "bbox": [x, y, w, h],
                                    }
                                )
                            except ValueError:
                                continue

                data.append(entry)

            # Save to cache
            np.save(self.data_cache_path, data)

        return data, class_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        # Load Image
        img_path = os.path.join(self.input_dir, item["file_path"])
        img = cv2.imread(img_path)

        # Handle potential read errors
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        height, width = img.shape[:2]

        # Prepare Affine Transformation (Resize with aspect ratio preservation)
        c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        s = max(height, width) * 1.0
        rot = 0

        # Geometric Augmentation (Cite solution_lesson_node_00006)
        if self.split == "train":
            s = s * np.random.uniform(0.7, 1.3)
            if np.random.random() < 0.5:
                rot = np.random.uniform(-30, 30)
            # Random shift
            c[0] += s * np.random.uniform(-0.1, 0.1)
            c[1] += s * np.random.uniform(-0.1, 0.1)

        # Input transform (to 1024x1024)
        trans_input = utils.get_affine_transform(
            c, s, rot, [self.img_size, self.img_size]
        )
        inp = cv2.warpAffine(
            img, trans_input, (self.img_size, self.img_size), flags=cv2.INTER_LINEAR
        )

        # Normalize (ImageNet stats)
        inp = inp.astype(np.float32) / 255.0
        inp = (inp - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        inp = inp.transpose(2, 0, 1)  # HWC -> CHW

        # Base return dictionary
        ret = {
            "image": torch.from_numpy(inp).float(),
            "image_id": item["image_id"],
            "orig_size": np.array([width, height], dtype=np.int32),
            "center": c,
            "scale": s,
        }

        # If test set, return only image and meta info
        if self.split == "test":
            return ret

        # Generate Targets for Train/Val
        output_h = self.img_size // 4
        output_w = self.img_size // 4

        hm = np.zeros((1, output_h, output_w), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        cls_ids = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        # Output transform (to 256x256)
        trans_output = utils.get_affine_transform(c, s, rot, [output_w, output_h])

        num_objs = 0
        for label in item["labels"]:
            if num_objs >= self.max_objs:
                break

            bbox = label["bbox"]
            cls_id = label["class_id"]

            # Calculate center in original coordinates
            x, y, w, h = bbox
            ct = np.array([x + w / 2, y + h / 2], dtype=np.float32)

            # Transform center to output map coordinates
            ct_out = utils.affine_transform(ct, trans_output)

            ct_x_int = int(ct_out[0])
            ct_y_int = int(ct_out[1])

            # Check bounds
            if not (0 <= ct_x_int < output_w and 0 <= ct_y_int < output_h):
                continue

            # Calculate radius based on transformed dimensions
            resize_ratio = output_w / s
            h_out = h * resize_ratio
            w_out = w * resize_ratio
            radius = utils.gaussian_radius((math.ceil(h_out), math.ceil(w_out)))
            radius = max(0, int(radius))

            # Draw Gaussian on heatmap
            utils.draw_umich_gaussian(hm[0], (ct_x_int, ct_y_int), radius)

            # Fill sparse arrays
            ind[num_objs] = ct_y_int * output_w + ct_x_int
            reg[num_objs] = [ct_out[0] - ct_x_int, ct_out[1] - ct_y_int]
            cls_ids[num_objs] = cls_id
            reg_mask[num_objs] = 1

            num_objs += 1

        ret.update(
            {
                "hm": torch.from_numpy(hm),
                "ind": torch.from_numpy(ind),
                "reg": torch.from_numpy(reg),
                "cls_ids": torch.from_numpy(cls_ids),
                "reg_mask": torch.from_numpy(reg_mask),
            }
        )

        return ret
