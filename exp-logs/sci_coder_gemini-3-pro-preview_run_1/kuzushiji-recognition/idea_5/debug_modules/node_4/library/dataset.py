import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import gaussian_radius, draw_umich_gaussian


class KuzushijiDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True, debug_size=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load parsed annotations from cache.
            debug_size (int, optional): Limit dataset size for debugging.
        """
        self.split = split
        self.debug_size = debug_size
        self.max_objs = 1200  # Maximum number of objects per image

        # Select metadata file
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA_PATH
        elif split == "test":
            self.metadata_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load class mapping
        self.char_to_idx = self._load_unicode_map()

        # Load data (with caching)
        self.data = self._process_data(load_cached_data)

        # Setup transforms
        self.transforms = self._get_transforms()

    def _load_unicode_map(self):
        """Loads the unicode translation file and creates a mapping to indices."""
        df = pd.read_csv(Config.UNICODE_MAP_PATH)
        # Assuming the file has a 'Unicode' column.
        if "Unicode" in df.columns:
            chars = df["Unicode"].values
        else:
            chars = df.iloc[:, 0].values
        return {c: i for i, c in enumerate(chars)}

    def _process_data(self, load_cached_data):
        """Parses metadata and caches the result."""
        # Ensure working directory exists
        os.makedirs(Config.WORK_DIR, exist_ok=True)

        cache_file = os.path.join(Config.WORK_DIR, f"{self.split}_parsed.npy")

        # Try loading from cache
        if load_cached_data and os.path.exists(cache_file):
            try:
                print(f"Loading {self.split} data from cache: {cache_file}")
                data = np.load(cache_file, allow_pickle=True).tolist()
                if self.debug_size:
                    return data[: self.debug_size]
                return data
            except Exception as e:
                print(f"Error loading cache: {e}. Reprocessing data.")

        # Process from scratch
        print(f"Processing metadata for {self.split}...")
        df = pd.read_csv(self.metadata_path)

        if self.debug_size:
            df = df.iloc[: self.debug_size]

        data = []
        for _, row in df.iterrows():
            entry = {
                "image_id": row["image_id"],
                "file_path": row["file_path"],
                "bboxes": [],
                "labels": [],
            }

            # Parse labels for train/val sets
            if (
                self.split != "test"
                and "labels" in row
                and isinstance(row["labels"], str)
            ):
                label_str = row["labels"].strip()
                if label_str:
                    parts = label_str.split(" ")
                    # Format: Unicode X Y W H ...
                    for i in range(0, len(parts), 5):
                        try:
                            u_char = parts[i]
                            x = int(parts[i + 1])
                            y = int(parts[i + 2])
                            w = int(parts[i + 3])
                            h = int(parts[i + 4])

                            if u_char in self.char_to_idx:
                                entry["bboxes"].append([x, y, w, h])
                                entry["labels"].append(self.char_to_idx[u_char])
                        except (ValueError, IndexError):
                            continue

            data.append(entry)

        # Save to cache
        try:
            np.save(cache_file, np.array(data, dtype=object))
            print(f"Saved processed data to {cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

        return data

    def _get_transforms(self):
        """Returns Albumentations transforms."""
        if self.split == "train":
            return A.Compose(
                [
                    # Geometric Augmentations Only
                    A.ShiftScaleRotate(
                        shift_limit=Config.SHIFT_LIMIT,
                        scale_limit=(
                            Config.SCALE_RANGE[0] - 1,
                            Config.SCALE_RANGE[1] - 1,
                        ),
                        rotate_limit=Config.ROTATION_DEG,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        p=0.5,
                    ),
                    A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
                    A.Normalize(mean=Config.MEAN, std=Config.STD),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
            )
        else:
            # Val/Test: Resize and Normalize only
            return A.Compose(
                [
                    A.Resize(height=Config.IMG_SIZE[0], width=Config.IMG_SIZE[1]),
                    A.Normalize(mean=Config.MEAN, std=Config.STD),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, entry["file_path"])
        image = cv2.imread(img_path)
        if image is None:
            # Fallback (should not happen given EDA)
            image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        # Prepare for augmentation
        bboxes = entry["bboxes"]
        labels = entry["labels"]

        # Apply Transforms
        if self.split == "test":
            # Test set requires dummy lists for bbox_params
            augmented = self.transforms(image=image, bboxes=[], category_ids=[])
            return {
                "image": augmented["image"],
                "image_id": entry["image_id"],
                "orig_size": np.array([orig_h, orig_w], dtype=np.int32),
            }
        else:
            augmented = self.transforms(image=image, bboxes=bboxes, category_ids=labels)
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            labels = augmented["category_ids"]

            # Generate CenterNet Targets
            output_h = Config.IMG_SIZE[0] // Config.OUTPUT_STRIDE
            output_w = Config.IMG_SIZE[1] // Config.OUTPUT_STRIDE

            # 1. Heatmap (1-channel objectness)
            heatmap = np.zeros((1, output_h, output_w), dtype=np.float32)

            # 2. Dense Regression Maps
            wh = np.zeros((2, output_h, output_w), dtype=np.float32)
            reg = np.zeros((2, output_h, output_w), dtype=np.float32)

            # 3. Sparse Classification Targets
            ind = np.zeros((self.max_objs,), dtype=np.int64)
            cls_ids = np.zeros((self.max_objs,), dtype=np.int64)
            mask = np.zeros((self.max_objs,), dtype=np.float32)

            num_objs = min(len(bboxes), self.max_objs)

            for k in range(num_objs):
                x, y, w, h = bboxes[k]
                cls_id = labels[k]

                # Center in output coordinates
                ct_x = (x + w / 2) / Config.OUTPUT_STRIDE
                ct_y = (y + h / 2) / Config.OUTPUT_STRIDE

                ct_x_int = int(ct_x)
                ct_y_int = int(ct_y)

                if (
                    ct_x_int < 0
                    or ct_x_int >= output_w
                    or ct_y_int < 0
                    or ct_y_int >= output_h
                ):
                    continue

                # Gaussian Radius
                radius = gaussian_radius(
                    (h / Config.OUTPUT_STRIDE, w / Config.OUTPUT_STRIDE)
                )
                radius = max(0, int(radius))

                # Draw Gaussian
                draw_umich_gaussian(heatmap[0], (ct_x_int, ct_y_int), radius)

                # Set Regression Targets
                wh[0, ct_y_int, ct_x_int] = w / Config.OUTPUT_STRIDE
                wh[1, ct_y_int, ct_x_int] = h / Config.OUTPUT_STRIDE

                reg[0, ct_y_int, ct_x_int] = ct_x - ct_x_int
                reg[1, ct_y_int, ct_x_int] = ct_y - ct_y_int

                # Set Sparse Targets
                ind[k] = ct_y_int * output_w + ct_x_int
                cls_ids[k] = cls_id
                mask[k] = 1

            return {
                "image": image,
                "heatmap": torch.from_numpy(heatmap),
                "wh": torch.from_numpy(wh),
                "reg": torch.from_numpy(reg),
                "ind": torch.from_numpy(ind),
                "cls_ids": torch.from_numpy(cls_ids),
                "mask": torch.from_numpy(mask),
                "image_id": entry["image_id"],
            }
