import os
import cv2
import math
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.utils import (
    gaussian_radius,
    draw_gaussian,
    get_affine_transform,
    affine_transform,
)


class KuzushijiDetectionDataset(Dataset):
    """
    Stage 1 Dataset: Global Context Detection.
    - Input: Images resized to 1024x1024 (preserving aspect ratio via padding).
    - Targets: Heatmap (Textness), Size (W, H), Offset (X, Y).
    - Augmentation: Random Scaling (+/- 30%), Random Rotation (+/- 5 deg).
    """

    def __init__(self, split="train", debug=False, load_cached_data=True):
        self.split = split
        self.debug = debug

        # Select Metadata File
        if split == "train":
            self.meta_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            self.meta_path = Config.VAL_METADATA_PATH
        else:
            self.meta_path = Config.TEST_METADATA_PATH

        # Load Metadata
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        self.df = pd.read_csv(self.meta_path)

        if self.debug:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Prepend input dir to file paths
        self.df["full_path"] = self.df["file_path"].apply(
            lambda x: os.path.join(Config.INPUT_DIR, x)
        )

        # Parameters
        self.input_size = Config.DETECTOR_INPUT_SIZE
        self.output_size = self.input_size // 4  # CenterNet default stride is 4
        self.max_rotation = 5
        self.scale_range = 0.3

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["full_path"]

        # Load Image
        img = cv2.imread(img_path)
        if img is None:
            # Fallback for corrupt images
            img = np.zeros((512, 512, 3), dtype=np.uint8)

        height, width = img.shape[:2]
        c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        s = max(height, width) * 1.0

        # Augmentation (Train only)
        rot = 0
        if self.split == "train":
            scale_factor = random.uniform(1 - self.scale_range, 1 + self.scale_range)
            s = s * scale_factor
            rot = random.uniform(-self.max_rotation, self.max_rotation)

        # Affine Transform for Input (1024x1024)
        trans_input = get_affine_transform(
            c, s, rot, [self.input_size, self.input_size]
        )
        inp = cv2.warpAffine(
            img, trans_input, (self.input_size, self.input_size), flags=cv2.INTER_LINEAR
        )

        # Normalize Input (ImageNet stats)
        inp = inp.astype(np.float32) / 255.0
        inp = (inp - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        inp = inp.transpose(2, 0, 1)  # HWC -> CHW

        # Initialize Targets
        hm = np.zeros((1, self.output_size, self.output_size), dtype=np.float32)
        wh = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        reg = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        reg_mask = np.zeros((1, self.output_size, self.output_size), dtype=np.float32)

        # Affine Transform for Output (256x256)
        trans_output = get_affine_transform(
            c, s, rot, [self.output_size, self.output_size]
        )

        # Process Labels
        labels_str = row["labels"]
        if isinstance(labels_str, str) and len(labels_str) > 0:
            parts = labels_str.split()
            # Format: Code X Y W H ...
            num_objs = len(parts) // 5

            for i in range(num_objs):
                try:
                    # Parse GT
                    x = float(parts[i * 5 + 1])
                    y = float(parts[i * 5 + 2])
                    w = float(parts[i * 5 + 3])
                    h = float(parts[i * 5 + 4])

                    # Calculate Center
                    ct_x = x + w / 2
                    ct_y = y + h / 2
                    ct = np.array([ct_x, ct_y], dtype=np.float32)

                    # Transform Center to Output Coordinates
                    ct_out = affine_transform(ct, trans_output)

                    # Transform Width/Height
                    # Since rotation is small, we approximate new w/h by scaling
                    resize_ratio = self.output_size / s
                    h_out = h * resize_ratio
                    w_out = w * resize_ratio

                    # Integer Coordinates for Heatmap
                    ct_x_int = int(ct_out[0])
                    ct_y_int = int(ct_out[1])

                    # Check bounds
                    if (
                        ct_x_int >= 0
                        and ct_x_int < self.output_size
                        and ct_y_int >= 0
                        and ct_y_int < self.output_size
                    ):

                        # Draw Gaussian
                        radius = gaussian_radius((math.ceil(h_out), math.ceil(w_out)))
                        radius = max(0, int(radius))
                        draw_gaussian(hm[0], (ct_x_int, ct_y_int), radius)

                        # Regression Targets
                        wh[0, ct_y_int, ct_x_int] = w_out
                        wh[1, ct_y_int, ct_x_int] = h_out

                        # Offset Targets (Sub-pixel)
                        reg[0, ct_y_int, ct_x_int] = ct_out[0] - ct_x_int
                        reg[1, ct_y_int, ct_x_int] = ct_out[1] - ct_y_int

                        reg_mask[0, ct_y_int, ct_x_int] = 1
                except ValueError:
                    continue

        return torch.from_numpy(inp).float(), {
            "hm": torch.from_numpy(hm).float(),
            "wh": torch.from_numpy(wh).float(),
            "reg": torch.from_numpy(reg).float(),
            "reg_mask": torch.from_numpy(reg_mask).float(),
        }


class KuzushijiClassificationDataset(Dataset):
    """
    Stage 2 Dataset: Fine-grained Classification & Verification.
    - Input: High-resolution crops (128x128).
    - Classes: 3848 Characters + 1 Background.
    - Features: Class-Balanced Sampling, Background Generation, Caching.
    """

    def __init__(self, split="train", debug=False, load_cached_data=True):
        self.split = split
        self.debug = debug
        self.cache_dir = Config.CACHE_DIR
        self.cache_file = os.path.join(
            self.cache_dir, f"classifier_samples_{split}.npy"
        )

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load Class Map (Deterministic)
        self.class_map = self._get_class_map()
        self.bg_class_id = Config.BACKGROUND_CLASS_ID

        # Load or Generate Data
        if load_cached_data and os.path.exists(self.cache_file):
            print(f"Loading cached classifier data from {self.cache_file}")
            self.samples = np.load(self.cache_file, allow_pickle=True).tolist()
        else:
            print(f"Processing classifier data for {split}...")
            self.samples = self._process_data()
            np.save(self.cache_file, self.samples)

        if self.debug:
            self.samples = self.samples[:1000]

        print(f"Classifier Dataset ({split}): {len(self.samples)} samples loaded.")

    def _get_class_map(self):
        """
        Generates a deterministic map from Unicode strings to Integer IDs based on Training Data.
        """
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        all_codes = set()
        for labels in train_df["labels"].dropna():
            parts = labels.split()
            # Code is every 5th element
            for i in range(0, len(parts), 5):
                all_codes.add(parts[i])

        # Sort for determinism
        unique_codes = sorted(list(all_codes))
        return {code: idx for idx, code in enumerate(unique_codes)}

    def _process_data(self):
        """
        Parses metadata to create a list of crops.
        Generates background samples and balances classes for training.
        """
        if self.split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        else:
            meta_path = Config.VAL_METADATA_PATH

        df = pd.read_csv(meta_path)
        df["full_path"] = df["file_path"].apply(
            lambda x: os.path.join(Config.INPUT_DIR, x)
        )

        samples = []

        for idx, row in df.iterrows():
            img_path = row["full_path"]

            # Read image dimensions for background generation
            # We read the image to ensure valid crop coordinates
            img = cv2.imread(img_path)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]

            gt_boxes = []

            # 1. Extract Ground Truth Character Crops
            if isinstance(row["labels"], str) and row["labels"]:
                parts = row["labels"].split()
                for i in range(0, len(parts), 5):
                    try:
                        code = parts[i]
                        x = int(parts[i + 1])
                        y = int(parts[i + 2])
                        w = int(parts[i + 3])
                        h = int(parts[i + 4])

                        if code in self.class_map:
                            label_idx = self.class_map[code]
                            samples.append(
                                {
                                    "path": img_path,
                                    "bbox": (x, y, w, h),
                                    "label": label_idx,
                                }
                            )
                            gt_boxes.append([x, y, x + w, y + h])
                    except ValueError:
                        continue

            # 2. Generate Background Crops (Train/Val only)
            # We generate 5 background samples per image to train the 'No-Text' class
            num_bg = 5
            count = 0
            attempts = 0
            while count < num_bg and attempts < 20:
                attempts += 1
                # Random crop size (approximate character size)
                cw = random.randint(50, 120)
                ch = random.randint(50, 120)

                if w_img - cw <= 0 or h_img - ch <= 0:
                    continue

                cx = random.randint(0, w_img - cw)
                cy = random.randint(0, h_img - ch)

                # Check Overlap with GT
                cand_box = [cx, cy, cx + cw, cy + ch]
                overlap = False
                for gb in gt_boxes:
                    # Intersection
                    ix1 = max(cand_box[0], gb[0])
                    iy1 = max(cand_box[1], gb[1])
                    ix2 = min(cand_box[2], gb[2])
                    iy2 = min(cand_box[3], gb[3])
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    if iw * ih > 0:  # Any overlap
                        overlap = True
                        break

                if not overlap:
                    samples.append(
                        {
                            "path": img_path,
                            "bbox": (cx, cy, cw, ch),
                            "label": self.bg_class_id,
                        }
                    )
                    count += 1

        # 3. Class Balancing (Train Only)
        if self.split == "train":
            samples = self._balance_samples(samples)

        return samples

    def _balance_samples(self, samples):
        """
        Balances the dataset by oversampling rare classes.
        """
        from collections import defaultdict

        groups = defaultdict(list)
        for s in samples:
            groups[s["label"]].append(s)

        # Calculate target count (Mean of all class counts)
        counts = [len(v) for v in groups.values()]
        if not counts:
            return samples

        target_count = int(np.mean(counts))

        balanced = []
        for label, items in groups.items():
            if len(items) >= target_count:
                # Keep all samples for frequent classes (don't throw away data)
                balanced.extend(items)
            else:
                # Oversample rare classes
                needed = target_count - len(items)
                extras = random.choices(items, k=needed)
                balanced.extend(items + extras)

        random.shuffle(balanced)
        return balanced

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = cv2.imread(item["path"])

        if img is None:
            # Fallback
            crop = np.zeros(
                (Config.CLASSIFIER_INPUT_SIZE, Config.CLASSIFIER_INPUT_SIZE, 3),
                dtype=np.float32,
            )
        else:
            x, y, w, h = item["bbox"]
            # Ensure coords are within bounds
            h_img, w_img = img.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))

            crop = img[y : y + h, x : x + w]

            if crop.size == 0:
                crop = np.zeros(
                    (Config.CLASSIFIER_INPUT_SIZE, Config.CLASSIFIER_INPUT_SIZE, 3),
                    dtype=np.uint8,
                )

        # Resize to fixed classifier input size
        crop = cv2.resize(
            crop, (Config.CLASSIFIER_INPUT_SIZE, Config.CLASSIFIER_INPUT_SIZE)
        )

        # Augmentation (Train only)
        if self.split == "train":
            # Simple color jitter simulation could go here
            pass

        # Normalize
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        crop = crop.transpose(2, 0, 1)

        return torch.from_numpy(crop).float(), torch.tensor(
            item["label"], dtype=torch.long
        )
