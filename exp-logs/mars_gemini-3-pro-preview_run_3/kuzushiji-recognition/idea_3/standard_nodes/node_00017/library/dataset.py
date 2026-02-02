import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config
from library.utils import gaussian_radius, draw_umich_gaussian

# ==========================================
# 1. Utilities and Transforms
# ==========================================


def get_detector_transforms(mode="train"):
    """
    Returns Albumentations transforms for the detector.
    Includes random scaling, safe rotation, and cropping.
    """
    if mode == "train":
        return A.Compose(
            [
                # Random scaling
                A.RandomScale(
                    scale_limit=(Config.SCALE_RANGE[0] - 1, Config.SCALE_RANGE[1] - 1),
                    p=0.5,
                ),
                # Safe rotation (small angles for vertical text)
                A.SafeRotate(
                    limit=Config.ROTATION_RANGE,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Ensure image is at least the crop size after scaling/rotation
                A.PadIfNeeded(
                    min_height=Config.DETECTOR_INPUT_SIZE[0],
                    min_width=Config.DETECTOR_INPUT_SIZE[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Random Crop to fixed size
                A.RandomCrop(
                    height=Config.DETECTOR_INPUT_SIZE[0],
                    width=Config.DETECTOR_INPUT_SIZE[1],
                    p=1.0,
                ),
                # Color Jitter
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3
                ),
                # Normalize and Convert
                A.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco", label_fields=["class_labels"], min_visibility=0.3
            ),
        )
    else:
        # Validation/Test: Just normalize.
        # Note: Tiling logic handles cropping during inference, but for val set evaluation
        # we might want center crops or similar if we were doing standard eval.
        # However, the detector dataset is primarily for training.
        return A.Compose(
            [
                A.CenterCrop(
                    height=Config.DETECTOR_INPUT_SIZE[0],
                    width=Config.DETECTOR_INPUT_SIZE[1],
                ),
                A.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD),
                ToTensorV2(),
            ],
            # No bbox params needed for pure inference, but kept for validation consistency
            bbox_params=A.BboxParams(
                format="coco", label_fields=["class_labels"], min_visibility=0.3
            ),
        )


def get_classifier_transforms(mode="train"):
    """
    Returns transforms for the classifier (64x64 crops).
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(
                    p=0.5
                ),  # Some characters might be symmetric, but be careful.
                # Actually, Kuzushiji is rarely symmetric. Let's disable flip.
                # A.Rotate(limit=10, p=0.5), # Small rotation
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
                A.CoarseDropout(max_holes=4, max_height=8, max_width=8, p=0.2),
                A.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD),
                ToTensorV2(),
            ]
        )


def parse_labels(label_str):
    """
    Parses the space-separated label string into a list of [code, x, y, w, h].
    """
    if not isinstance(label_str, str) or not label_str:
        return []

    parts = label_str.split()
    num_chars = len(parts) // 5
    labels = []
    for i in range(num_chars):
        code = parts[i * 5]
        try:
            x = int(parts[i * 5 + 1])
            y = int(parts[i * 5 + 2])
            w = int(parts[i * 5 + 3])
            h = int(parts[i * 5 + 4])
            labels.append([code, x, y, w, h])
        except ValueError:
            continue
    return labels


# ==========================================
# 2. Detector Dataset
# ==========================================


class KuzushijiDetectorDataset(Dataset):
    def __init__(self, metadata_path, mode="train", transform=None):
        self.df = pd.read_csv(metadata_path, keep_default_na=False)
        self.mode = mode
        self.transform = transform or get_detector_transforms(mode)

        # Prepend input dir to file paths
        self.df["full_path"] = self.df["file_path"].apply(
            lambda x: os.path.join(Config.INPUT_DIR, x)
        )

        # Filter out images that don't exist (safety check)
        self.df = self.df[self.df["full_path"].apply(os.path.exists)].reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["full_path"]
        label_str = row["labels"]

        # 1. Load Image
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for corrupt images
            image = np.zeros(
                (Config.DETECTOR_INPUT_SIZE[0], Config.DETECTOR_INPUT_SIZE[1], 3),
                dtype=np.uint8,
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = image.shape

        # 2. Parse Labels
        # Format: [code, x, y, w, h]
        raw_labels = parse_labels(label_str)

        bboxes = []
        class_labels = []

        for item in raw_labels:
            # item: [code, x, y, w, h]
            # Albumentations expects [x, y, w, h] for COCO format
            code, x, y, w, h = item
            bboxes.append([x, y, w, h])
            class_labels.append(1)  # Class 1 for "Character" (Class agnostic detection)

        # 3. Augmentation (including Random Crop)
        # If image is smaller than crop size, pad it first
        pad_h = max(0, Config.DETECTOR_INPUT_SIZE[0] - h_img)
        pad_w = max(0, Config.DETECTOR_INPUT_SIZE[1] - w_img)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            )

        if self.transform:
            # Albumentations requires bboxes to be list of lists
            # Handle empty bboxes case
            if len(bboxes) == 0:
                # Dummy box to satisfy albumentations if needed, or handle separately
                # Albumentations handles empty bboxes if configured correctly
                transformed = self.transform(image=image, bboxes=[], class_labels=[])
            else:
                try:
                    transformed = self.transform(
                        image=image, bboxes=bboxes, class_labels=class_labels
                    )
                except ValueError:
                    # Fallback if augmentation fails (e.g. bbox out of bounds)
                    transformed = self.transform(
                        image=image, bboxes=[], class_labels=[]
                    )

            image_tensor = transformed["image"]
            aug_bboxes = transformed["bboxes"]
        else:
            image_tensor = ToTensorV2()(image=image)["image"]
            aug_bboxes = bboxes

        # 4. Generate Heatmap and Regression Targets
        output_h = Config.DETECTOR_INPUT_SIZE[0] // Config.DETECTOR_OUTPUT_STRIDE
        output_w = Config.DETECTOR_INPUT_SIZE[1] // Config.DETECTOR_OUTPUT_STRIDE

        hm = np.zeros((1, output_h, output_w), dtype=np.float32)
        wh = np.zeros((2, output_h, output_w), dtype=np.float32)  # Width, Height
        reg = np.zeros((2, output_h, output_w), dtype=np.float32)  # Offset x, y
        reg_mask = np.zeros(
            (1, output_h, output_w), dtype=np.float32
        )  # Mask for valid objects

        for bbox in aug_bboxes:
            x, y, w, h = bbox

            # Center of the box
            ct_x = x + w / 2
            ct_y = y + h / 2

            # Scale to feature map size
            ct_x_feat = ct_x / Config.DETECTOR_OUTPUT_STRIDE
            ct_y_feat = ct_y / Config.DETECTOR_OUTPUT_STRIDE
            w_feat = w / Config.DETECTOR_OUTPUT_STRIDE
            h_feat = h / Config.DETECTOR_OUTPUT_STRIDE

            # Integer coordinates
            ct_x_int = int(ct_x_feat)
            ct_y_int = int(ct_y_feat)

            # Check bounds
            if (
                ct_x_int >= 0
                and ct_x_int < output_w
                and ct_y_int >= 0
                and ct_y_int < output_h
            ):
                # Gaussian Radius
                radius = gaussian_radius(
                    (np.ceil(h_feat), np.ceil(w_feat)), min_overlap=Config.GAUSSIAN_IOU
                )
                radius = max(0, int(radius))

                # Draw Gaussian on Heatmap
                draw_umich_gaussian(hm[0], (ct_x_int, ct_y_int), radius)

                # Regression Targets
                wh[0, ct_y_int, ct_x_int] = w_feat
                wh[1, ct_y_int, ct_x_int] = h_feat

                reg[0, ct_y_int, ct_x_int] = ct_x_feat - ct_x_int
                reg[1, ct_y_int, ct_x_int] = ct_y_feat - ct_y_int

                reg_mask[0, ct_y_int, ct_x_int] = 1

        return {
            "image": image_tensor,
            "hm": torch.from_numpy(hm),
            "wh": torch.from_numpy(wh),
            "reg": torch.from_numpy(reg),
            "reg_mask": torch.from_numpy(reg_mask),
        }


# ==========================================
# 3. Classifier Dataset
# ==========================================


class KuzushijiClassifierDataset(Dataset):
    def __init__(self, data_array, label_array, transform=None):
        self.data = data_array
        self.labels = label_array
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Data is stored as uint8 (H, W, C)
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = ToTensorV2()(image=image)["image"]

        return image, label


def prepare_classifier_data(metadata_path, cache_name, load_cached_data=True):
    """
    Prepares crop data for the classifier.
    Caches the result as .npy files.
    """
    cache_data_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_data.npy")
    cache_label_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_labels.npy")
    class_map_path = Config.CLASS_MAP_PATH

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(cache_data_path)
        and os.path.exists(cache_label_path)
    ):
        print(f"Loading cached classifier data from {cache_data_path}...")
        data = np.load(cache_data_path)
        labels = np.load(cache_label_path)
        return data, labels

    print(f"Generating classifier data from {metadata_path}...")
    df = pd.read_csv(metadata_path, keep_default_na=False)

    all_crops = []
    all_codes = []

    # Load or Create Class Map
    if os.path.exists(class_map_path):
        class_map = np.load(class_map_path, allow_pickle=True).item()
    else:
        # If generating training data, we build the map.
        # If generating val data, we must assume map exists (or rebuild from full train).
        # For simplicity, we build map from unicode_translation.csv (all possible chars)
        uni_df = pd.read_csv(Config.UNICODE_MAP_PATH)
        class_map = {row["Unicode"]: i for i, row in uni_df.iterrows()}
        # Ensure we save this map
        np.save(class_map_path, class_map)

    # Iterate images
    for idx, row in df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        label_str = row["labels"]

        if not os.path.exists(img_path) or not label_str:
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        parsed = parse_labels(label_str)

        for code, x, y, w, h in parsed:
            if code not in class_map:
                continue  # Skip unknown classes

            # Crop
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]

            # Resize
            crop_resized = cv2.resize(crop, Config.CLASSIFIER_INPUT_SIZE)

            all_crops.append(crop_resized)
            all_codes.append(class_map[code])

    # Convert to arrays
    data_array = np.array(all_crops, dtype=np.uint8)
    label_array = np.array(all_codes, dtype=np.int64)

    # Save Cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_data_path, data_array)
    np.save(cache_label_path, label_array)

    print(f"Saved classifier data: {data_array.shape}")
    return data_array, label_array


# ==========================================
# 4. Data Loaders
# ==========================================


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for both Detector and Classifier stages.
    """
    # ---------------------------
    # Detector DataLoaders
    # ---------------------------
    print("Initializing Detector DataLoaders...")
    train_det_ds = KuzushijiDetectorDataset(
        Config.TRAIN_METADATA, mode="train", transform=get_detector_transforms("train")
    )
    val_det_ds = KuzushijiDetectorDataset(
        Config.VAL_METADATA, mode="val", transform=get_detector_transforms("val")
    )

    train_det_loader = DataLoader(
        train_det_ds,
        batch_size=Config.DETECTOR_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_det_loader = DataLoader(
        val_det_ds,
        batch_size=Config.DETECTOR_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ---------------------------
    # Classifier DataLoaders
    # ---------------------------
    print("Initializing Classifier DataLoaders...")

    # Prepare Data
    train_X, train_y = prepare_classifier_data(
        Config.TRAIN_METADATA, "classifier_train", load_cached_data
    )
    val_X, val_y = prepare_classifier_data(
        Config.VAL_METADATA, "classifier_val", load_cached_data
    )

    # Class Balancing for Training
    # Calculate weights: 1 / frequency
    class_counts = np.bincount(train_y)
    # Handle classes with 0 counts in this split (though unlikely if map is built from data)
    # If map is from unicode_translation, some indices might not appear in train_y.
    # bincount length is max(train_y) + 1.

    # We need weights for every sample in train_y
    class_weights = np.zeros(len(class_counts))
    class_weights[class_counts > 0] = 1.0 / class_counts[class_counts > 0]

    # Map weights to samples
    sample_weights = class_weights[train_y]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_cls_ds = KuzushijiClassifierDataset(
        train_X, train_y, transform=get_classifier_transforms("train")
    )
    val_cls_ds = KuzushijiClassifierDataset(
        val_X, val_y, transform=get_classifier_transforms("val")
    )

    train_cls_loader = DataLoader(
        train_cls_ds,
        batch_size=Config.CLASSIFIER_BATCH_SIZE,
        sampler=sampler,  # Use sampler instead of shuffle
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_cls_loader = DataLoader(
        val_cls_ds,
        batch_size=Config.CLASSIFIER_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return {
        "detector_train": train_det_loader,
        "detector_val": val_det_loader,
        "classifier_train": train_cls_loader,
        "classifier_val": val_cls_loader,
    }
