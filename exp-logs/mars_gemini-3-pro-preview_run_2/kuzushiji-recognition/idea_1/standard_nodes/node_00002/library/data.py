import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from library.config import Config, seed_everything, get_label_map
from library.utils import load_image, parse_labels, generate_heatmap_target

# Ensure reproducibility
seed_everything(Config.SEED)


class SegmentationDataset(Dataset):
    """
    Dataset for Heatmap Regression (CenterNet-style).
    Cite solution_lesson_node_00001: Explicitly regressing center points.
    """

    def __init__(self, df, mode="train", transform=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.target_size = Config.SEG_IMG_SIZE  # (H, W)
        self.sigma = Config.SEG_SIGMA

        # Standard ImageNet normalization
        self.normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        file_path = row["file_path"]

        # Load Image
        img = load_image(file_path)
        orig_h, orig_w = img.shape[:2]

        # Resize Image
        img_resized = cv2.resize(img, (self.target_size[1], self.target_size[0]))

        # Apply Normalization / Transform
        if self.transform:
            img_tensor = self.transform(img_resized)
        else:
            img_tensor = self.normalize(img_resized)

        if self.mode == "test":
            # For inference, we need original dimensions to rescale predictions
            return img_tensor, image_id, torch.tensor([orig_h, orig_w])

        # Generate Heatmap for Train/Val
        labels_str = row.get("labels", "")
        boxes = parse_labels(labels_str)

        # Generate heatmap at target resolution
        heatmap = generate_heatmap_target(
            self.target_size, boxes, original_shape=(orig_h, orig_w), sigma=self.sigma
        )

        # Heatmap to Tensor (Add channel dim: 1xHxW)
        heatmap_tensor = torch.from_numpy(heatmap).float()

        return img_tensor, heatmap_tensor


class ClassificationDataset(Dataset):
    """
    Dataset for Character Classification.
    Crops individual characters from pages.
    """

    def __init__(self, df, mode="train", transform=None, load_cached_data=True):
        self.mode = mode
        self.transform = transform
        self.crop_size = Config.CLS_CROP_SIZE

        # Load label mapping
        self.label2id, self.id2label = get_label_map(load_cached_data=load_cached_data)

        # Prepare or load flattened metadata (one row per character)
        self.samples = self._prepare_data(df, load_cached_data)

        # Standard ImageNet normalization
        self.normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Augmentation for training
        if mode == "train":
            self.augment = T.Compose(
                [
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                ]
            )
        else:
            self.augment = None

    def _prepare_data(self, df, load_cached_data):
        """
        Flattens the image-level dataframe into a character-level dataframe.
        Caches the result to disk.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(Config.CACHE_DIR, f"cls_meta_{self.mode}.parquet")

        if load_cached_data and os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Error loading cache {cache_path}: {e}. Recomputing...")

        # Recompute
        samples = []
        for _, row in df.iterrows():
            image_id = row["image_id"]
            file_path = row["file_path"]
            labels_str = row.get("labels", "")

            boxes = parse_labels(labels_str)

            for box in boxes:
                char = box["char"]
                if char not in self.label2id:
                    continue  # Skip unknown chars if any

                samples.append(
                    {
                        "image_id": image_id,
                        "file_path": file_path,
                        "char": char,
                        "label_id": self.label2id[char],
                        "x": box["x"],
                        "y": box["y"],
                        "w": box["w"],
                        "h": box["h"],
                    }
                )

        samples_df = pd.DataFrame(samples)

        # Save to cache
        if not samples_df.empty:
            samples_df.to_parquet(cache_path, index=False)

        return samples_df

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]

        # Load full image
        # Note: OS file cache will handle repeated reads efficiently given large RAM
        img = load_image(row["file_path"])

        # Crop
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]

        # Handle boundary conditions (padding if necessary, or simple clipping)
        img_h, img_w = img.shape[:2]

        # Clip coordinates
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 > x1 and y2 > y1:
            crop = img[y1:y2, x1:x2]
        else:
            # Fallback for invalid boxes: return black image
            crop = np.zeros((self.crop_size[0], self.crop_size[1], 3), dtype=np.uint8)

        # Resize
        crop_resized = cv2.resize(crop, (self.crop_size[1], self.crop_size[0]))

        # Transform
        # Convert to PIL for torchvision transforms if needed, or apply tensor transforms
        # Here we apply augmentations on the numpy array or tensor

        # Convert to tensor first (0-1 float)
        img_tensor = T.functional.to_tensor(crop_resized)

        if self.augment:
            img_tensor = self.augment(img_tensor)

        # Normalize
        img_tensor = T.functional.normalize(
            img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        label_id = row["label_id"]

        return img_tensor, torch.tensor(label_id, dtype=torch.long)


def get_dataloaders(stage, split="train", batch_size=None, debug=False):
    """
    Factory function to get dataloaders.

    Args:
        stage (str): 'segmentation' or 'classification'
        split (str): 'train', 'val', or 'test'
        batch_size (int): Batch size. If None, uses Config default.
        debug (bool): If True, uses a small subset of data.
    """
    # Select Metadata File
    if split == "train":
        csv_path = Config.TRAIN_CSV
    elif split == "val":
        csv_path = Config.VAL_CSV
    elif split == "test":
        csv_path = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Debug subset
    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    # Dataset Creation
    if stage == "segmentation":
        if batch_size is None:
            batch_size = Config.SEG_BATCH_SIZE

        dataset = SegmentationDataset(df, mode=split)

        shuffle = split == "train"

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    elif stage == "classification":
        if batch_size is None:
            batch_size = Config.CLS_BATCH_SIZE

        # Classification dataset needs labels, so it doesn't support 'test' mode directly
        # (Test inference is done via crops from segmentation output)
        if split == "test":
            raise ValueError(
                "ClassificationDataset does not support 'test' split directly. Use inference pipeline."
            )

        dataset = ClassificationDataset(df, mode=split)

        # Handle Class Imbalance for Training
        sampler = None
        shuffle = split == "train"

        if split == "train":
            # Calculate weights for WeightedRandomSampler
            labels = dataset.samples["label_id"].values
            class_counts = np.bincount(labels)

            # Avoid division by zero
            class_counts = np.maximum(class_counts, 1)
            class_weights = 1.0 / class_counts

            # Assign weight to each sample
            sample_weights = class_weights[labels]

            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False  # Sampler and shuffle are mutually exclusive

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    else:
        raise ValueError(f"Unknown stage: {stage}")
