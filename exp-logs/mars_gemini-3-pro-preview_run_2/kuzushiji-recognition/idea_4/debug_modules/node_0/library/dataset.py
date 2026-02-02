import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


def get_label_map(load_cached_data=True):
    """
    Retrieves the Unicode to Integer Class ID mapping.
    Uses caching to ensure consistency and speed.
    IDs start at 1 (0 is reserved for background).
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "label_map.parquet")

    # Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_map = pd.read_parquet(cache_path)
            return dict(zip(df_map["Unicode"], df_map["id"]))
        except Exception:
            # If load fails, fall back to creation
            pass

    # Create from scratch using the provided unicode translation file
    unicode_df = pd.read_csv(Config.UNICODE_MAP_PATH)

    # Sort unique characters to ensure deterministic mapping
    unique_chars = sorted(unicode_df["Unicode"].unique())

    # Create map: Char -> ID (1-based index)
    label_map = {char: i + 1 for i, char in enumerate(unique_chars)}

    # Save to cache
    df_map = pd.DataFrame(list(label_map.items()), columns=["Unicode", "id"])
    df_map.to_parquet(cache_path, index=False)

    return label_map


class KuzushijiDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None, load_cached_data=True):
        """
        Args:
            dataframe (pd.DataFrame): Dataframe containing image_id, labels, file_path.
            image_dir (str): Root directory for images (Config.INPUT_DIR).
            transforms (albumentations.Compose): Transforms to apply.
            load_cached_data (bool): Whether to use cached label map.
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.label_map = get_label_map(load_cached_data=load_cached_data)

        # Pre-compute full image paths
        # Metadata file_path is relative to input dir (e.g., "train_images/id.jpg")
        self.image_paths = [
            os.path.join(self.image_dir, p) for p in self.df["file_path"]
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id_str = row["image_id"]
        image_path = self.image_paths[idx]

        # 1. Load Image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        # 2. Resize Logic
        # Scale short edge to MIN_SIZE, limit long edge to MAX_SIZE
        min_size = Config.MIN_SIZE
        max_size = Config.MAX_SIZE

        scale = min_size / min(h, w)
        if max(h, w) * scale > max_size:
            scale = max_size / max(h, w)

        new_w = int(w * scale)
        new_h = int(h * scale)

        image = cv2.resize(image, (new_w, new_h))

        # 3. Parse Labels
        boxes = []
        labels = []

        label_str = row.get("labels", "")

        # Check for valid label string.
        # Ignore the sample submission placeholder for test set images.
        is_placeholder = label_str == "U+003F 1 1 U+FF2F 2 2"

        if pd.notna(label_str) and label_str != "" and not is_placeholder:
            parts = label_str.split()
            # Format: Unicode X Y W H ...
            if len(parts) % 5 == 0:
                for i in range(0, len(parts), 5):
                    char = parts[i]
                    try:
                        x = int(parts[i + 1])
                        y = int(parts[i + 2])
                        bw = int(parts[i + 3])
                        bh = int(parts[i + 4])
                    except ValueError:
                        continue

                    if char in self.label_map:
                        label_id = self.label_map[char]

                        # Convert to XYXY format
                        x1 = x
                        y1 = y
                        x2 = x + bw
                        y2 = y + bh

                        # Apply scaling
                        x1 *= scale
                        y1 *= scale
                        x2 *= scale
                        y2 *= scale

                        # Clip to new image boundaries
                        x1 = max(0, min(x1, new_w))
                        y1 = max(0, min(y1, new_h))
                        x2 = max(0, min(x2, new_w))
                        y2 = max(0, min(y2, new_h))

                        # Filter invalid boxes (area <= 0)
                        if (x2 > x1) and (y2 > y1):
                            boxes.append([x1, y1, x2, y2])
                            labels.append(label_id)

        # Convert to numpy
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Handle empty boxes (Test set or empty pages)
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        # 4. Apply Transforms (Albumentations)
        if self.transforms:
            # Albumentations expects bboxes and labels
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed["image"]
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)
        else:
            # Fallback manual conversion
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # Ensure boxes is [N, 4] even if empty
        if boxes.numel() == 0:
            boxes = boxes.reshape(0, 4)

        # 5. Construct Target Dict
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        # Area (needed for COCO evaluation metrics)
        if len(boxes) > 0:
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)

        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)

        # Metadata for coordinate restoration during inference
        target["orig_size"] = torch.tensor([h, w])
        target["new_size"] = torch.tensor([new_h, new_w])
        target["scale_factor"] = torch.tensor([scale])
        target["image_id_str"] = image_id_str

        return image, target
