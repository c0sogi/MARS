import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, Dict, List, Tuple

from library.config import Config
from library.utils import seed_everything

# ------------------------------------------------------------------------------
# 1. MegaDetector Processing & Caching
# ------------------------------------------------------------------------------


def process_megadetector_results(
    json_path: str, cache_path: str, load_cached_data: bool = True
) -> Dict[str, List[float]]:
    """
    Parses the MegaDetector JSON to extract the highest confidence bounding box
    for category '1' (animal) for each image.

    Returns a dictionary: {image_id: [x, y, w, h]} (relative coordinates).
    If no animal detection, returns [0.0, 0.0, 1.0, 1.0].
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached bounding boxes from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            # Convert to dictionary for O(1) lookup
            # Expected columns: id, x, y, w, h
            # We use orient='index' logic via T.to_dict
            bbox_dict = df.set_index("id")[["x", "y", "w", "h"]].T.to_dict("list")
            return bbox_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    # 2. Process from scratch
    print(f"Processing MegaDetector results from {json_path}...")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"MegaDetector file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    records = []

    # The JSON structure: data['images'] is a list of dicts
    for img_entry in data["images"]:
        img_id = img_entry["id"]
        detections = img_entry.get("detections", [])

        # Filter for category "1" (animal)
        animal_detections = [d for d in detections if d["category"] == "1"]

        if animal_detections:
            # Sort by confidence descending
            animal_detections.sort(key=lambda x: x["conf"], reverse=True)
            best_det = animal_detections[0]
            bbox = best_det["bbox"]  # [x, y, w, h]
        else:
            # No animal detected, use full image
            bbox = [0.0, 0.0, 1.0, 1.0]

        records.append(
            {"id": img_id, "x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}
        )

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save to cache
    df.to_parquet(cache_path, index=False)
    print(f"Saved bounding boxes to {cache_path}")

    # Convert to dict
    bbox_dict = df.set_index("id")[["x", "y", "w", "h"]].T.to_dict("list")
    return bbox_dict


# ------------------------------------------------------------------------------
# 2. Dataset Class
# ------------------------------------------------------------------------------


class WildCamDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        bbox_map: Dict[str, List[float]],
        transform=None,
        is_test: bool = False,
        root_dir: str = Config.INPUT_DIR,
    ):
        """
        Args:
            metadata_path: Path to the metadata CSV (train/val/test).
            bbox_map: Dictionary mapping image_id to [x, y, w, h].
            transform: Torchvision transforms.
            is_test: If True, returns dummy label.
            root_dir: Root directory for images.
        """
        self.df = pd.read_csv(metadata_path)
        self.bbox_map = bbox_map
        self.transform = transform
        self.is_test = is_test
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        file_path = os.path.join(self.root_dir, row["file_path"])

        # Load Image
        # cv2 loads BGR, convert to RGB
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing images (create black image)
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get Bounding Box
        # Default to full image if ID not in map
        bbox_rel = self.bbox_map.get(img_id, [0.0, 0.0, 1.0, 1.0])

        # Crop
        h_img, w_img, _ = image.shape
        x_rel, y_rel, w_rel, h_rel = bbox_rel

        x1 = int(x_rel * w_img)
        y1 = int(y_rel * h_img)
        w_px = int(w_rel * w_img)
        h_px = int(h_rel * h_img)

        # Clip to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x1 + w_px)
        y2 = min(h_img, y1 + h_px)

        # Ensure valid crop
        if x2 > x1 and y2 > y1:
            crop = image[y1:y2, x1:x2]
        else:
            crop = image

        # Convert to PIL for transforms
        pil_img = Image.fromarray(crop)

        if self.transform:
            pil_img = self.transform(pil_img)

        if self.is_test:
            target = 0  # Dummy
        else:
            target = int(row["category_id"])

        return pil_img, target, img_id


# ------------------------------------------------------------------------------
# 3. Mixup/CutMix Collator
# ------------------------------------------------------------------------------


class MixupCutmixCollator:
    def __init__(self, num_classes, mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(self, batch):
        """
        Batch is a list of tuples (image, label, img_id)
        Returns:
            images: mixed images
            targets: soft targets (one-hot mixed)
        """
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

        # Convert labels to one-hot
        targets = torch.zeros(len(labels), self.num_classes, device=images.device)
        targets.scatter_(1, labels.view(-1, 1), 1)

        # Apply mixup/cutmix with probability
        if np.random.rand() > self.prob:
            return images, targets

        # Decide Mixup or CutMix
        use_cutmix = np.random.rand() > 0.5

        if use_cutmix:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            rand_index = torch.randperm(images.size(0))

            target_a = targets
            target_b = targets[rand_index]

            bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # Adjust lambda to match pixel ratio exactly
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2))
            )

            targets = target_a * lam + target_b * (1.0 - lam)

        else:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            rand_index = torch.randperm(images.size(0))

            target_a = targets
            target_b = targets[rand_index]

            images = images * lam + images[rand_index] * (1.0 - lam)
            targets = target_a * lam + target_b * (1.0 - lam)

        return images, targets

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


# ------------------------------------------------------------------------------
# 4. Data Loaders
# ------------------------------------------------------------------------------


def get_dataloaders(load_cached_data: bool = True, sample_size: Optional[int] = None):
    """
    Creates and returns dataloaders for train, val, and test.

    Args:
        load_cached_data: Whether to use cached bbox data.
        sample_size: If provided, subsets the data for debugging.
    """

    # 1. Process/Load Bounding Boxes
    bbox_map = process_megadetector_results(
        Config.MEGADETECTOR_PATH,
        Config.CACHED_BBOXES_PATH,
        load_cached_data=load_cached_data,
    )

    # 2. Define Transforms
    # ImageNet stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Val/Test: Resize -> ToTensor -> Normalize
    eval_transform = transforms.Compose(
        [
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # 3. Create Datasets
    train_dataset = WildCamDataset(
        Config.TRAIN_METADATA_PATH, bbox_map, transform=train_transform, is_test=False
    )
    val_dataset = WildCamDataset(
        Config.VAL_METADATA_PATH, bbox_map, transform=eval_transform, is_test=False
    )
    test_dataset = WildCamDataset(
        Config.TEST_METADATA_PATH, bbox_map, transform=eval_transform, is_test=True
    )

    # Subset if requested
    if sample_size:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(min(len(train_dataset), sample_size))
        )
        val_dataset = torch.utils.data.Subset(
            val_dataset, range(min(len(val_dataset), sample_size))
        )
        test_dataset = torch.utils.data.Subset(
            test_dataset, range(min(len(test_dataset), sample_size))
        )

    # 5. Create Loaders
    # Train loader uses default collator (returns images, targets, ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val/Test loaders use default collator (returns images, labels, ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
