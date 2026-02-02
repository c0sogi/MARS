import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import timm
from library.config import Config
from library.utils import BSONIterator


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


class DualBackbone(nn.Module):
    """
    Dual-stream backbone combining ResNet50 and EfficientNet-B0.
    Both backbones are frozen (no gradients) to serve as a fixed feature extractor.
    """

    def __init__(self):
        super(DualBackbone, self).__init__()

        # 1. ResNet50 Backbone
        # Load pretrained weights
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        resnet = models.resnet50(weights=weights)
        # Remove the final FC layer. Keep the feature extractor up to the pooling layer.
        # ResNet50 structure: ... -> avgpool -> fc
        # We want the output of avgpool which is (B, 2048, 1, 1)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])

        # 2. EfficientNet-B0 Backbone
        # Load pretrained model from timm
        # num_classes=0 returns the pooled features (B, 1280)
        self.effnet = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )

        # Freeze parameters to save memory and compute
        for param in self.resnet_features.parameters():
            param.requires_grad = False
        for param in self.effnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        # ResNet Path
        # Input: (B, 3, 224, 224)
        r = self.resnet_features(x)  # Output: (B, 2048, 1, 1)
        r = torch.flatten(r, 1)  # Output: (B, 2048)

        # EfficientNet Path
        e = self.effnet(x)  # Output: (B, 1280)

        # Fusion via Concatenation
        return torch.cat([r, e], dim=1)  # Output: (B, 3328)


class BSONDataset(Dataset):
    """
    Dataset wrapper for BSONIterator to apply transforms and format for DataLoader.
    """

    def __init__(self, bson_path, metadata_path, transform=None):
        self.meta = pd.read_csv(metadata_path)
        self.iterator = BSONIterator(bson_path, self.meta)
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        # Retrieve raw data: product_id, list of images, category_id
        pid, imgs, lbl = self.iterator[idx]

        processed_imgs = []
        if self.transform:
            for img in imgs:
                processed_imgs.append(self.transform(img))
        else:
            to_tensor = transforms.ToTensor()
            for img in imgs:
                processed_imgs.append(to_tensor(img))

        # Ensure at least one image exists (pad with black image if empty - edge case)
        if len(processed_imgs) == 0:
            processed_imgs.append(torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE))

        # Stack images: (K, C, H, W) where K is number of images for this product
        img_stack = torch.stack(processed_imgs)

        return pid, img_stack, lbl


def collate_fn(batch):
    """
    Custom collate function to handle variable number of images per product.
    Flattens the image batch and keeps track of counts for reconstruction.
    """
    pids = []
    all_imgs = []
    labels = []
    counts = []

    for pid, img_stack, lbl in batch:
        pids.append(pid)
        all_imgs.append(img_stack)
        # Handle missing label (test set)
        labels.append(lbl if lbl is not None else -1)
        counts.append(img_stack.shape[0])

    # Concatenate all images into a single batch dimension
    # Shape: (Total_Images_In_Batch, C, H, W)
    flat_imgs = torch.cat(all_imgs, dim=0)

    return (
        torch.tensor(pids, dtype=torch.int64),
        flat_imgs,
        torch.tensor(labels, dtype=torch.int64),
        counts,
    )


def get_transforms():
    """
    Standard ImageNet preprocessing.
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def process_split(
    split_name, bson_path, meta_path, output_feat_path, output_meta_path, model, device
):
    """
    Extracts features for a specific data split and saves to disk using memory mapping.
    """
    print(f"Starting feature extraction for {split_name}...")

    # Initialize Dataset and Loader
    dataset = BSONDataset(bson_path, meta_path, transform=get_transforms())
    loader = DataLoader(
        dataset,
        batch_size=Config.EXTRACT_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    num_samples = len(dataset)
    feat_dim = Config.TOTAL_FEAT_DIM

    # Prepare Output Directory
    os.makedirs(os.path.dirname(output_feat_path), exist_ok=True)

    # Initialize Memory-Mapped Array for Features
    # This allows writing large arrays to disk without holding them in RAM
    features_mmap = np.lib.format.open_memmap(
        output_feat_path, mode="w+", dtype="float32", shape=(num_samples, feat_dim)
    )

    # List to collect metadata (labels or IDs)
    meta_list = []

    model.eval()
    ptr = 0

    with torch.no_grad():
        for pids, imgs, labels, counts in loader:
            imgs = imgs.to(device)

            # Forward Pass
            # imgs shape: (Total_Images_In_Batch, 3, 224, 224)
            features = model(imgs)  # (Total_Images_In_Batch, 3328)

            # Split features back to individual products
            # torch.split returns a tuple of tensors based on the counts list
            split_features = torch.split(features, counts)

            # Mean Pooling per product
            # Stack results into (Batch_Size, 3328)
            pooled_features = torch.stack([f.mean(dim=0) for f in split_features])

            # Move to CPU and write to memmap
            batch_np = pooled_features.cpu().numpy()
            batch_len = batch_np.shape[0]

            features_mmap[ptr : ptr + batch_len] = batch_np
            ptr += batch_len

            # Collect Metadata
            if split_name == "test":
                meta_list.append(pids.numpy())
            else:
                meta_list.append(labels.numpy())

    # Flush changes to disk
    features_mmap.flush()

    # Save Metadata Array
    full_meta = np.concatenate(meta_list)
    np.save(output_meta_path, full_meta)

    print(f"Completed {split_name}: Saved {num_samples} records to {output_feat_path}")


def extract_features(load_cached_data=True):
    """
    Main function to orchestrate feature extraction.
    Checks for cached data and runs the pipeline if necessary.
    """
    set_seed(Config.SEED)

    # Define all expected output files
    expected_files = [
        Config.TRAIN_FEATURES,
        Config.TRAIN_LABELS,
        Config.VAL_FEATURES,
        Config.VAL_LABELS,
        Config.TEST_FEATURES,
        Config.TEST_IDS,
    ]

    # Check Cache
    if load_cached_data:
        missing = [f for f in expected_files if not os.path.exists(f)]
        if not missing:
            print("All cached feature files found. Skipping extraction.")
            return
        else:
            print(f"Cache incomplete. Missing: {missing}. Starting extraction...")

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Feature Extraction Device: {device}")

    # Load Model
    print("Initializing Dual-Backbone Model (ResNet50 + EfficientNet-B0)...")
    model = DualBackbone()
    model.to(device)

    # 1. Process Train
    process_split(
        "train",
        Config.TRAIN_BSON,
        Config.TRAIN_META,
        Config.TRAIN_FEATURES,
        Config.TRAIN_LABELS,
        model,
        device,
    )

    # 2. Process Validation
    process_split(
        "val",
        Config.TRAIN_BSON,
        Config.VAL_META,
        Config.VAL_FEATURES,
        Config.VAL_LABELS,
        model,
        device,
    )

    # 3. Process Test
    process_split(
        "test",
        Config.TEST_BSON,
        Config.TEST_META,
        Config.TEST_FEATURES,
        Config.TEST_IDS,
        model,
        device,
    )

    print("Feature extraction pipeline finished successfully.")
