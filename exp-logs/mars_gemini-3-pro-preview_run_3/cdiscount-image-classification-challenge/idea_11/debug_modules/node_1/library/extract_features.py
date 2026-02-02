import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_BSON_PATH,
    TEST_BSON_PATH,
    TRAIN_FEATURES_PATH,
    TRAIN_LABELS_PATH,
    VAL_FEATURES_PATH,
    VAL_LABELS_PATH,
    TEST_FEATURES_PATH,
    TEST_IDS_PATH,
    CACHE_DIR,
    DEVICE,
    IMG_SIZE,
    EXTRACTION_BATCH_SIZE,
    SEED,
    NUM_WORKERS,
)
from library.data_utils import BSONReader


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DualBackbone(nn.Module):
    """
    A dual-backbone feature extractor using ResNet50 and EfficientNet-B0.
    Both backbones are frozen and their outputs are concatenated.
    Output Dimension: 2048 (ResNet) + 1280 (EffNet) = 3328.
    """

    def __init__(self):
        super(DualBackbone, self).__init__()
        # ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Replace FC with Identity to get features
        self.resnet.fc = nn.Identity()

        # EfficientNet-B0
        self.effnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        # Replace classifier with Identity
        self.effnet.classifier = nn.Identity()

        # Freeze parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.effnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: (B, 3, 224, 224)
        r_feat = self.resnet(x)  # (B, 2048)
        e_feat = self.effnet(x)  # (B, 1280)
        return torch.cat([r_feat, e_feat], dim=1)  # (B, 3328)


class ProductDataset(Dataset):
    """
    Dataset that reads product images from BSON files using metadata.
    Handles products with multiple images.
    """

    def __init__(self, meta_path, bson_path):
        self.meta = pd.read_csv(meta_path)
        self.reader = BSONReader(bson_path)
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]

        # Read images
        try:
            img_bytes_list = self.reader.read_images(offset, length)
        except Exception:
            img_bytes_list = []

        tensors = []
        for img_bytes in img_bytes_list:
            try:
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                tensors.append(self.transform(img_pil))
            except Exception:
                continue

        if len(tensors) == 0:
            # Handle missing/corrupt images with a zero tensor
            tensors.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))

        # Stack images for this product: (N_images, 3, H, W)
        product_images = torch.stack(tensors)

        # Get label and ID
        category_id = row["category_id"] if "category_id" in row else -1
        _id = row["_id"]

        return product_images, category_id, _id


def collate_fn(batch):
    """
    Custom collate function to handle variable number of images per product.
    Flattens all images into a single batch for efficient GPU processing.
    """
    images_list = []
    counts = []
    labels = []
    ids = []

    for imgs, label, pid in batch:
        images_list.append(imgs)
        counts.append(imgs.shape[0])
        labels.append(label)
        ids.append(pid)

    # Concatenate all images into one large batch: (Total_Images, 3, H, W)
    all_images = torch.cat(images_list, dim=0)

    return all_images, torch.tensor(counts), torch.tensor(labels), torch.tensor(ids)


def extract_features_from_loader(model, loader, device):
    """
    Runs inference on the loader and aggregates features per product.
    """
    model.eval()
    all_features = []
    all_labels = []
    all_ids = []

    print(f"Starting extraction for {len(loader.dataset)} products...")

    with torch.no_grad():
        for i, (images, counts, labels, ids) in enumerate(loader):
            images = images.to(device)

            # Forward pass on all images
            features = model(images)  # (Total_Images, 3328)

            # Split features back to products based on image counts
            features_split = torch.split(features, counts.tolist())

            # Mean pooling per product
            pooled_features = [f.mean(dim=0) for f in features_split]
            pooled_features = torch.stack(pooled_features)  # (Batch_Size, 3328)

            all_features.append(pooled_features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_ids.append(ids.numpy())

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} batches...")

    return (
        np.concatenate(all_features),
        np.concatenate(all_labels),
        np.concatenate(all_ids),
    )


def extract_and_save(load_cached_data=True):
    """
    Main function to extract features for Train, Val, and Test sets.
    Checks for cached data first.
    """
    set_seed(SEED)
    os.makedirs(CACHE_DIR, exist_ok=True)

    required_files = [
        TRAIN_FEATURES_PATH,
        TRAIN_LABELS_PATH,
        VAL_FEATURES_PATH,
        VAL_LABELS_PATH,
        TEST_FEATURES_PATH,
        TEST_IDS_PATH,
    ]

    # Check if all files exist
    if load_cached_data and all(os.path.exists(f) for f in required_files):
        print("Loading cached features from disk...")
        train_feats = np.load(TRAIN_FEATURES_PATH)
        train_lbls = np.load(TRAIN_LABELS_PATH)
        val_feats = np.load(VAL_FEATURES_PATH)
        val_lbls = np.load(VAL_LABELS_PATH)
        test_feats = np.load(TEST_FEATURES_PATH)
        test_ids = np.load(TEST_IDS_PATH)
        return train_feats, train_lbls, val_feats, val_lbls, test_feats, test_ids

    print("Cached data not found or reload forced. Starting feature extraction...")

    # Initialize Model
    print("Initializing DualBackbone model (ResNet50 + EfficientNet-B0)...")
    model = DualBackbone()
    model.to(DEVICE)
    model.eval()

    # --- Process Train Split ---
    print("Processing Train Split...")
    train_dataset = ProductDataset(TRAIN_META_PATH, TRAIN_BSON_PATH)
    train_loader = DataLoader(
        train_dataset,
        batch_size=EXTRACTION_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    train_feats, train_lbls, _ = extract_features_from_loader(
        model, train_loader, DEVICE
    )
    print(f"Saving Train features to {TRAIN_FEATURES_PATH}...")
    np.save(TRAIN_FEATURES_PATH, train_feats)
    np.save(TRAIN_LABELS_PATH, train_lbls)

    # --- Process Val Split ---
    print("Processing Val Split...")
    # Note: Val split also comes from train.bson
    val_dataset = ProductDataset(VAL_META_PATH, TRAIN_BSON_PATH)
    val_loader = DataLoader(
        val_dataset,
        batch_size=EXTRACTION_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_feats, val_lbls, _ = extract_features_from_loader(model, val_loader, DEVICE)
    print(f"Saving Val features to {VAL_FEATURES_PATH}...")
    np.save(VAL_FEATURES_PATH, val_feats)
    np.save(VAL_LABELS_PATH, val_lbls)

    # --- Process Test Split ---
    print("Processing Test Split...")
    test_dataset = ProductDataset(TEST_META_PATH, TEST_BSON_PATH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=EXTRACTION_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_feats, _, test_ids = extract_features_from_loader(model, test_loader, DEVICE)
    print(f"Saving Test features to {TEST_FEATURES_PATH}...")
    np.save(TEST_FEATURES_PATH, test_feats)
    np.save(TEST_IDS_PATH, test_ids)

    print("Feature extraction complete.")
    return train_feats, train_lbls, val_feats, val_lbls, test_feats, test_ids
