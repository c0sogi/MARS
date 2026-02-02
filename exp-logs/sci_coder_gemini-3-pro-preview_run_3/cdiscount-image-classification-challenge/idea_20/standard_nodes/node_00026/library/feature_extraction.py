import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import timm
from torch_scatter import scatter_mean

from library.config import Config
from library.bson_io import BSONImageReader
from library.utils import set_seed


class BSONDataset(Dataset):
    """
    PyTorch Dataset for reading product images from BSON files using metadata.
    """

    def __init__(self, metadata_path, bson_path, transform=None, is_test=False):
        self.metadata = pd.read_csv(metadata_path)
        self.bson_path = bson_path
        self.transform = transform
        self.is_test = is_test

        # We initialize the reader in __getitem__ or via worker_init_fn to be safe
        # but BSONImageReader handles pickling by resetting file handle, so it's safe to hold here
        # provided we don't open it in __init__
        self.reader = BSONImageReader(bson_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]

        # Read images
        # Returns list of RGB numpy arrays
        images_np = self.reader.read_product(offset, length)

        # Handle case with no images (should be rare/impossible in this dataset)
        if len(images_np) == 0:
            # Create a black image as placeholder
            images_np = [np.zeros((224, 224, 3), dtype=np.uint8)]

        # Apply transforms
        images_tensor = []
        if self.transform:
            for img in images_np:
                images_tensor.append(self.transform(img))
        else:
            # Default to tensor conversion
            for img in images_np:
                images_tensor.append(T.functional.to_tensor(img))

        # Stack images: (N_imgs, C, H, W)
        images_tensor = torch.stack(images_tensor)

        if self.is_test:
            return images_tensor, row["_id"]
        else:
            return images_tensor, row["category_id"]


def collate_fn(batch):
    """
    Custom collate function to handle variable number of images per product.

    Args:
        batch: List of tuples (images_tensor, label_or_id)
               images_tensor shape: (N_imgs, C, H, W)

    Returns:
        flat_images: (Total_Images, C, H, W)
        batch_indices: (Total_Images,) - maps each image to its product index in the batch
        targets: (Batch_Size,) - category_ids or _ids
    """
    images_list = []
    targets_list = []
    batch_indices_list = []

    for i, (imgs, target) in enumerate(batch):
        images_list.append(imgs)
        targets_list.append(target)
        # Create indices [i, i, i...] for the number of images in this product
        batch_indices_list.append(torch.full((imgs.shape[0],), i, dtype=torch.long))

    flat_images = torch.cat(images_list, dim=0)
    batch_indices = torch.cat(batch_indices_list, dim=0)
    targets = torch.tensor(targets_list, dtype=torch.int64)

    return flat_images, batch_indices, targets


class DualBackbone(nn.Module):
    """
    Wrapper for ResNet50 and EfficientNet-B0 feature extraction.
    """

    def __init__(self):
        super().__init__()
        # ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove fc layer, keep up to avgpool
        # resnet.avgpool output is (B, 2048, 1, 1)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])

        # EfficientNet-B0
        # num_classes=0 returns pooled features (B, 1280)
        self.effnet_backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )

    def forward(self, x):
        # ResNet features
        r_feat = self.resnet_backbone(x)
        r_feat = torch.flatten(r_feat, 1)  # (B, 2048)

        # EffNet features
        e_feat = self.effnet_backbone(x)  # (B, 1280)

        # Concatenate
        return torch.cat([r_feat, e_feat], dim=1)


class FeatureExtractor:
    def __init__(self, device=None):
        self.device = device if device else torch.device(Config.DEVICE)
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_model(self):
        print("Loading Dual-Backbone Model (ResNet50 + EfficientNet-B0)...")
        model = DualBackbone()
        model.to(self.device)
        model.eval()
        return model

    def process_dataset(
        self,
        metadata_path,
        bson_path,
        output_feat_path,
        output_label_path,
        is_test=False,
        load_cached=True,
    ):
        # 1. Check Cache
        if (
            load_cached
            and os.path.exists(output_feat_path)
            and os.path.exists(output_label_path)
        ):
            print(f"Loading cached features from {output_feat_path}...")
            return

        print(f"Starting feature extraction for {os.path.basename(metadata_path)}...")

        # 2. Setup Data
        dataset = BSONDataset(
            metadata_path, bson_path, transform=self.transform, is_test=is_test
        )
        loader = DataLoader(
            dataset,
            batch_size=Config.EXTRACT_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # 3. Setup Model
        model = self._load_model()

        # 4. Allocate Output Arrays
        # Dimensions: N_samples x (2048 + 1280)
        num_samples = len(dataset)
        feat_dim = Config.RESNET_DIM + Config.EFFNET_DIM

        # We use numpy arrays in memory since we have 220GB RAM
        # and the largest set (Train) is ~5M * 3328 * 4 bytes ~= 66GB
        all_features = np.zeros((num_samples, feat_dim), dtype=np.float32)
        all_labels = np.zeros((num_samples,), dtype=np.int64)

        # 5. Extraction Loop
        ptr = 0
        print_interval = max(1, len(loader) // 10)

        with torch.no_grad():
            for batch_idx, (images, batch_indices, targets) in enumerate(loader):
                images = images.to(self.device)
                batch_indices = batch_indices.to(self.device)

                # Forward Pass (Dual Backbone)
                features = model(images)  # (Total_Images, 3328)

                # Mean Pooling per Product
                # scatter_mean aggregates features based on batch_indices
                # Output shape: (Batch_Size, 3328)
                product_features = scatter_mean(features, batch_indices, dim=0)

                # Move to CPU
                batch_size = product_features.size(0)
                all_features[ptr : ptr + batch_size] = product_features.cpu().numpy()
                all_labels[ptr : ptr + batch_size] = targets.numpy()

                ptr += batch_size

                if (batch_idx + 1) % print_interval == 0:
                    print(f"Processed {batch_idx + 1}/{len(loader)} batches...")

        # 6. Save
        print(f"Saving features to {output_feat_path}...")
        np.save(output_feat_path, all_features)
        np.save(output_label_path, all_labels)
        print("Done.")


def run_feature_extraction(load_cached_data=True):
    """
    Main entry point for feature extraction.
    """
    set_seed(Config.SEED)

    extractor = FeatureExtractor()

    # 1. Train Set
    extractor.process_dataset(
        metadata_path=Config.TRAIN_META,
        bson_path=Config.TRAIN_BSON,
        output_feat_path=Config.TRAIN_FEATURES,
        output_label_path=Config.TRAIN_LABELS,
        is_test=False,
        load_cached=load_cached_data,
    )

    # 2. Validation Set
    extractor.process_dataset(
        metadata_path=Config.VAL_META,
        bson_path=Config.TRAIN_BSON,
        output_feat_path=Config.VAL_FEATURES,
        output_label_path=Config.VAL_LABELS,
        is_test=False,
        load_cached=load_cached_data,
    )

    # 3. Test Set
    extractor.process_dataset(
        metadata_path=Config.TEST_META,
        bson_path=Config.TEST_BSON,
        output_feat_path=Config.TEST_FEATURES,
        output_label_path=Config.TEST_IDS,
        is_test=True,
        load_cached=load_cached_data,
    )
