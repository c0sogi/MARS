import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# Import from library
from library.config import Config
from library.utils import BSONImageLoader, seed_everything

# Try importing scatter_mean, handle if not present (though it is listed)
try:
    from torch_scatter import scatter_mean
except ImportError:
    print("Warning: torch_scatter not found. Using naive fallback (slower).")

    def scatter_mean(src, index, dim=0, dim_size=None):
        # Naive implementation for fallback
        out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
        ones = torch.ones(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        count = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
        count.index_add_(0, index, torch.ones_like(src))
        return out / count.clamp(min=1)


class FeatureExtractor(nn.Module):
    """
    Dual-Backbone Model: ResNet50 + EfficientNet-B0.
    Extracts and concatenates features. Frozen weights.
    """

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # 1. ResNet50 (2048 dim)
        weights_resnet = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights_resnet)
        self.resnet.fc = nn.Identity()  # Remove classification head

        # 2. EfficientNet-B0 (1280 dim)
        weights_effnet = models.EfficientNet_B0_Weights.DEFAULT
        self.effnet = models.efficientnet_b0(weights=weights_effnet)
        self.effnet.classifier = nn.Identity()  # Remove classification head

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: (N_images, 3, 224, 224)

        # ResNet features
        f1 = self.resnet(x)  # (N, 2048)

        # EfficientNet features
        f2 = self.effnet(x)  # (N, 1280)

        # Concatenate
        out = torch.cat([f1, f2], dim=1)  # (N, 3328)
        return out


class ProductDataset(Dataset):
    """
    Dataset that reads images from BSON using metadata.
    """

    def __init__(self, metadata_df, bson_path, transform=None):
        self.records = metadata_df.to_dict("records")
        self.bson_path = bson_path
        self.transform = transform
        self.loader = None  # Initialized lazily in worker

    def _get_loader(self):
        if self.loader is None:
            self.loader = BSONImageLoader(self.bson_path)
        return self.loader

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        loader = self._get_loader()
        row = self.records[idx]

        offset = row["bson_offset"]
        length = row["bson_length"]

        # Load images
        try:
            imgs = loader.load_images(offset, length)
        except Exception:
            imgs = []

        # Handle empty or corrupt records
        if len(imgs) == 0:
            # Return a black image
            imgs = [np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)]

        processed_imgs = []
        for img in imgs:
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            processed_imgs.append(img)

        # Stack images: (K, C, H, W)
        imgs_tensor = torch.stack(processed_imgs)

        # Get ID and Label
        _id = row["_id"]
        category_id = row.get("category_id", -1)  # -1 for test set

        return imgs_tensor, category_id, _id


def collate_fn(batch):
    """
    Custom collate to handle variable number of images per product.
    Flattens images into a single batch and creates indices for aggregation.
    """
    all_imgs = []
    batch_indices = []
    labels = []
    ids = []

    for i, (imgs, label, pid) in enumerate(batch):
        all_imgs.append(imgs)
        # Create index vector [i, i, ...] for the K images of this product
        k = imgs.shape[0]
        batch_indices.append(torch.full((k,), i, dtype=torch.long))
        labels.append(label)
        ids.append(pid)

    # Concatenate all images: (Total_Batch_Images, C, H, W)
    flat_imgs = torch.cat(all_imgs, dim=0)

    # Concatenate indices: (Total_Batch_Images,)
    flat_indices = torch.cat(batch_indices, dim=0)

    # Labels and IDs
    labels = torch.tensor(labels, dtype=torch.long)
    ids = torch.tensor(ids, dtype=torch.long)

    return flat_imgs, flat_indices, labels, ids


class FeatureEngine:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        seed_everything(Config.SEED)

        # Preprocessing
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _extract_dataset(self, df, bson_path, desc="Extracting"):
        """
        Runs inference on a dataset and returns features, labels, and ids.
        """
        dataset = ProductDataset(df, bson_path, transform=self.transform)
        loader = DataLoader(
            dataset,
            batch_size=Config.EXTRACT_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Pre-allocate memory using numpy to avoid list overhead
        num_samples = len(df)
        feature_dim = Config.INPUT_DIM  # 3328

        all_features = np.zeros((num_samples, feature_dim), dtype=np.float32)
        all_labels = np.zeros((num_samples,), dtype=np.int64)
        all_ids = np.zeros((num_samples,), dtype=np.int64)

        model = FeatureExtractor().to(self.device)
        model.eval()

        ptr = 0
        with torch.no_grad():
            for (
                imgs,
                indices,
                labels,
                pids,
            ) in loader:  # tqdm(loader, desc=desc, mininterval=10):
                imgs = imgs.to(self.device)
                indices = indices.to(self.device)

                # Forward pass
                features = model(imgs)  # (Total_Images, 3328)

                # Aggregate per product (Mean Pooling)
                # dim_size ensures we get exactly batch_size outputs even if some are missing (unlikely)
                batch_size = labels.size(0)
                product_features = scatter_mean(
                    features, indices, dim=0, dim_size=batch_size
                )

                # Store in numpy arrays
                batch_len = batch_size
                all_features[ptr : ptr + batch_len] = product_features.cpu().numpy()
                all_labels[ptr : ptr + batch_len] = labels.numpy()
                all_ids[ptr : ptr + batch_len] = pids.numpy()

                ptr += batch_len

        return all_features, all_labels, all_ids

    def generate_features(self, load_cached_data=True):
        """
        Main pipeline to generate or load features.
        """
        # Define paths
        paths = {
            "train_feat": Config.TRAIN_FEATURES,
            "train_lbl": Config.TRAIN_LABELS,
            "val_feat": Config.VAL_FEATURES,
            "val_lbl": Config.VAL_LABELS,
            "test_feat": Config.TEST_FEATURES,
            "test_ids": Config.TEST_IDS,
        }

        # Check if all exist
        all_exist = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and all_exist:
            print("Loading cached features from disk...")
            # We don't necessarily need to load them into RAM here if the caller handles it,
            # but usually this function ensures they are available.
            # To save memory, we won't return them. The training script will load them via mmap.
            return

        print("Extracting features from scratch...")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Load Metadata
        print("Loading metadata...")
        train_df = pd.read_csv(Config.TRAIN_META)
        val_df = pd.read_csv(Config.VAL_META)
        test_df = pd.read_csv(Config.TEST_META)

        # Debug Mode
        if Config.DEBUG:
            print(f"DEBUG MODE: Limiting to {Config.DEBUG_SAMPLES} samples.")
            train_df = train_df.iloc[: Config.DEBUG_SAMPLES]
            val_df = val_df.iloc[: Config.DEBUG_SAMPLES]
            test_df = test_df.iloc[: Config.DEBUG_SAMPLES]

        # 1. Process Train
        print(f"Processing Train Set ({len(train_df)} samples)...")
        train_feats, train_lbls, _ = self._extract_dataset(
            train_df, Config.TRAIN_BSON, "Train"
        )
        np.save(paths["train_feat"], train_feats)
        np.save(paths["train_lbl"], train_lbls)
        del train_feats, train_lbls  # Free RAM

        # 2. Process Val
        print(f"Processing Val Set ({len(val_df)} samples)...")
        val_feats, val_lbls, _ = self._extract_dataset(val_df, Config.TRAIN_BSON, "Val")
        np.save(paths["val_feat"], val_feats)
        np.save(paths["val_lbl"], val_lbls)
        del val_feats, val_lbls

        # 3. Process Test
        print(f"Processing Test Set ({len(test_df)} samples)...")
        test_feats, _, test_ids = self._extract_dataset(
            test_df, Config.TEST_BSON, "Test"
        )
        np.save(paths["test_feat"], test_feats)
        np.save(paths["test_ids"], test_ids)
        del test_feats, test_ids

        print("Feature extraction complete. Files saved to working directory.")


if __name__ == "__main__":
    # For testing purposes only
    engine = FeatureEngine()
    engine.generate_features(load_cached_data=True)
