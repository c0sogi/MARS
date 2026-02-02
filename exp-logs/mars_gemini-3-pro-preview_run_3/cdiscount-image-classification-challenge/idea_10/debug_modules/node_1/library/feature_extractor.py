import os
import torch
import torch.nn as nn
import numpy as np
import timm
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean

from library.config import Config
from library.datasets import RawImageDataset

# Batch size for feature extraction (inference)
# Adjusted for A100 GPU memory with dual backbones
INFERENCE_BATCH_SIZE = 64


class DualBackbone(nn.Module):
    """
    Dual-Backbone Feature Extractor.
    Combines ResNet50 and EfficientNet-B0 features.
    Output Dimension: 2048 (ResNet) + 1280 (EffNet) = 3328.
    """

    def __init__(self):
        super(DualBackbone, self).__init__()
        # Load ResNet50 - Output dim 2048
        # num_classes=0 returns the pooled feature vector
        self.resnet = timm.create_model("resnet50", pretrained=True, num_classes=0)

        # Load EfficientNet-B0 - Output dim 1280
        self.effnet = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )

        # Freeze parameters to save memory and compute
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.effnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x shape: (N_images, 3, H, W)
        res_feat = self.resnet(x)  # (N, 2048)
        eff_feat = self.effnet(x)  # (N, 1280)

        # Concatenate features
        return torch.cat([res_feat, eff_feat], dim=1)  # (N, 3328)


def collate_fn(batch):
    """
    Custom collate function to handle variable number of images per product.
    Flattens the batch of images and creates indices for scatter reduction.
    """
    all_imgs = []
    batch_indices = []
    ids = []
    cats = []

    for i, (imgs, _id, cat) in enumerate(batch):
        # imgs shape: (N_imgs_i, 3, H, W)
        all_imgs.append(imgs)

        # Create indices mapping these images to the current product index 'i'
        # If product 'i' has 3 images, we append [i, i, i]
        batch_indices.append(torch.full((imgs.shape[0],), i, dtype=torch.long))

        ids.append(_id)
        cats.append(cat)

    # Concatenate all images into a single large batch
    flat_imgs = torch.cat(all_imgs, dim=0)
    batch_indices = torch.cat(batch_indices, dim=0)

    ids = torch.tensor(ids, dtype=torch.long)
    cats = torch.tensor(cats, dtype=torch.long)

    return flat_imgs, batch_indices, ids, cats


def extract_dataset(dataset, model, device, desc="Extracting"):
    """
    Iterates over the dataset, extracts features, and aggregates them per product.
    Returns numpy arrays of features and metadata.
    """
    loader = DataLoader(
        dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Pre-allocate memory for features
    num_samples = len(dataset)
    feature_dim = Config.INPUT_DIM  # 3328

    # Use float32 for features
    features_array = np.zeros((num_samples, feature_dim), dtype=np.float32)

    # Metadata lists
    ids_list = []
    cats_list = []

    start_idx = 0

    model.eval()
    with torch.no_grad():
        for flat_imgs, batch_indices, ids, cats in loader:
            flat_imgs = flat_imgs.to(device)
            batch_indices = batch_indices.to(device)

            # Extract features for all images in the batch
            raw_feats = model(flat_imgs)  # (Total_Imgs_In_Batch, 3328)

            # Aggregate per product (Mean Pooling)
            # dim_size ensures the output has size equal to the batch size (number of products)
            pooled_feats = scatter_mean(
                raw_feats, batch_indices, dim=0, dim_size=len(ids)
            )

            # Store in numpy array
            batch_size_actual = pooled_feats.shape[0]
            end_idx = start_idx + batch_size_actual

            features_array[start_idx:end_idx] = pooled_feats.cpu().numpy()

            ids_list.extend(ids.numpy())
            cats_list.extend(cats.numpy())

            start_idx = end_idx

    return features_array, np.array(ids_list), np.array(cats_list)


def extract_and_save_features(load_cached_data=True, subset_size=None):
    """
    Main function to manage feature extraction for Train, Val, and Test sets.

    Args:
        load_cached_data (bool): If True, skips extraction if files exist.
        subset_size (int, optional): Number of records to process (for debugging).
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define tasks: (Name, MetaPath, BsonPath, FeaturePath, LabelPath/IdPath, IsTest)
    tasks = [
        (
            "Train",
            Config.TRAIN_META,
            Config.TRAIN_BSON,
            Config.TRAIN_FEATURES,
            Config.TRAIN_LABELS,
            False,
        ),
        (
            "Val",
            Config.VAL_META,
            Config.TRAIN_BSON,
            Config.VAL_FEATURES,
            Config.VAL_LABELS,
            False,
        ),
        (
            "Test",
            Config.TEST_META,
            Config.TEST_BSON,
            Config.TEST_FEATURES,
            Config.TEST_IDS,
            True,
        ),
    ]

    # Check if all files exist to potentially skip initialization
    all_exist = True
    if load_cached_data:
        for _, _, _, feat_path, meta_path, _ in tasks:
            if not (os.path.exists(feat_path) and os.path.exists(meta_path)):
                all_exist = False
                break
    else:
        all_exist = False

    if all_exist:
        print("All feature files found in cache. Skipping extraction.")
        return

    # Initialize Model only if we need to extract
    print("Initializing DualBackbone model...")
    device = torch.device(Config.DEVICE)
    model = DualBackbone()
    model.to(device)
    model.eval()

    for name, meta_path, bson_path, feat_path, meta_out_path, is_test in tasks:
        # Check cache for this specific task
        if (
            load_cached_data
            and os.path.exists(feat_path)
            and os.path.exists(meta_out_path)
        ):
            print(f"Cached features found for {name}. Skipping.")
            continue

        print(f"Starting feature extraction for {name} set...")

        # Initialize Dataset
        dataset = RawImageDataset(
            metadata_path=meta_path, bson_path=bson_path, subset_size=subset_size
        )

        # Run Extraction
        feats, ids, cats = extract_dataset(dataset, model, device, desc=name)

        # Save Results
        print(f"Saving {name} features to {feat_path}...")
        np.save(feat_path, feats)

        if is_test:
            print(f"Saving {name} IDs to {meta_out_path}...")
            np.save(meta_out_path, ids)
        else:
            print(f"Saving {name} Labels to {meta_out_path}...")
            np.save(meta_out_path, cats)

    print("Feature extraction pipeline completed.")
