import os
import numpy as np
import torch
import timm
from library.config import Config
from library.utils import seed_everything


class DualStreamExtractor:
    """
    Extracts features using DINOv2 (Global Geometry) and ConvNeXt (Local Texture) backbones.
    Performs Multi-View Canonical Averaging for rotation invariance.
    """

    def __init__(self):
        self.device = Config.DEVICE

        print(f"Initializing DINOv2: {Config.MODEL_DINOV2}")
        self.dino = timm.create_model(
            Config.MODEL_DINOV2,
            pretrained=True,
            num_classes=0,
            img_size=Config.IMAGE_SIZE,
        )

        print(f"Initializing ConvNeXt: {Config.MODEL_CONVNEXT}")
        self.conv = timm.create_model(
            Config.MODEL_CONVNEXT, pretrained=True, num_classes=0
        )

        self.dino.to(self.device)
        self.conv.to(self.device)

        self.dino.eval()
        self.conv.eval()

    def extract_features(self, dataloader):
        """
        Iterates over the dataloader, runs forward passes, and aggregates features.

        Args:
            dataloader: PyTorch DataLoader yielding (stacked_views, tabular, label, id)

        Returns:
            Tuple of numpy arrays: (dino_feats, conv_feats, tab_feats, labels, ids)
        """
        dino_feats_list = []
        conv_feats_list = []
        tab_feats_list = []
        labels_list = []
        ids_list = []

        # Disable gradient calculation for inference
        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch
                # stacked_views: (B, 4, 3, H, W)
                # tabular: (B, 192)
                # label: (B,)
                # image_id: (B,)
                stacked_views, tabular, label, image_id = batch

                stacked_views = stacked_views.to(self.device)

                # Dimensions
                B, V, C, H, W = stacked_views.shape

                # Flatten Batch and View dimensions for efficient processing
                # Input becomes (B*4, 3, H, W)
                flat_views = stacked_views.view(B * V, C, H, W)

                # --- Forward Pass: DINOv2 ---
                # Output: (B*4, EmbedDim_Dino)
                out_dino = self.dino(flat_views)

                # --- Forward Pass: ConvNeXt ---
                # Output: (B*4, EmbedDim_Conv)
                out_conv = self.conv(flat_views)

                # --- Multi-View Averaging ---
                # Reshape back to (B, V, EmbedDim)
                out_dino = out_dino.view(B, V, -1)
                out_conv = out_conv.view(B, V, -1)

                # Mean over views (dim 1) -> (B, EmbedDim)
                avg_dino = out_dino.mean(dim=1)
                avg_conv = out_conv.mean(dim=1)

                # Store results (move to CPU numpy)
                dino_feats_list.append(avg_dino.cpu().numpy())
                conv_feats_list.append(avg_conv.cpu().numpy())
                tab_feats_list.append(tabular.numpy())
                labels_list.append(label.numpy())
                ids_list.append(image_id.numpy())

        # Concatenate all batches
        return (
            np.concatenate(dino_feats_list, axis=0),
            np.concatenate(conv_feats_list, axis=0),
            np.concatenate(tab_feats_list, axis=0),
            np.concatenate(labels_list, axis=0),
            np.concatenate(ids_list, axis=0),
        )


def process_split(dataloader, split_name, load_cached_data=True):
    """
    Handles feature extraction with caching mechanism.

    Args:
        dataloader: DataLoader for the specific split.
        split_name: String identifier (e.g., 'train', 'val', 'test').
        load_cached_data: If True, attempts to load from disk first.

    Returns:
        Tuple of numpy arrays: (X_dino, X_conv, X_tab, y, ids)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define file paths
    files = {
        "dino": os.path.join(Config.CACHE_DIR, f"{split_name}_dino.npy"),
        "conv": os.path.join(Config.CACHE_DIR, f"{split_name}_conv.npy"),
        "tab": os.path.join(Config.CACHE_DIR, f"{split_name}_tab.npy"),
        "y": os.path.join(Config.CACHE_DIR, f"{split_name}_y.npy"),
        "ids": os.path.join(Config.CACHE_DIR, f"{split_name}_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in files.values())

    if load_cached_data and cache_exists:
        print(f"[{split_name}] Loading cached features from {Config.CACHE_DIR}...")
        X_dino = np.load(files["dino"])
        X_conv = np.load(files["conv"])
        X_tab = np.load(files["tab"])
        y = np.load(files["y"])
        ids = np.load(files["ids"])
        return X_dino, X_conv, X_tab, y, ids

    # If not cached or reload forced, run extraction
    print(f"[{split_name}] Extracting features (DINOv2 + ConvNeXt)...")
    extractor = DualStreamExtractor()
    X_dino, X_conv, X_tab, y, ids = extractor.extract_features(dataloader)

    # Save to cache
    print(f"[{split_name}] Saving features to cache...")
    np.save(files["dino"], X_dino)
    np.save(files["conv"], X_conv)
    np.save(files["tab"], X_tab)
    np.save(files["y"], y)
    np.save(files["ids"], ids)

    return X_dino, X_conv, X_tab, y, ids
