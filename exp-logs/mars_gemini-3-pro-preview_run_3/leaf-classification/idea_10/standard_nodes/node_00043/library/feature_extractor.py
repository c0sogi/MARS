import os
import torch
import numpy as np
from transformers import AutoModel
from library import config


class DualStreamExtractor:
    """
    Extracts features using DINOv2 (Global Geometry) and ConvNeXt (Local Texture).
    Performs multi-view canonical averaging for rotation invariance.
    """

    def __init__(self, device=None):
        """
        Initialize the DualStreamExtractor with pretrained models.

        Args:
            device (torch.device, optional): Device to load models on. Defaults to CUDA if available.
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"Initializing DualStreamExtractor on {self.device}...")

        # Load DINOv2 (ViT-Large)
        # Captures global geometric priors
        print(f"Loading DINOv2: {config.MODEL_DINOV2}")
        self.dino_model = AutoModel.from_pretrained(config.MODEL_DINOV2)
        self.dino_model.to(self.device)
        self.dino_model.eval()

        # Load ConvNeXt (Large)
        # Captures local texture details
        print(f"Loading ConvNeXt: {config.MODEL_CONVNEXT}")
        self.conv_model = AutoModel.from_pretrained(config.MODEL_CONVNEXT)
        self.conv_model.to(self.device)
        self.conv_model.eval()

    def _process_batch(self, images):
        """
        Process a batch of multi-view images to extract rotation-invariant features.

        Args:
            images: Tensor of shape (B, 4, 3, H, W)

        Returns:
            dino_feats: Tensor of shape (B, D_dino)
            conv_feats: Tensor of shape (B, D_conv)
        """
        B, V, C, H, W = images.shape

        # Flatten batch and views for parallel inference: (B*V, 3, H, W)
        flat_images = images.view(B * V, C, H, W).to(self.device, non_blocking=True)

        with torch.no_grad():
            # --- Stream 1: DINOv2 ---
            # DINOv2 uses the CLS token (index 0) for global representation
            # Output shape: (B*V, Seq, Dim)
            dino_out = self.dino_model(flat_images).last_hidden_state
            # Extract CLS token: (B*V, Dim)
            dino_emb = dino_out[:, 0, :]

            # --- Stream 2: ConvNeXt ---
            # ConvNeXt base model outputs feature map (B*V, Dim, H', W')
            conv_out = self.conv_model(flat_images).last_hidden_state
            # Perform Global Average Pooling: (B*V, Dim)
            conv_emb = conv_out.mean(dim=[-2, -1])

        # Reshape to (B, V, Dim) and Average over Views to enforce rotation invariance
        dino_feats = dino_emb.view(B, V, -1).mean(dim=1)
        conv_feats = conv_emb.view(B, V, -1).mean(dim=1)

        return dino_feats.cpu(), conv_feats.cpu()

    def extract_features(self, dataloader, split_name, load_cached_data=True):
        """
        Extracts features for the entire dataloader.
        Manages caching of features to disk to avoid redundant computation.

        Args:
            dataloader: PyTorch DataLoader yielding (images, tabular, [label], id)
            split_name: String identifier for the split (e.g., 'train', 'val', 'test')
            load_cached_data: Boolean, whether to load from cache if available.

        Returns:
            dict: {
                'dino_features': np.ndarray,
                'conv_features': np.ndarray,
                'ids': np.ndarray,
                'labels': np.ndarray (or None if not present)
            }
        """
        # Ensure cache directory exists
        cache_dir = config.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        # Define cache file paths
        paths = {
            "dino": os.path.join(cache_dir, f"{split_name}_dino_features.npy"),
            "conv": os.path.join(cache_dir, f"{split_name}_conv_features.npy"),
            "ids": os.path.join(cache_dir, f"{split_name}_ids.npy"),
            "labels": os.path.join(cache_dir, f"{split_name}_labels.npy"),
        }

        # Attempt to load from cache
        if load_cached_data:
            files_exist = (
                os.path.exists(paths["dino"])
                and os.path.exists(paths["conv"])
                and os.path.exists(paths["ids"])
            )

            if files_exist:
                print(f"Loading cached features for '{split_name}' from {cache_dir}...")
                data = {
                    "dino_features": np.load(paths["dino"]),
                    "conv_features": np.load(paths["conv"]),
                    "ids": np.load(paths["ids"]),
                    "labels": (
                        np.load(paths["labels"])
                        if os.path.exists(paths["labels"])
                        else None
                    ),
                }
                return data

        print(
            f"Extracting features for '{split_name}' (Cache miss or force refresh)..."
        )

        dino_list = []
        conv_list = []
        id_list = []
        label_list = []
        has_labels = False

        # Iterate over dataloader
        total_batches = len(dataloader)
        for i, batch in enumerate(dataloader):
            if (i + 1) % 10 == 0:
                print(f"Processing batch {i + 1}/{total_batches}...")

            # Unpack batch dynamically based on whether labels are present
            # LeafDataset returns: (images, tab, label, id) OR (images, tab, id)
            if len(batch) == 4:
                images, _, labels, ids = batch
                label_list.append(labels.numpy())
                has_labels = True
            else:
                images, _, ids = batch

            # Extract features
            d_feats, c_feats = self._process_batch(images)

            dino_list.append(d_feats.numpy())
            conv_list.append(c_feats.numpy())
            id_list.append(ids.numpy())

        # Concatenate all batches
        dino_features = np.concatenate(dino_list, axis=0)
        conv_features = np.concatenate(conv_list, axis=0)
        ids_arr = np.concatenate(id_list, axis=0)

        labels_arr = None
        if has_labels:
            labels_arr = np.concatenate(label_list, axis=0)

        # Save to cache
        print(f"Saving extracted features to {cache_dir}...")
        np.save(paths["dino"], dino_features)
        np.save(paths["conv"], conv_features)
        np.save(paths["ids"], ids_arr)
        if labels_arr is not None:
            np.save(paths["labels"], labels_arr)

        return {
            "dino_features": dino_features,
            "conv_features": conv_features,
            "ids": ids_arr,
            "labels": labels_arr,
        }
