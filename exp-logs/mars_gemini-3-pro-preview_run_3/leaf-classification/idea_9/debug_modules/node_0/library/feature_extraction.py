import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from library.config import Config
from library.data_processing import LeafDataset, get_transforms


class FeatureExtractor:
    """
    Handles the extraction of visual features from DINOv2 and ConvNeXt models.
    Implements multi-view averaging and caching.
    """

    def __init__(self, device=Config.DEVICE):
        """
        Initialize the FeatureExtractor.

        Args:
            device (str): Device to run the models on ('cuda' or 'cpu').
        """
        self.device = device
        self.dino_model = None
        self.convnext_model = None

    def _load_models(self):
        """
        Lazily loads the pre-trained models onto the specified device.
        """
        if self.dino_model is not None and self.convnext_model is not None:
            return

        print(f"Loading models to {self.device}...")

        # Load DINOv2 (Stream A: Global Geometry)
        # We use the base model to get the last hidden state
        self.dino_model = AutoModel.from_pretrained(Config.MODEL_DINO)
        self.dino_model.to(self.device)
        self.dino_model.eval()

        # Load ConvNeXt (Stream B: Local Texture)
        # We use the base model to get the pooled output
        self.convnext_model = AutoModel.from_pretrained(Config.MODEL_CONVNEXT)
        self.convnext_model.to(self.device)
        self.convnext_model.eval()

        print("Models loaded successfully.")

    def extract_features(
        self,
        df,
        split_name,
        class_to_idx=None,
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
    ):
        """
        Extracts features for the provided dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing image paths and metadata.
            split_name (str): Name of the split (e.g., 'train', 'val', 'test') for cache naming.
            class_to_idx (dict, optional): Mapping for species labels. Required if 'species' is in df.
            load_cached_data (bool): Whether to load features from cache if available.
            batch_size (int): Batch size for inference.

        Returns:
            tuple: (dino_features, conv_features, ids)
                dino_features (np.ndarray): (N, 1024)
                conv_features (np.ndarray): (N, 1536)
                ids (np.ndarray): (N,)
        """
        # Define cache paths
        cache_dir = Config.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        dino_path = os.path.join(cache_dir, f"{split_name}_dino_features.npy")
        conv_path = os.path.join(cache_dir, f"{split_name}_conv_features.npy")
        ids_path = os.path.join(cache_dir, f"{split_name}_ids.npy")

        # 1. Check Cache
        if load_cached_data:
            if (
                os.path.exists(dino_path)
                and os.path.exists(conv_path)
                and os.path.exists(ids_path)
            ):
                print(f"Loading cached features for '{split_name}' from {cache_dir}...")
                dino_features = np.load(dino_path)
                conv_features = np.load(conv_path)
                ids = np.load(ids_path)
                return dino_features, conv_features, ids
            else:
                print(
                    f"Cache missing for '{split_name}'. Starting feature extraction..."
                )
        else:
            print(f"Forcing re-computation of features for '{split_name}'...")

        # 2. Setup Data Loading
        self._load_models()

        transforms = get_transforms()
        dataset = LeafDataset(
            df=df,
            transforms=transforms,
            class_to_idx=class_to_idx,
            split_name=split_name,
            load_cached_data=load_cached_data,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 3. Inference Loop
        all_dino_feats = []
        all_conv_feats = []
        all_ids = []

        print(f"Processing {len(dataset)} images...")

        with torch.no_grad():
            for batch in dataloader:
                # Inputs: (Batch, Views=4, Channels, Height, Width)
                images = batch["images"].to(self.device)
                batch_ids = batch["id"].numpy()

                b, v, c, h, w = images.shape

                # Flatten views into batch dimension for parallel inference
                # Shape: (Batch * 4, C, H, W)
                flat_images = images.view(b * v, c, h, w)

                # --- Stream A: DINOv2 ---
                # Forward pass
                dino_outputs = self.dino_model(flat_images)
                # Extract CLS token (index 0). Shape: (Batch * 4, Hidden_Dim)
                dino_cls = dino_outputs.last_hidden_state[:, 0, :]

                # --- Stream B: ConvNeXt ---
                # Forward pass
                conv_outputs = self.convnext_model(flat_images)
                # Use pooler_output if available, else Global Average Pool
                if (
                    hasattr(conv_outputs, "pooler_output")
                    and conv_outputs.pooler_output is not None
                ):
                    conv_emb = conv_outputs.pooler_output
                else:
                    # Fallback: Mean over spatial dimensions (H, W)
                    # last_hidden_state shape: (Batch*4, C, H, W)
                    conv_emb = conv_outputs.last_hidden_state.mean(dim=[-2, -1])

                # --- Aggregation ---
                # Reshape back to (Batch, Views, Dim) and average across views
                dino_avg = dino_cls.view(b, v, -1).mean(dim=1)
                conv_avg = conv_emb.view(b, v, -1).mean(dim=1)

                # Store results
                all_dino_feats.append(dino_avg.cpu().numpy())
                all_conv_feats.append(conv_avg.cpu().numpy())
                all_ids.append(batch_ids)

        # 4. Consolidate and Cache
        dino_features = np.concatenate(all_dino_feats, axis=0)
        conv_features = np.concatenate(all_conv_feats, axis=0)
        ids = np.concatenate(all_ids, axis=0)

        print(f"Saving features to {cache_dir}...")
        np.save(dino_path, dino_features)
        np.save(conv_path, conv_features)
        np.save(ids_path, ids)

        return dino_features, conv_features, ids
