import os
import numpy as np
import torch
import timm
from torch.utils.data import DataLoader
from library.config import Config
from library.data_utils import LeafImageDataset


# Set seeds for reproducibility
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(Config.SEED)


class DualStreamExtractor:
    """
    Extracts features using DINOv2 (Global Geometry) and ConvNeXt (Local Texture).
    Processes 4 views per image.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        # Initialize DINOv2 (Global Geometry Stream)
        print(f"Initializing DINOv2 model: {Config.DINO_MODEL_NAME}")
        self.dino_model = timm.create_model(
            Config.DINO_MODEL_NAME,
            pretrained=True,
            num_classes=0,  # Get embeddings (Global Average Pooling or CLS token)
        )
        self.dino_model.to(self.device)
        self.dino_model.eval()

        # Initialize ConvNeXt (Local Texture Stream)
        print(f"Initializing ConvNeXt model: {Config.CONVNEXT_MODEL_NAME}")
        self.convnext_model = timm.create_model(
            Config.CONVNEXT_MODEL_NAME,
            pretrained=True,
            num_classes=0,  # Get embeddings (Global Average Pooling)
        )
        self.convnext_model.to(self.device)
        self.convnext_model.eval()

    def _get_cache_paths(self, prefix):
        """Helper to generate cache file paths."""
        base = os.path.join(Config.WORKING_DIR, prefix)
        return {
            "embeddings": f"{base}_embeddings.npy",  # Shape: (N, 4, EmbedDim)
            "tabular": f"{base}_tabular.npy",  # Shape: (N, 192)
            "ids": f"{base}_ids.npy",  # Shape: (N,)
            "labels": f"{base}_labels.npy",  # Shape: (N,) - optional
        }

    def extract_features(self, metadata_path, dataset_key, load_cached_data=True):
        """
        Extracts features for a given dataset (train/val/test).

        Args:
            metadata_path (str): Path to the metadata CSV.
            dataset_key (str): Identifier for caching (e.g., 'train', 'val', 'test').
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: Dictionary containing 'embeddings', 'tabular', 'ids', and optionally 'labels'.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        paths = self._get_cache_paths(dataset_key)

        # 1. Try Loading from Cache
        if load_cached_data:
            files_exist = (
                os.path.exists(paths["embeddings"])
                and os.path.exists(paths["tabular"])
                and os.path.exists(paths["ids"])
            )

            if files_exist:
                print(
                    f"Loading cached features for '{dataset_key}' from {Config.WORKING_DIR}..."
                )
                try:
                    data = {
                        "embeddings": np.load(paths["embeddings"]),
                        "tabular": np.load(paths["tabular"]),
                        "ids": np.load(paths["ids"]),
                    }
                    # Load labels if they exist (train/val sets)
                    if os.path.exists(paths["labels"]):
                        data["labels"] = np.load(paths["labels"], allow_pickle=True)
                    return data
                except Exception as e:
                    print(f"Failed to load cache: {e}. Recomputing...")
            else:
                print(f"Cache missing for '{dataset_key}'. Computing features...")
        else:
            print(f"Force recompute for '{dataset_key}'...")

        # 2. Compute Features
        print(f"Starting feature extraction for {dataset_key}...")
        dataset = LeafImageDataset(metadata_path, return_labels=True)
        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if Config.DEVICE == "cuda" else False,
        )

        all_embeddings = []
        all_tabular = []
        all_ids = []
        all_labels = []
        has_labels = False

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Unpack batch
                # batch["dino_views"] shape: (B, 4, 3, 518, 518)
                # batch["convnext_views"] shape: (B, 4, 3, 384, 384)
                dino_imgs = batch["dino_views"].to(self.device)
                convnext_imgs = batch["convnext_views"].to(self.device)
                tabular = batch["tabular"].numpy()
                ids = batch["id"].numpy()

                # Check labels
                if "label" in batch:
                    labels = batch["label"]  # list of strings
                    all_labels.extend(labels)
                    has_labels = True

                # Flatten views into batch dimension for inference
                # (B, 4, 3, H, W) -> (B*4, 3, H, W)
                B, V, C_d, H_d, W_d = dino_imgs.shape
                dino_input = dino_imgs.view(B * V, C_d, H_d, W_d)

                _, _, C_c, H_c, W_c = convnext_imgs.shape
                convnext_input = convnext_imgs.view(B * V, C_c, H_c, W_c)

                # Inference
                # DINOv2
                dino_feats = self.dino_model(dino_input)  # (B*4, 1024)

                # ConvNeXt
                convnext_feats = self.convnext_model(convnext_input)  # (B*4, 1536)

                # Concatenate features
                # (B*4, 1024 + 1536) = (B*4, 2560)
                fused_feats = torch.cat([dino_feats, convnext_feats], dim=1)

                # Reshape back to (B, 4, FeatureDim)
                fused_feats = fused_feats.view(B, V, -1)

                # Move to CPU and store
                all_embeddings.append(fused_feats.cpu().numpy())
                all_tabular.append(tabular)
                all_ids.append(ids)

                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

        # Concatenate all batches
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        final_tabular = np.concatenate(all_tabular, axis=0)
        final_ids = np.concatenate(all_ids, axis=0)

        # 3. Save to Cache
        print(f"Saving features for '{dataset_key}' to {Config.WORKING_DIR}...")
        np.save(paths["embeddings"], final_embeddings)
        np.save(paths["tabular"], final_tabular)
        np.save(paths["ids"], final_ids)

        result = {
            "embeddings": final_embeddings,
            "tabular": final_tabular,
            "ids": final_ids,
        }

        if has_labels:
            final_labels = np.array(all_labels)
            np.save(paths["labels"], final_labels)
            result["labels"] = final_labels

        print(
            f"Feature extraction complete for {dataset_key}. Embedding Shape: {final_embeddings.shape}"
        )
        return result
