import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel
from library.config import Config
from library.utils import seed_everything
from library.data_factory import get_loader


class BackboneExtractor:
    """
    Handles feature extraction for the defined backbones.
    Manages caching, model loading, and inference with feature-space augmentation.
    """

    def __init__(self):
        self.device = Config.DEVICE
        seed_everything(Config.SEED)

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def _get_file_paths(self, backbone_name, subset_name):
        """Generates file paths for cached artifacts."""
        base_path = os.path.join(Config.CACHE_DIR, f"{backbone_name}_{subset_name}")
        return {
            "features": f"{base_path}_features.npy",
            "ids": f"{base_path}_ids.npy",
            "meta": f"{base_path}_meta.npy",
            "targets": f"{base_path}_targets.npy",
        }

    def _load_model(self, backbone_name):
        """Loads the specific HuggingFace model for the backbone."""
        model_config = Config.BACKBONES[backbone_name]
        model_name = model_config["model_name"]

        print(f"Loading model: {model_name}...")
        try:
            model = AutoModel.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    def _compute_embedding(self, model, pixel_values, backbone_name):
        """
        Performs the forward pass specific to the backbone architecture.
        Returns the embedding vector.
        """
        with torch.no_grad():
            pixel_values = pixel_values.to(self.device)

            if backbone_name == "siglip":
                # SigLIP: Use get_image_features for the projected embedding
                # Note: AutoModel for SigLIP usually loads SiglipModel
                if hasattr(model, "get_image_features"):
                    embeddings = model.get_image_features(pixel_values=pixel_values)
                else:
                    # Fallback if loaded as VisionModel
                    outputs = model(pixel_values=pixel_values)
                    # SigLIP vision model output is (B, Seq, D), usually pooled or use CLS equivalent?
                    # Standard SiglipModel uses attention pooling or similar.
                    # If we used AutoModel.from_pretrained("google/siglip..."), we get SiglipModel.
                    # get_image_features is the correct API.
                    raise AttributeError(
                        "Model does not have get_image_features. Ensure full SiglipModel is loaded."
                    )

            elif backbone_name == "dinov2":
                # DINOv2: Use the CLS token from the last hidden state
                outputs = model(pixel_values=pixel_values)
                # last_hidden_state shape: (Batch, Seq_Len, Hidden_Dim)
                # CLS token is at index 0
                embeddings = outputs.last_hidden_state[:, 0, :]

            elif backbone_name == "convnext":
                # ConvNeXt: Use the pooler_output (Global Average Pooling)
                outputs = model(pixel_values=pixel_values)
                embeddings = outputs.pooler_output

            else:
                raise ValueError(
                    f"Unknown backbone extraction logic for {backbone_name}"
                )

            return embeddings

    def extract(
        self,
        df: pd.DataFrame,
        backbone_name: str,
        subset_name: str,
        load_cached_data: bool = True,
    ):
        """
        Extracts features for a given dataframe and backbone.

        Args:
            df: DataFrame containing metadata and file paths.
            backbone_name: Key from Config.BACKBONES.
            subset_name: Identifier for the subset (e.g., 'train', 'val', 'test').
            load_cached_data: If True, attempts to load from disk first.

        Returns:
            Dictionary containing 'features', 'ids', 'meta', 'targets' as numpy arrays.
        """
        paths = self._get_file_paths(backbone_name, subset_name)

        # 1. Check Cache
        if load_cached_data:
            files_exist = all(os.path.exists(p) for p in paths.values())
            if files_exist:
                print(f"[{backbone_name}] Loading cached features for {subset_name}...")
                try:
                    data = {k: np.load(v, allow_pickle=True) for k, v in paths.items()}
                    return data
                except Exception as e:
                    print(f"Error loading cache: {e}. Recomputing...")
            else:
                print(
                    f"[{backbone_name}] Cache missing for {subset_name}. Computing..."
                )
        else:
            print(f"[{backbone_name}] Force recompute for {subset_name}...")

        # 2. Setup
        backbone_cfg = Config.BACKBONES[backbone_name]
        batch_size = backbone_cfg["batch_size"]

        # Initialize lists
        all_features = []
        all_ids = []
        all_meta = []
        all_targets = []

        # Load Model
        model = self._load_model(backbone_name)

        # Create Loader
        # We need flip augmentation enabled
        loader = get_loader(
            df=df,
            backbone_name=backbone_name,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            return_flip=Config.USE_FLIP_AUGMENTATION,
        )

        print(f"[{backbone_name}] Extracting features for {len(df)} images...")

        # 3. Inference Loop
        for batch in loader:
            # Metadata and IDs
            ids = batch["id"]
            meta = batch["meta_features"].numpy()

            # Targets (Handle missing targets for test set)
            if "target" in batch:
                targets = batch["target"].numpy()
            else:
                targets = np.full(len(ids), np.nan)

            # Feature Extraction (Original)
            emb_orig = self._compute_embedding(
                model, batch["pixel_values"], backbone_name
            )

            # Feature Extraction (Flipped) - Feature Space Augmentation
            if "pixel_values_flip" in batch:
                emb_flip = self._compute_embedding(
                    model, batch["pixel_values_flip"], backbone_name
                )
                # Average embeddings
                emb_final = (emb_orig + emb_flip) / 2.0
            else:
                emb_final = emb_orig

            # Move to CPU and store
            all_features.append(emb_final.cpu().numpy())
            all_ids.append(np.array(ids))
            all_meta.append(meta)
            all_targets.append(targets)

        # 4. Aggregate
        data = {
            "features": np.concatenate(all_features, axis=0),
            "ids": np.concatenate(all_ids, axis=0),
            "meta": np.concatenate(all_meta, axis=0),
            "targets": np.concatenate(all_targets, axis=0),
        }

        # 5. Save to Cache
        print(f"[{backbone_name}] Saving features to {Config.CACHE_DIR}...")
        np.save(paths["features"], data["features"])
        np.save(paths["ids"], data["ids"])
        np.save(paths["meta"], data["meta"])
        np.save(paths["targets"], data["targets"])

        # Cleanup model to free GPU memory
        del model
        torch.cuda.empty_cache()

        return data
