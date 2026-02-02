import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModel
from library.config import Config
from library.utils import save_to_cache, load_from_cache, seed_everything


class FeatureExtractor:
    """
    Handles feature extraction using HuggingFace backbones.
    Supports SigLIP, DINOv2, and ConvNeXt V2 architectures.
    """

    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load model with trust_remote_code for safety with newer architectures
        try:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def _get_embeddings(self, pixel_values):
        """
        Extracts embeddings from the model, handling different architecture interfaces.
        """
        with torch.no_grad():
            # 1. SigLIP / CLIP Interface
            if hasattr(self.model, "get_image_features"):
                return self.model.get_image_features(pixel_values)

            # 2. General Backbone Interface
            outputs = self.model(pixel_values)

            # 3. Use pooler_output if available (ConvNeXt V2, some ViTs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output

            # 4. Fallback to CLS token (ViT / DINOv2)
            # DINOv2 output is (Batch, Seq, Dim), CLS is at index 0
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state[:, 0]

            raise ValueError(
                f"Could not determine embedding output for model {self.model_name}"
            )

    def extract(self, dataset, batch_size, num_workers=Config.NUM_WORKERS):
        """
        Iterates over the dataset and extracts features.
        Handles Feature-Space Augmentation Averaging if the dataset returns multiple views.

        Args:
            dataset: PyTorch Dataset returning dict with 'pixel_values', 'id', 'metadata'.
            batch_size: Batch size for inference.
            num_workers: Number of dataloader workers.

        Returns:
            dict: Dictionary with 'ids', 'embeddings', 'metadata', 'targets'.
        """
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        all_ids = []
        all_embeddings = []
        all_targets = []
        all_metadata = []

        # Ensure deterministic behavior
        seed_everything()

        with torch.no_grad():
            for batch in dataloader:
                # Move pixel values to device
                # Shape: (B, C, H, W) or (B, N_AUG, C, H, W)
                pixel_values = batch["pixel_values"].to(self.device)
                ids = batch["id"]
                metadata = batch["metadata"].numpy()

                # Handle targets if present (Train/Val sets)
                if "target" in batch:
                    targets = batch["target"].numpy()
                    all_targets.append(targets)

                # Feature-Space Augmentation Logic
                if pixel_values.ndim == 5:
                    b, n_aug, c, h, w = pixel_values.shape
                    # Flatten: (B * N_AUG, C, H, W)
                    pixel_values_flat = pixel_values.view(-1, c, h, w)

                    # Extract
                    embeddings_flat = self._get_embeddings(
                        pixel_values_flat
                    )  # (B * N_AUG, D)

                    # Reshape and Average: (B, N_AUG, D) -> (B, D)
                    d = embeddings_flat.shape[1]
                    embeddings = embeddings_flat.view(b, n_aug, d).mean(dim=1)
                else:
                    # Standard extraction
                    embeddings = self._get_embeddings(pixel_values)

                all_embeddings.append(embeddings.cpu().numpy())
                all_ids.extend(ids)
                all_metadata.append(metadata)

        # Aggregate results
        result = {
            "ids": np.array(all_ids),
            "embeddings": np.vstack(all_embeddings),
            "metadata": np.vstack(all_metadata),
        }

        if all_targets:
            result["targets"] = np.concatenate(all_targets)
        else:
            result["targets"] = None

        return result


def process_and_cache_features(
    dataset, model_name, batch_size, cache_paths, load_cached_data=True
):
    """
    Orchestrates feature extraction with caching mechanism.

    Args:
        dataset (Dataset): The PyTorch dataset.
        model_name (str): HuggingFace model identifier.
        batch_size (int): Batch size for inference.
        cache_paths (dict): Dictionary with keys 'ids', 'embeddings', 'metadata', 'targets' mapping to file paths.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing 'ids', 'embeddings', 'metadata', 'targets'.
    """
    # 1. Attempt to load from cache
    if load_cached_data:
        # Check required keys
        p_emb = cache_paths.get("embeddings")
        p_ids = cache_paths.get("ids")
        p_meta = cache_paths.get("metadata")
        p_tgt = cache_paths.get("targets")

        # Verify existence of core files (embeddings, ids, metadata)
        if (
            p_emb
            and os.path.exists(p_emb)
            and p_ids
            and os.path.exists(p_ids)
            and p_meta
            and os.path.exists(p_meta)
        ):

            # If targets path is provided, it must exist unless targets are None (e.g. test set)
            # We'll try to load it if the path is provided and exists.
            targets = None
            if p_tgt and os.path.exists(p_tgt):
                targets = load_from_cache(p_tgt)

            print(f"Loading cached features for {model_name}...")
            return {
                "ids": load_from_cache(p_ids),
                "embeddings": load_from_cache(p_emb),
                "metadata": load_from_cache(p_meta),
                "targets": targets,
            }

    # 2. Run Extraction
    print(f"Extracting features for {model_name}...")
    extractor = FeatureExtractor(model_name)
    data = extractor.extract(dataset, batch_size)

    # 3. Save to Cache
    if cache_paths.get("embeddings"):
        save_to_cache(data["embeddings"], cache_paths["embeddings"])
    if cache_paths.get("ids"):
        save_to_cache(data["ids"], cache_paths["ids"])
    if cache_paths.get("metadata"):
        save_to_cache(data["metadata"], cache_paths["metadata"])
    if cache_paths.get("targets") and data["targets"] is not None:
        save_to_cache(data["targets"], cache_paths["targets"])

    return data
