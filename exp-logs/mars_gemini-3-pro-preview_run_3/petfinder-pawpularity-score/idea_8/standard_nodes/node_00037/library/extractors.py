import os
import gc
import hashlib
import numpy as np
import torch
import torch.nn as nn
import timm
from transformers import AutoModel, CLIPVisionModel, SwinModel
from library.config import Config
from library.utils import get_device, save_cache, load_cache


class FeatureExtractor:
    """
    Handles loading of pre-trained backbones, extracting features with Dual-Pooling,
    applying TTA aggregation, and caching results to disk.
    """

    def __init__(self):
        self.device = get_device()
        self.backbones = Config.BACKBONES
        self.working_dir = Config.WORKING_DIR

    def _get_cache_filename(self, backbone_name: str, split: str) -> str:
        """
        Generates a deterministic filename for caching based on backbone name and split.
        """
        # Create a safe filename from the backbone name using MD5 hash
        h = hashlib.md5(backbone_name.encode()).hexdigest()
        return f"{split}_features_{h}.npy"

    def _load_model(self, backbone_cfg: dict):
        """
        Loads a specific model based on configuration (timm or transformers).
        """
        name = backbone_cfg["name"]
        source = backbone_cfg["source"]
        model_type = backbone_cfg["type"]

        print(f"Loading model: {name} ({source})")

        if source == "timm":
            # Create model with no classifier and no global pooling to get raw feature maps
            # num_classes=0 removes the head, global_pool='' keeps spatial dims
            model = timm.create_model(
                name, pretrained=True, num_classes=0, global_pool=""
            )
        elif source == "transformers":
            if model_type == "swin":
                model = SwinModel.from_pretrained(name)
            elif model_type == "clip":
                model = CLIPVisionModel.from_pretrained(name)
            elif model_type == "vit":
                # DINOv2 and generic ViTs
                model = AutoModel.from_pretrained(name)
            else:
                raise ValueError(f"Unknown transformer type: {model_type}")
        else:
            raise ValueError(f"Unknown source: {source}")

        model.to(self.device)
        model.eval()
        return model

    def _dual_pool(self, features: torch.Tensor, model_type: str) -> torch.Tensor:
        """
        Applies Dual-Statistic Pooling (Avg + Max).

        Args:
            features: Tensor from model.
                      CNN: (B, C, H, W)
                      ViT/CLIP/Swin: (B, L, C)
            model_type: 'cnn', 'swin', 'vit', 'clip'

        Returns:
            pooled: (B, 2*C)
        """
        if model_type == "cnn":
            # features: (B, C, H, W)
            # Avg Pool over spatial dims
            avg_pool = torch.mean(features, dim=(2, 3))
            # Max Pool over spatial dims
            max_pool = torch.amax(features, dim=(2, 3))
            return torch.cat([avg_pool, max_pool], dim=1)

        elif model_type in ["swin", "vit", "clip"]:
            # features: (B, L, C)

            # Determine sequence slice based on type
            if model_type == "swin":
                # Swin transformers (HF) output last_hidden_state as (B, H*W, C).
                # There is no separate CLS token in the sequence for SwinModel.
                x = features
            else:
                # ViT (DINO) and CLIP have CLS token at index 0.
                # We want to pool over spatial tokens only, excluding CLS.
                x = features[:, 1:, :]

            # Avg Pool over sequence dimension (dim 1)
            avg_pool = torch.mean(x, dim=1)
            # Max Pool over sequence dimension
            max_pool = torch.amax(x, dim=1)
            return torch.cat([avg_pool, max_pool], dim=1)

        else:
            raise ValueError(f"Unknown model type for pooling: {model_type}")

    def _process_loader(self, loader, model, model_type: str) -> np.ndarray:
        """
        Iterates through the loader, performs inference, dual pooling, and TTA averaging.
        """
        all_features = []

        # Disable gradients for inference
        with torch.no_grad():
            for i, batch in enumerate(loader):
                # Batch['image'] is (B, 2, 3, H, W) due to TTA (stack of orig + flip)
                # Or (B, 3, H, W) if TTA is disabled
                images = batch["image"].to(self.device)

                if images.dim() == 5:
                    b, t, c, h, w = images.shape
                    # Flatten TTA dimension into batch dimension -> (B*T, C, H, W)
                    inputs = images.view(b * t, c, h, w)
                else:
                    b, c, h, w = images.shape
                    t = 1
                    inputs = images

                # Forward pass
                if hasattr(model, "forward_features"):
                    # timm models
                    raw_feats = model.forward_features(inputs)
                elif isinstance(model, CLIPVisionModel):
                    # CLIP expects pixel_values
                    outputs = model(pixel_values=inputs)
                    raw_feats = outputs.last_hidden_state
                else:
                    # Transformers (Swin, ViT)
                    outputs = model(inputs)
                    raw_feats = outputs.last_hidden_state

                # Apply Dual Pooling -> Output shape (B*2, 2*C)
                pooled = self._dual_pool(raw_feats, model_type)

                # Reshape to separate TTA: (B, 2, 2*C)
                pooled = pooled.view(b, t, -1)

                # Average over TTA dimension: (B, 2*C)
                final_feats = torch.mean(pooled, dim=1)

                all_features.append(final_feats.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def extract_all(
        self, train_loader, val_loader, test_loader, load_cached_data: bool = True
    ):
        """
        Extracts features for all backbones and all splits.

        Args:
            train_loader, val_loader, test_loader: DataLoaders for respective splits.
            load_cached_data: If True, attempts to load from disk first.

        Returns:
            results: Dictionary {backbone_name: {'train': np.array, 'val': ..., 'test': ...}}
        """
        results = {}

        for backbone_cfg in self.backbones:
            name = backbone_cfg["name"]
            model_type = backbone_cfg["type"]

            results[name] = {}

            # Define cache filenames
            train_cache_name = self._get_cache_filename(name, "train")
            val_cache_name = self._get_cache_filename(name, "val")
            test_cache_name = self._get_cache_filename(name, "test")

            # Check if all exist
            train_data = load_cache(train_cache_name) if load_cached_data else None
            val_data = load_cache(val_cache_name) if load_cached_data else None
            test_data = load_cache(test_cache_name) if load_cached_data else None

            if (
                train_data is not None
                and val_data is not None
                and test_data is not None
            ):
                print(f"Loaded cached features for {name}")
                results[name]["train"] = train_data
                results[name]["val"] = val_data
                results[name]["test"] = test_data
                continue

            # If any missing, recompute all for this backbone
            print(f"Extracting features for {name}...")
            model = self._load_model(backbone_cfg)

            # Process Train
            print(f"Processing Train set for {name}...")
            train_data = self._process_loader(train_loader, model, model_type)
            save_cache(train_data, train_cache_name)
            results[name]["train"] = train_data

            # Process Val
            print(f"Processing Val set for {name}...")
            val_data = self._process_loader(val_loader, model, model_type)
            save_cache(val_data, val_cache_name)
            results[name]["val"] = val_data

            # Process Test
            print(f"Processing Test set for {name}...")
            test_data = self._process_loader(test_loader, model, model_type)
            save_cache(test_data, test_cache_name)
            results[name]["test"] = test_data

            # Cleanup to save memory for next backbone
            del model
            gc.collect()
            torch.cuda.empty_cache()

        return results
