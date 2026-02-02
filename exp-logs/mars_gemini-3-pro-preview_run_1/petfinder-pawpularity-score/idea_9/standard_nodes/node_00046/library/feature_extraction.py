import os
import torch
import numpy as np
from transformers import CLIPModel, AutoModel, ConvNextModel
from library.config import Config
from library.data_loader import get_dataloader, load_metadata
from library.utils import seed_everything


class FeatureExtractor:
    """
    Handles feature extraction using multiple backbones with a Dual-View strategy
    (Global + Local) and Feature-Space Augmentation.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        seed_everything(Config.SEED)

    def _get_model(self, backbone_name):
        """
        Loads the specific HuggingFace model based on the backbone name.
        """
        model_path = Config.BACKBONES[backbone_name]
        print(f"Loading model: {backbone_name} ({model_path})...")

        if "clip" in backbone_name:
            model = CLIPModel.from_pretrained(model_path)
        elif "dinov2" in backbone_name:
            model = AutoModel.from_pretrained(model_path)
        elif "convnext" in backbone_name:
            model = ConvNextModel.from_pretrained(model_path)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        model.to(self.device)
        model.eval()
        return model

    def _forward_pass(self, model, images, backbone_name):
        """
        Performs inference with Feature-Space Augmentation (Original + Flip Average).
        """
        # 1. Original Forward Pass
        # 2. Flipped Forward Pass (Horizontal Flip on dim 3: B, C, H, W)
        images_flip = torch.flip(images, dims=[3])

        with torch.no_grad():
            # Helper to get embeddings based on architecture
            def get_emb(img_tensor):
                if "clip" in backbone_name:
                    # CLIP: project to multi-modal space
                    return model.get_image_features(pixel_values=img_tensor)
                elif "dinov2" in backbone_name:
                    # DINOv2: Use CLS token (index 0) from last hidden state
                    out = model(pixel_values=img_tensor)
                    return out.last_hidden_state[:, 0, :]
                elif "convnext" in backbone_name:
                    # ConvNeXt: Use pooler output (Global Average Pooling)
                    out = model(pixel_values=img_tensor)
                    return out.pooler_output
                else:
                    raise ValueError(f"Unknown backbone logic for {backbone_name}")

            emb_orig = get_emb(images)
            emb_flip = get_emb(images_flip)

        # Feature-Space Averaging
        return (emb_orig + emb_flip) / 2.0

    def extract_features(self, mode, backbone_name, load_cached_data=True):
        """
        Extracts features for a specific mode and backbone.
        Implements caching logic: checks if files exist before computing.

        Args:
            mode (str): Dataset mode ('train_all', 'test').
            backbone_name (str): Key from Config.BACKBONES.
            load_cached_data (bool): If True, attempts to load from disk first.
        """
        # Define cache file paths
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        # We generate files for Global and Local views
        file_map = {
            "global": os.path.join(
                cache_dir, f"{mode}_{backbone_name}_global_features.npy"
            ),
            "local": os.path.join(
                cache_dir, f"{mode}_{backbone_name}_local_features.npy"
            ),
            "ids": os.path.join(cache_dir, f"{mode}_ids.npy"),
            "meta": os.path.join(cache_dir, f"{mode}_meta.npy"),
            "targets": os.path.join(cache_dir, f"{mode}_targets.npy"),
        }

        # 1. Check Cache
        all_files_exist = all(os.path.exists(p) for p in file_map.values())

        if load_cached_data and all_files_exist:
            # Validate cache dimension
            try:
                # Use mmap_mode='r' to read shape without loading entire array
                # Validate against the feature file itself, not just the IDs file
                cached_feats = np.load(file_map["global"], mmap_mode="r")
                current_df = load_metadata(mode)
                if len(cached_feats) != len(current_df):
                    print(
                        f"Cache dimension mismatch for {backbone_name} (Cached: {len(cached_feats)}, Expected: {len(current_df)}). Recomputing..."
                    )
                else:
                    print(f"Loading cached features for {mode} - {backbone_name}...")
                    # We don't need to return the data here, just ensure it exists on disk.
                    # The downstream tasks will load these files.
                    return
            except Exception as e:
                print(
                    f"Error validating cache for {backbone_name}: {e}. Recomputing..."
                )

        # 2. Compute from Scratch
        print(f"Extracting features for {mode} - {backbone_name}...")

        # Load Model
        model = self._get_model(backbone_name)

        # Load Data
        dataloader = get_dataloader(
            mode=mode, backbone_name=backbone_name, shuffle=False
        )

        # Storage
        feats_global = []
        feats_local = []
        ids_list = []
        meta_list = []
        targets_list = []

        for batch in dataloader:
            # Move data to device
            global_imgs = batch["global_view"].to(self.device)
            local_imgs = batch["local_view"].to(self.device)
            meta = batch["meta"].numpy()
            targets = batch["target"].numpy()
            ids = batch["id"]  # Keep as list/tuple of strings

            # Inference
            f_global = self._forward_pass(model, global_imgs, backbone_name)
            f_local = self._forward_pass(model, local_imgs, backbone_name)

            # Store (move to CPU numpy)
            feats_global.append(f_global.cpu().numpy())
            feats_local.append(f_local.cpu().numpy())
            ids_list.extend(ids)
            meta_list.append(meta)
            targets_list.append(targets)

        # Concatenate
        feats_global = np.vstack(feats_global)
        feats_local = np.vstack(feats_local)
        meta_arr = np.vstack(meta_list)
        targets_arr = np.concatenate(targets_list)
        ids_arr = np.array(ids_list)

        # Save to Disk
        print(f"Saving features to {cache_dir}...")
        np.save(file_map["global"], feats_global)
        np.save(file_map["local"], feats_local)

        # Save metadata/targets (overwrite is fine, content is identical across backbones)
        np.save(file_map["ids"], ids_arr)
        np.save(file_map["meta"], meta_arr)
        np.save(file_map["targets"], targets_arr)

        # Cleanup
        del model
        torch.cuda.empty_cache()
        print(f"Completed {mode} - {backbone_name}.")

    def run(self, load_cached_data=True):
        """
        Orchestrates feature extraction for all backbones and required modes.
        """
        # We need features for full training (CV) and testing
        modes = ["train_all", "test"]

        for mode in modes:
            for backbone in Config.BACKBONES.keys():
                self.extract_features(mode, backbone, load_cached_data=load_cached_data)
