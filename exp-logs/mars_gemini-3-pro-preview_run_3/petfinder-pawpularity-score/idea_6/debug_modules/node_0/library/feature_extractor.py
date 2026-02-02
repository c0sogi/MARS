import os
import gc
import numpy as np
import pandas as pd
import torch
import timm
from transformers import CLIPVisionModel
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import generate_config_hash, load_cache, save_cache, set_seed
from library.data_loader import PawpularityDataset, get_transforms, load_metadata_splits


class FeatureEngine:
    """
    Orchestrates feature extraction from multiple backbones (Timm & Transformers).
    Handles caching, Test-Time Augmentation (TTA), and memory management.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.config_hash = generate_config_hash(Config)
        set_seed(Config.SEED)

    def _get_transform(self, backbone_type):
        """
        Returns the appropriate transform pipeline based on the backbone type.
        CLIP requires specific mean/std normalization.
        """
        if backbone_type == "clip":
            # OpenAI CLIP normalization constants
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)
        else:
            # Default ImageNet normalization
            mean = None  # defaults to (0.485, 0.456, 0.406) inside get_transforms
            std = None  # defaults to (0.229, 0.224, 0.225) inside get_transforms

        return get_transforms(Config.IMAGE_SIZE, mean=mean, std=std)

    def _load_model(self, backbone_cfg):
        """
        Loads the model architecture and weights.
        """
        name = backbone_cfg["name"]
        lib = backbone_cfg["library"]

        print(f"Loading model: {name} ({lib})")

        if lib == "timm":
            # Load Timm model (Swin, EffNet, DINOv2)
            # num_classes=0 returns the pooled feature vector
            model = timm.create_model(name, pretrained=True, num_classes=0)
        elif lib == "transformers":
            # Load Transformers model (CLIP)
            # We use the Vision Model to get image embeddings
            model = CLIPVisionModel.from_pretrained(name)
        else:
            raise ValueError(f"Unsupported library: {lib}")

        model.to(self.device)
        model.eval()
        return model

    def _run_inference(self, model, loader, backbone_type):
        """
        Runs inference on the dataloader. Handles TTA averaging.
        """
        embeddings = []

        with torch.no_grad():
            for batch_idx, (images, _, _, _) in enumerate(loader):
                images = images.to(self.device)

                # Handle Test-Time Augmentation
                if Config.USE_TTA:
                    # Input shape: (B, 2, C, H, W) -> Stack to (B*2, C, H, W)
                    b, n_crops, c, h, w = images.shape
                    images = images.view(-1, c, h, w)

                # Forward Pass
                if backbone_type == "clip":
                    # CLIPVisionModel expects 'pixel_values'
                    # Output has 'pooler_output' and 'last_hidden_state'
                    # pooler_output is the projected embedding (768/1024 dim)
                    output = model(pixel_values=images).pooler_output
                else:
                    # Timm models return the feature vector directly
                    output = model(images)

                # Handle TTA Averaging
                if Config.USE_TTA:
                    # Reshape back to (B, 2, Dim) and average
                    output = output.view(b, n_crops, -1).mean(dim=1)

                embeddings.append(output.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def extract_features(self, load_cached_data=True):
        """
        Main method to extract features for Train, Val, and Test sets.

        Args:
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            tuple: (features_dict, meta_dict, target_dict, ids_dict)
                - features_dict: Nested dict {split: {backbone_name: np.array}}
                - meta_dict: {split: np.array}
                - target_dict: {split: np.array}
                - ids_dict: {split: np.array}
        """
        # 1. Load Data Splits
        train_df, val_df, test_df = load_metadata_splits()
        dfs = {"train": train_df, "val": val_df, "test": test_df}

        # 2. Prepare Output Containers
        features_dict = {"train": {}, "val": {}, "test": {}}

        # 3. Iterate over Backbones (Sequential Processing to save VRAM)
        for backbone_cfg in Config.BACKBONES:
            backbone_name = backbone_cfg["name"]
            backbone_type = backbone_cfg["type"]

            # Determine if we need to run inference for any split
            splits_to_compute = []
            loaded_data = {}

            for split_name in dfs.keys():
                cache_filename = (
                    f"{split_name}_features_{backbone_name}_{self.config_hash}.npy"
                )
                cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

                data = None
                if load_cached_data:
                    data = load_cache(cache_path)

                if data is not None:
                    print(f"Loaded cached features for {backbone_name} ({split_name})")
                    loaded_data[split_name] = data
                else:
                    splits_to_compute.append(split_name)

            # If computation is needed
            if splits_to_compute:
                model = self._load_model(backbone_cfg)
                transform = self._get_transform(backbone_type)

                for split_name in splits_to_compute:
                    print(f"Extracting features for {backbone_name} ({split_name})...")
                    df = dfs[split_name]

                    dataset = PawpularityDataset(
                        df,
                        Config.INPUT_DIR,
                        transform=transform,
                        use_tta=Config.USE_TTA,
                    )

                    loader = DataLoader(
                        dataset,
                        batch_size=Config.BATCH_SIZE,
                        shuffle=False,
                        num_workers=Config.NUM_WORKERS,
                        pin_memory=True,
                    )

                    # Run Inference
                    embeddings = self._run_inference(model, loader, backbone_type)

                    # Save to Cache
                    cache_filename = (
                        f"{split_name}_features_{backbone_name}_{self.config_hash}.npy"
                    )
                    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)
                    save_cache(embeddings, cache_path)

                    loaded_data[split_name] = embeddings

                # Cleanup Model
                del model
                torch.cuda.empty_cache()
                gc.collect()

            # Store in result dictionary
            for split_name in dfs.keys():
                features_dict[split_name][backbone_name] = loaded_data[split_name]

        # 4. Extract Auxiliary Data (Meta, Targets, IDs)
        # We extract this directly from DataFrames to be fast and robust
        meta_dict = {}
        target_dict = {}
        ids_dict = {}

        print("Extracting metadata and targets from DataFrames...")
        for split_name, df in dfs.items():
            # Metadata Features (Binary)
            meta_dict[split_name] = df[Config.METADATA_COLS].values.astype(np.float32)

            # Targets (Pawpularity) - Test set might not have it, handled in Dataset but here we check col
            if "Pawpularity" in df.columns:
                target_dict[split_name] = df["Pawpularity"].values.astype(np.float32)
            else:
                # Dummy targets for test if column missing (though test_meta usually doesn't have it)
                target_dict[split_name] = np.zeros(len(df), dtype=np.float32)

            # IDs
            ids_dict[split_name] = df["Id"].values

        return features_dict, meta_dict, target_dict, ids_dict
