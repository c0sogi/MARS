import os
import gc
import numpy as np
import torch
import torch.nn as nn
import timm
import transformers
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import PetDataset, get_transforms
from library.utils import seed_everything, get_logger

# Suppress verbose logging
transformers.logging.set_verbosity_error()


def spatial_pyramid_pooling(feature_map):
    """
    Applies Spatial Pyramid Pooling (SPP) with levels 1x1 (GAP) and 2x2.

    Args:
        feature_map (torch.Tensor): Input tensor of shape (B, C, H, W)

    Returns:
        torch.Tensor: Flattened feature vector of shape (B, C * 5)
    """
    B, C, H, W = feature_map.shape

    # Level 0: Global Average Pooling (1x1) -> (B, C)
    # Equivalent to adaptive_avg_pool2d(x, (1, 1))
    l0 = torch.mean(feature_map, dim=(2, 3))

    # Level 1: 2x2 Quadrants -> (B, C, 2, 2)
    l1 = nn.functional.adaptive_avg_pool2d(feature_map, (2, 2))

    # Flatten 2x2 grid: (B, C, 4) -> (B, C*4)
    # Grid order: (0,0)=TL, (0,1)=TR, (1,0)=BL, (1,1)=BR
    l1_flat = l1.reshape(B, C, 4).reshape(B, -1)

    # Concatenate: (B, C + 4C) = (B, 5C)
    return torch.cat([l0, l1_flat], dim=1)


class BackboneWrapper:
    def __init__(self, model_name, device, img_size=224):
        self.name = model_name
        self.device = device
        self.model_type = "timm"

        if "clip" in model_name.lower():
            self.model_type = "clip"
            self.model = transformers.CLIPVisionModel.from_pretrained(model_name).to(
                device
            )
        else:
            # Create timm model without classifier and without global pooling (keep spatial)
            # global_pool='' ensures we get feature maps
            kwargs = {"pretrained": True, "num_classes": 0, "global_pool": ""}

            # Cite debug_lesson_7: Conditionally pass img_size for Transformers (ViT/Swin) to enable
            # resolution interpolation, while avoiding TypeError for CNNs (EfficientNet) which don't support it.
            if "vit" in model_name or "swin" in model_name:
                kwargs["img_size"] = img_size

            self.model = timm.create_model(model_name, **kwargs).to(device)

        self.model.eval()

    def get_features(self, x):
        """
        Extracts spatial feature maps and normalizes shape to (B, C, H, W).
        """
        with torch.no_grad(), torch.cuda.amp.autocast():
            if self.model_type == "clip":
                # CLIP output: (B, L, C)
                # L = 1 (CLS) + H*W (Patches)
                outputs = self.model(pixel_values=x)
                out = outputs.last_hidden_state

                # Remove CLS token
                feat_map = out[:, 1:, :]  # (B, 256, 1024) for 224x224 patch14

                # Reshape to (B, C, H, W)
                B, L, C = feat_map.shape
                H = W = int(L**0.5)
                feat_map = feat_map.permute(0, 2, 1).reshape(B, C, H, W)

            else:
                # Timm models
                out = self.model(x)

                # Handle various output shapes from timm
                if out.ndim == 4:
                    # Case: (B, C, H, W) - e.g., EfficientNet
                    feat_map = out

                    # Some models (like Swin in some configs) might output (B, H, W, C)
                    # Heuristic: Check if last dim is channel-like (large) and 2nd dim is spatial (small)
                    if (
                        feat_map.shape[1] < feat_map.shape[3]
                        and feat_map.shape[3] > 100
                    ):
                        feat_map = feat_map.permute(0, 3, 1, 2)

                elif out.ndim == 3:
                    # Case: (B, L, C) - e.g., ViT, Swin (sometimes)
                    # Check for CLS token based on sequence length
                    B, L, C = out.shape
                    H = W = int(L**0.5)

                    if H * W != L:
                        # Likely has CLS token (L = HW + 1) or Register tokens
                        # Assuming 1 CLS token at index 0 for standard ViTs
                        if (H * W + 1) == L:
                            out = out[:, 1:, :]
                        # Recalculate H, W just in case
                        L_new = out.shape[1]
                        H = W = int(L_new**0.5)

                    feat_map = out.permute(0, 2, 1).reshape(B, C, H, W)
                else:
                    raise ValueError(
                        f"Unexpected output shape {out.shape} from model {self.name}"
                    )

        return feat_map


class FeatureEngine:
    def __init__(self):
        self.config = Config
        self.device = Config.DEVICE
        self.logger = get_logger("FeatureEngine")

    def _extract_dataset(self, backbone, dataset, batch_size):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        features_list = []
        ids_list = []

        for imgs, _, _, img_ids in loader:
            imgs = imgs.to(self.device)

            # 1. Original Images
            feats_orig_map = backbone.get_features(imgs)  # (B, C, H, W)
            spp_orig = spatial_pyramid_pooling(feats_orig_map)  # (B, 5C)

            # 2. Flipped Images (TTA)
            imgs_flip = torch.flip(imgs, [3])  # Horizontal flip
            feats_flip_map = backbone.get_features(imgs_flip)
            spp_flip = spatial_pyramid_pooling(feats_flip_map)

            # 3. Spatial Un-flipping / Alignment
            # SPP structure: [GAP (C), TL (C), TR (C), BL (C), BR (C)]
            # Indices:       0:C      C:2C    2C:3C   3C:4C   4C:5C

            B, dim = spp_orig.shape
            C = dim // 5

            # Split GAP and Quadrants
            gap_flip = spp_flip[:, :C]
            quads_flip = spp_flip[:, C:]  # (B, 4C)

            # Reshape quadrants to (B, C, 4) for permutation
            # Current order of quads_flip: [TL', TR', BL', BR']
            # Where TL' of flipped image corresponds to TR of original scene
            # We want: [TL_scene, TR_scene, BL_scene, BR_scene]
            # Mapping:
            # TL_scene = TR' (Index 1)
            # TR_scene = TL' (Index 0)
            # BL_scene = BR' (Index 3)
            # BR_scene = BL' (Index 2)

            quads_flip = quads_flip.reshape(B, C, 4)
            # Permute last dim: [0, 1, 2, 3] -> [1, 0, 3, 2]
            quads_flip_aligned = quads_flip[:, :, [1, 0, 3, 2]]
            quads_flip_aligned = quads_flip_aligned.reshape(B, -1)

            # Reassemble aligned flipped features
            spp_flip_aligned = torch.cat([gap_flip, quads_flip_aligned], dim=1)

            # 4. Average
            final_feats = (spp_orig + spp_flip_aligned) / 2.0

            features_list.append(final_feats.cpu().numpy())
            ids_list.extend(img_ids)

        return np.concatenate(features_list, axis=0), np.array(ids_list)

    def run(self, load_cached_data=True):
        self.logger.info("Starting Feature Extraction Pipeline...")

        # Ensure working directory exists
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

        # Define Datasets
        # We use the same transform for all (Resize + Norm)
        transforms = get_transforms(self.config.IMG_SIZE)

        datasets = {
            "train": PetDataset(self.config.TRAIN_META_PATH, transform=transforms),
            "val": PetDataset(self.config.VAL_META_PATH, transform=transforms),
            "test": PetDataset(self.config.TEST_META_PATH, transform=transforms),
        }

        # Iterate over Backbones
        for friendly_name, model_name in self.config.BACKBONES.items():
            self.logger.info(f"Processing Backbone: {friendly_name} ({model_name})")

            # Check if all files exist for this backbone
            all_exist = True
            for split in datasets.keys():
                feat_path = os.path.join(
                    self.config.WORKING_DIR, f"{friendly_name}_{split}_features.npy"
                )
                id_path = os.path.join(
                    self.config.WORKING_DIR, f"{friendly_name}_{split}_ids.npy"
                )
                if not (os.path.exists(feat_path) and os.path.exists(id_path)):
                    all_exist = False
                    break

            if load_cached_data and all_exist:
                self.logger.info(
                    f"Found cached features for {friendly_name}. Skipping extraction."
                )
                continue

            # Initialize Model
            try:
                backbone = BackboneWrapper(
                    model_name, self.device, img_size=self.config.IMG_SIZE
                )
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                continue

            # Extract for each split
            for split, dataset in datasets.items():
                self.logger.info(f"Extracting {split} set features...")

                feat_path = os.path.join(
                    self.config.WORKING_DIR, f"{friendly_name}_{split}_features.npy"
                )
                id_path = os.path.join(
                    self.config.WORKING_DIR, f"{friendly_name}_{split}_ids.npy"
                )

                # Check individual split cache if partial run
                if (
                    load_cached_data
                    and os.path.exists(feat_path)
                    and os.path.exists(id_path)
                ):
                    self.logger.info(f"Split {split} already cached.")
                    continue

                features, ids = self._extract_dataset(
                    backbone, dataset, self.config.BATCH_SIZE
                )

                # Save
                np.save(feat_path, features)
                np.save(id_path, ids)
                self.logger.info(f"Saved {split} features: {features.shape}")

                # Clean up memory
                del features, ids
                gc.collect()

            # Clean up model
            del backbone
            torch.cuda.empty_cache()
            gc.collect()

        self.logger.info("Feature Extraction Completed.")
