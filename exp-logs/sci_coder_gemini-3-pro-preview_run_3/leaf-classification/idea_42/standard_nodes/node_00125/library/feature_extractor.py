import os
import numpy as np
import pandas as pd
import torch
import timm
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from library.utils import (
    load_metadata,
    save_to_cache,
    load_from_cache,
    get_device,
    INPUT_DIR,
)


class FeatureExtractor:
    """
    Handles feature extraction using DINOv2 and ConvNeXt Large with
    multi-view rotation augmentation.
    """

    def __init__(self, device=None):
        self.device = device if device else get_device()
        self.cache_subdir = "idea_42"

        # Initialize DINOv2 (ViT-Large)
        # Using a standard timm model name for DINOv2
        self.dino_model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,  # Feature extraction mode
            img_size=224,
        ).to(self.device)
        self.dino_model.eval()

        # Initialize ConvNeXt Large
        self.conv_model = timm.create_model(
            "convnext_large.fb_in22k_ft_in1k", pretrained=True, num_classes=0
        ).to(self.device)
        self.conv_model.eval()

        # Standard ImageNet normalization
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def _load_image(self, rel_path):
        """Loads an image from disk and converts to PIL RGB."""
        full_path = os.path.join(INPUT_DIR, rel_path)
        # Use cv2 for robust loading
        img_arr = cv2.imread(full_path)
        if img_arr is None:
            raise FileNotFoundError(f"Could not load image at {full_path}")
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_arr)

    def _generate_rotated_views(self, pil_img):
        """
        Generates 12 equidistant rotated views (0, 30, ..., 330).
        Returns a tensor of shape (12, 3, 224, 224).
        """
        views = []
        # 0 to 330 degrees with step 30
        for angle in range(0, 360, 30):
            # Fill with white (255) to match background
            rotated = F.rotate(pil_img, angle, fill=[255, 255, 255])
            tensor_view = self.transform(rotated)
            views.append(tensor_view)

        return torch.stack(views)

    def extract_and_save_features(
        self, split: str, load_cached_data: bool = True, limit: int = None
    ):
        """
        Extracts features for the given split (train/val/test).

        Args:
            split: 'train', 'val', or 'test'.
            load_cached_data: If True, attempts to load from disk first.
            limit: Optional integer to limit the number of samples (for debugging).

        Returns:
            dino_feats: (N, 12, 1024)
            conv_feats: (N, 12, 1536)
            tab_feats:  (N, 192)
            ids:        (N,)
        """
        # Define cache filenames
        dino_file = f"{split}_dino_features.npy"
        conv_file = f"{split}_conv_features.npy"
        tab_file = f"{split}_tab_features.npy"
        ids_file = f"{split}_ids.npy"

        # Load metadata to validate cache length
        df_full = load_metadata(split)
        expected_len = len(df_full)

        # 1. Try loading from cache
        if load_cached_data:
            dino_feats = load_from_cache(dino_file, sub_dir=self.cache_subdir)
            conv_feats = load_from_cache(conv_file, sub_dir=self.cache_subdir)
            tab_feats = load_from_cache(tab_file, sub_dir=self.cache_subdir)
            ids = load_from_cache(ids_file, sub_dir=self.cache_subdir)

            if (
                dino_feats is not None
                and conv_feats is not None
                and tab_feats is not None
                and ids is not None
            ):
                # Validate cache length against full metadata (Cite debug_lesson_3)
                if len(dino_feats) == expected_len:
                    print(
                        f"[{split}] Loaded features from cache ({self.cache_subdir})."
                    )
                    # If limit is applied, slice the cached data
                    if limit:
                        return (
                            dino_feats[:limit],
                            conv_feats[:limit],
                            tab_feats[:limit],
                            ids[:limit],
                        )
                    return dino_feats, conv_feats, tab_feats, ids
                else:
                    print(
                        f"[{split}] Cache mismatch (found {len(dino_feats)}, expected {expected_len}). "
                        "Ignoring cache and re-extracting."
                    )

        # 2. Extract from scratch
        print(f"[{split}] Extracting features from scratch...")

        # Use loaded metadata
        df = df_full
        if limit:
            df = df.head(limit)

        # Prepare tabular data
        # Columns: margin_1..64, shape_1..64, texture_1..64
        margin_cols = [c for c in df.columns if c.startswith("margin")]
        shape_cols = [c for c in df.columns if c.startswith("shape")]
        texture_cols = [c for c in df.columns if c.startswith("texture")]
        feat_cols = margin_cols + shape_cols + texture_cols

        tab_feats_arr = df[feat_cols].values.astype(np.float32)
        ids_arr = df["id"].values.astype(np.int32)
        paths = df["file_path"].tolist()

        # Lists to store deep features
        dino_list = []
        conv_list = []

        # Batch processing
        batch_size = 16  # Effective batch size = 16 * 12 = 192 images on GPU

        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i : i + batch_size]

                # Prepare batch of 12-view sets
                batch_tensors = []
                for p in batch_paths:
                    pil_img = self._load_image(p)
                    views = self._generate_rotated_views(pil_img)  # (12, 3, 224, 224)
                    batch_tensors.append(views)

                # Stack: (B*12, 3, 224, 224)
                full_batch = torch.cat(batch_tensors, dim=0).to(self.device)

                # Forward Pass
                dino_out = self.dino_model(full_batch)  # (B*12, 1024)
                conv_out = self.conv_model(full_batch)  # (B*12, 1536)

                # Reshape to (B, 12, D) and move to CPU
                B = len(batch_paths)
                dino_reshaped = dino_out.view(B, 12, -1).cpu().numpy()
                conv_reshaped = conv_out.view(B, 12, -1).cpu().numpy()

                dino_list.append(dino_reshaped)
                conv_list.append(conv_reshaped)

                if (i // batch_size) % 10 == 0:
                    print(
                        f"Processed {min(i + batch_size, len(paths))}/{len(paths)} images"
                    )

        # Concatenate
        dino_feats_arr = np.concatenate(dino_list, axis=0)
        conv_feats_arr = np.concatenate(conv_list, axis=0)

        # 3. Save to cache
        print(f"[{split}] Saving features to cache...")
        save_to_cache(dino_feats_arr, dino_file, sub_dir=self.cache_subdir)
        save_to_cache(conv_feats_arr, conv_file, sub_dir=self.cache_subdir)
        save_to_cache(tab_feats_arr, tab_file, sub_dir=self.cache_subdir)
        save_to_cache(ids_arr, ids_file, sub_dir=self.cache_subdir)

        return dino_feats_arr, conv_feats_arr, tab_feats_arr, ids_arr
