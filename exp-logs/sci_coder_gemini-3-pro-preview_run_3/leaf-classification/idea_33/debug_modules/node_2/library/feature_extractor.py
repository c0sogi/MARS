import os
import cv2
import numpy as np
import pandas as pd
import torch
import timm
from library.config import Config
from library.utils import get_device


class FeatureExtractor:
    """
    Extracts features from images using DINOv2 and ConvNeXt models.
    Implements Manifold Densification by generating 12 rotated views per image.
    """

    def __init__(self):
        self.device = get_device()
        self.img_size = Config.IMG_SIZE
        self.mean = np.array(Config.IMG_MEAN, dtype=np.float32)
        self.std = np.array(Config.IMG_STD, dtype=np.float32)

        # Prepare model names (handle HF Hub prefix if necessary for timm)
        self.dino_name = self._format_model_name(Config.MODEL_DINO)
        self.conv_name = self._format_model_name(Config.MODEL_CONVNEXT)

        print(f"Initializing FeatureExtractor on {self.device}")

        print(f"Loading DINOv2 model: {self.dino_name}")
        self.dino_model = (
            timm.create_model(
                self.dino_name,
                pretrained=True,
                num_classes=0,
                img_size=self.img_size,
            )
            .to(self.device)
            .eval()
        )

        print(f"Loading ConvNeXt model: {self.conv_name}")
        self.conv_model = (
            timm.create_model(self.conv_name, pretrained=True, num_classes=0)
            .to(self.device)
            .eval()
        )

    def _format_model_name(self, name):
        """Ensures model name has hf_hub prefix if it looks like a HF repo ID."""
        if "/" in name and not name.startswith("hf_hub:"):
            return f"hf_hub:{name}"
        return name

    def _get_rotations(self, img):
        """
        Generates 12 equidistant rotations of the input image.
        Args:
            img (np.ndarray): Input image (H, W, 3) in RGB.
        Returns:
            np.ndarray: Batch of rotated images (12, 3, H, W) normalized.
        """
        h, w = img.shape[:2]
        # Pad to square to prevent cropping during rotation
        dim = max(h, w)
        pad_h = (dim - h) // 2
        pad_w = (dim - w) // 2

        # White padding (255) for binary leaf images on white background
        padded = cv2.copyMakeBorder(
            img,
            pad_h,
            dim - h - pad_h,
            pad_w,
            dim - w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        center = (dim // 2, dim // 2)
        rotated_batch = []

        for angle in Config.ROTATION_ANGLES:
            # Rotate
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                padded,
                M,
                (dim, dim),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

            # Resize to model input size
            resized = cv2.resize(
                rotated, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC
            )

            # Normalize
            resized = resized.astype(np.float32) / 255.0
            resized = (resized - self.mean) / self.std

            # HWC -> CHW
            resized = resized.transpose(2, 0, 1)
            rotated_batch.append(resized)

        return np.stack(rotated_batch)

    def _process_dataset(self, df, split_name, load_cached_data):
        """
        Process a specific dataset split (train/val/test), handling caching.
        """
        # Define cache paths
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        dino_path = os.path.join(cache_dir, f"{split_name}_dino.npy")
        conv_path = os.path.join(cache_dir, f"{split_name}_conv.npy")
        ids_path = os.path.join(cache_dir, f"{split_name}_ids.npy")

        # Check cache
        if load_cached_data:
            if (
                os.path.exists(dino_path)
                and os.path.exists(conv_path)
                and os.path.exists(ids_path)
            ):
                print(f"[{split_name}] Loading cached features from {cache_dir}")
                return (np.load(dino_path), np.load(conv_path), np.load(ids_path))
            else:
                print(f"[{split_name}] Cache not found. Extracting features...")
        else:
            print(f"[{split_name}] Forcing feature extraction (ignoring cache)...")

        # Extract features
        dino_feats, conv_feats, ids = self._extract_features(df, split_name)

        # Save to cache
        print(f"[{split_name}] Saving features to {cache_dir}")
        np.save(dino_path, dino_feats)
        np.save(conv_path, conv_feats)
        np.save(ids_path, ids)

        return dino_feats, conv_feats, ids

    def _extract_features(self, df, split_name):
        """
        Iterates over the dataframe, loads images, generates rotations, and runs inference.
        """
        img_paths = df["file_path"].tolist()
        ids = df["id"].tolist()

        # Apply Debug limit if configured
        if Config.DEBUG:
            print(
                f"[{split_name}] Debug mode: Limiting to {Config.DEBUG_LIMIT} samples."
            )
            img_paths = img_paths[: Config.DEBUG_LIMIT]
            ids = ids[: Config.DEBUG_LIMIT]

        num_samples = len(img_paths)
        # Batch size of 8 images results in 8 * 12 = 96 inputs to the model.
        # This is safe for A100 memory with ViT-Large.
        batch_size = 8

        all_dino = []
        all_conv = []
        processed_ids = []

        input_root = Config.INPUT_DIR

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_paths = img_paths[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]

                batch_tensors = []
                valid_batch_ids = []

                for p, img_id in zip(batch_paths, batch_ids):
                    full_path = os.path.join(input_root, p)
                    if not os.path.exists(full_path):
                        print(f"Warning: Image not found at {full_path}. Skipping.")
                        continue

                    # Load image
                    img = cv2.imread(full_path)
                    if img is None:
                        print(f"Warning: Failed to load image {full_path}. Skipping.")
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Get 12 rotations: (12, 3, 224, 224)
                    rotations = self._get_rotations(img)
                    batch_tensors.append(rotations)
                    valid_batch_ids.append(img_id)

                if not batch_tensors:
                    continue

                # Stack images: (B, 12, 3, H, W)
                batch_stack = np.stack(batch_tensors)
                B, V, C, H, W = batch_stack.shape

                # Flatten for model inference: (B*12, 3, H, W)
                inp = torch.from_numpy(batch_stack).view(B * V, C, H, W).to(self.device)

                # Inference DINO
                dino_out = self.dino_model(inp)  # Output: (B*V, Embed_Dim)

                # Inference ConvNeXt
                conv_out = self.conv_model(inp)  # Output: (B*V, Embed_Dim)

                # Reshape back to (B, 12, Embed_Dim)
                dino_reshaped = dino_out.view(B, V, -1).cpu().numpy()
                conv_reshaped = conv_out.view(B, V, -1).cpu().numpy()

                all_dino.append(dino_reshaped)
                all_conv.append(conv_reshaped)
                processed_ids.extend(valid_batch_ids)

                # Simple progress logging
                if (i // batch_size) % 10 == 0:
                    print(
                        f"[{split_name}] Processed {i + len(batch_paths)}/{num_samples}"
                    )

        if not all_dino:
            raise RuntimeError(
                f"No features extracted for {split_name}. Check data paths."
            )

        return (
            np.concatenate(all_dino, axis=0),
            np.concatenate(all_conv, axis=0),
            np.array(processed_ids),
        )

    def extract_all(self, load_cached_data=True):
        """
        Main method to extract features for all splits (train, val, test).
        Returns tuples of (dino_features, conv_features, ids) for each split.
        """
        print("Starting feature extraction pipeline...")

        # Load metadata
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        # Process splits
        train_data = self._process_dataset(train_df, "train", load_cached_data)
        val_data = self._process_dataset(val_df, "val", load_cached_data)
        test_data = self._process_dataset(test_df, "test", load_cached_data)

        print("Feature extraction pipeline complete.")
        return train_data, val_data, test_data
