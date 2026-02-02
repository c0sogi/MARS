import os
import torch
import numpy as np
from transformers import AutoModel, AutoImageProcessor, CLIPVisionModel, logging
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import get_dataset
from library.utils import seed_everything

# Suppress Hugging Face warnings to keep output clean
logging.set_verbosity_error()


class FeatureExtractor:
    """
    Handles feature extraction from pre-trained backbones with Test-Time Augmentation (TTA)
    and disk-based caching.
    """

    def __init__(self, model_name, device=Config.DEVICE):
        """
        Args:
            model_name (str): Hugging Face model ID (e.g., 'openai/clip-vit-large-patch14').
            device (str): Computation device ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.device = device

        # Generate a safe alias for filenames (e.g., 'openai_clip...' -> 'clip')
        if "clip" in model_name.lower():
            self.alias = "clip"
        elif "dino" in model_name.lower():
            self.alias = "dino"
        elif "convnext" in model_name.lower():
            self.alias = "convnext"
        else:
            self.alias = model_name.replace("/", "_").replace("-", "_")

    def _load_model(self):
        """Loads the specific model architecture based on the model name."""
        print(f"Loading model: {self.model_name}")

        if "clip" in self.model_name.lower():
            # For CLIP, we only need the vision tower
            model = CLIPVisionModel.from_pretrained(self.model_name)
        else:
            # For DINOv2 and ConvNeXt, AutoModel works well
            model = AutoModel.from_pretrained(self.model_name)

        model.to(self.device)
        model.eval()
        return model

    def _get_transform(self):
        """
        Creates a transform function using the model's specific AutoImageProcessor.
        Adapts the processor's output (dict) to a single tensor for the Dataset.
        """
        try:
            processor = AutoImageProcessor.from_pretrained(self.model_name)
        except Exception:
            # Fallback for some CLIP models if AutoImageProcessor is not mapped
            from transformers import CLIPProcessor

            processor = CLIPProcessor.from_pretrained(self.model_name)

        def transform(image):
            # processor returns a dict containing 'pixel_values'
            # We assume input is a PIL image
            inputs = processor(images=image, return_tensors="pt")
            # Return the tensor (C, H, W) removing the batch dim added by processor
            return inputs["pixel_values"][0]

        return transform

    def _extract_batch(self, model, images):
        """
        Performs the forward pass and extracts the correct embedding vector
        depending on the model architecture.
        """
        inputs = {"pixel_values": images.to(self.device)}

        with torch.no_grad():
            outputs = model(**inputs)

        # Extraction logic based on model type
        if "clip" in self.model_name.lower():
            # CLIP: Use projected pooler output
            return outputs.pooler_output

        elif "dino" in self.model_name.lower():
            # DINOv2: Use CLS token (index 0 of last hidden state)
            # DINOv2 usually doesn't have a pooler head initialized
            return outputs.last_hidden_state[:, 0, :]

        elif "convnext" in self.model_name.lower():
            # ConvNeXt: Use pooler_output (Global Average Pooling + LayerNorm)
            return outputs.pooler_output

        else:
            # Generic fallback
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                return outputs.last_hidden_state[:, 0, :]

    def extract(self, split, load_cached_data=True, debug=False):
        """
        Extracts features for a given dataset split.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load from disk.
            debug (bool): If True, runs on a small subset.

        Returns:
            Tuple[np.ndarray]: (features, ids, meta, targets)
        """
        seed_everything(Config.SEED)

        # Ensure cache directory exists
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        # Construct filenames
        prefix = f"{self.alias}_{split}"
        if debug:
            prefix += "_debug"

        path_feats = os.path.join(cache_dir, f"{prefix}_features.npy")
        path_ids = os.path.join(cache_dir, f"{prefix}_ids.npy")
        path_meta = os.path.join(cache_dir, f"{prefix}_meta.npy")
        path_targets = os.path.join(cache_dir, f"{prefix}_targets.npy")

        # Check if files exist
        files_exist = all(
            os.path.exists(p) for p in [path_feats, path_ids, path_meta, path_targets]
        )

        # 1. Try to load from cache
        if load_cached_data and files_exist:
            print(f"[{self.alias}] Loading cached features for '{split}' split...")
            features = np.load(path_feats)
            ids = np.load(path_ids)
            meta = np.load(path_meta)
            targets = np.load(path_targets)
            return features, ids, meta, targets

        # 2. Compute from scratch
        print(f"[{self.alias}] Extracting features for '{split}' split...")

        model = self._load_model()
        transform = self._get_transform()

        dataset = get_dataset(split, transform=transform, debug=debug)
        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        all_feats = []
        all_ids = []
        all_meta = []
        all_targets = []

        for images, meta, targets, ids in dataloader:
            # TTA: Create horizontally flipped version
            # images shape: (B, C, H, W) -> Flip on W (dim 3)
            images_flip = torch.flip(images, dims=[3])

            # Extract features for both original and flipped
            emb_orig = self._extract_batch(model, images)
            emb_flip = self._extract_batch(model, images_flip)

            # Average embeddings (Enforce invariance)
            emb = (emb_orig + emb_flip) / 2.0

            # Move to CPU and store
            all_feats.append(emb.cpu().numpy())
            all_ids.extend(ids)
            all_meta.append(meta.numpy())
            all_targets.append(targets.numpy())

        # Concatenate results
        features = np.concatenate(all_feats, axis=0)
        ids = np.array(all_ids)
        meta = np.concatenate(all_meta, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Save to cache
        print(f"[{self.alias}] Saving features to {cache_dir}...")
        np.save(path_feats, features)
        np.save(path_ids, ids)
        np.save(path_meta, meta)
        np.save(path_targets, targets)

        # Cleanup to free GPU memory for next model
        del model
        torch.cuda.empty_cache()

        return features, ids, meta, targets
