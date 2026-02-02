import os
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPTokenizer

from library.config import Config
from library.utils import (
    seed_everything,
    get_device,
    save_array,
    load_array,
    ensure_dir,
)
from library.dataset import PawpularityDataset, get_transforms, load_dataset

# =========================================================================
# Model Wrappers
# =========================================================================


class CLIPWrapper(nn.Module):
    """
    Wrapper for CLIP model to handle image encoding and Zero-Shot Aesthetic Injection.
    """

    def __init__(self, model_name, device):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        # Pre-compute text embeddings for aesthetic prompts
        self.prompts = Config.AESTHETIC_PROMPTS
        self.text_features = self._encode_text_prompts()

    def _encode_text_prompts(self):
        inputs = self.tokenizer(self.prompts, padding=True, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        # Normalize text features
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

    def forward(self, images):
        """
        Returns normalized image embeddings.
        """
        image_features = self.model.get_image_features(images)
        # Normalize image features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def post_process(self, image_features):
        """
        Computes Zero-Shot Scores and concatenates them with image embeddings.
        Args:
            image_features: Normalized image embeddings (B, D)
        Returns:
            Concatenated features (B, D + N_prompts)
        """
        # Calculate Cosine Similarity: (B, D) @ (N_prompts, D).T -> (B, N_prompts)
        scores = image_features @ self.text_features.t()

        # Concatenate embeddings and scores
        return torch.cat([image_features, scores], dim=1)


class TimmWrapper(nn.Module):
    """
    Wrapper for timm models (DINOv2, ConvNeXt).
    """

    def __init__(self, model_name, device):
        super().__init__()
        self.device = device
        # Create model with num_classes=0 to get feature vector (pooling handled by timm)
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(
            device
        )
        self.model.eval()

    def forward(self, images):
        return self.model(images)

    def post_process(self, image_features):
        """
        Identity function for consistency with CLIPWrapper.
        """
        return image_features


# =========================================================================
# Feature Extraction Logic
# =========================================================================


def get_cache_paths(model_name, split):
    """
    Generates file paths for caching features.
    """
    sanitized_name = model_name.replace("/", "_")
    prefix = f"{sanitized_name}_{split}"

    if Config.DEBUG:
        prefix += "_debug"

    base = os.path.join(Config.WORKING_DIR, prefix)

    return {
        "features": f"{base}_features.npy",
        "ids": f"{base}_ids.npy",
        "targets": f"{base}_targets.npy",
        "meta": f"{base}_meta.npy",
    }


def extract_and_save_features(model_name, split, load_cached_data=True, batch_size=64):
    """
    Extracts features using the specified backbone, applies augmentation, and caches results.

    Args:
        model_name (str): Name of the model (from Config).
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        batch_size (int): Batch size for inference.

    Returns:
        dict: Dictionary containing 'features', 'ids', 'targets', 'meta'.
    """
    seed_everything(Config.SEED)
    device = get_device()
    paths = get_cache_paths(model_name, split)

    # 1. Check Cache
    if load_cached_data:
        if os.path.exists(paths["features"]):
            print(f"Loading cached features for {model_name} ({split})...")
            try:
                data = {
                    "features": load_array(paths["features"]),
                    "ids": load_array(paths["ids"]),
                    "meta": load_array(paths["meta"]),
                }
                # Targets might not exist for test set
                if os.path.exists(paths["targets"]):
                    data["targets"] = load_array(paths["targets"])
                else:
                    data["targets"] = None
                return data
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Extracting features for {model_name} ({split})...")

    # 2. Setup Data
    # Determine image size and normalization stats based on model type
    if model_name == Config.MODEL_CLIP:
        img_size = Config.IMG_SIZE_CLIP
        # OpenAI CLIP specific mean/std
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        WrapperClass = CLIPWrapper
    elif model_name == Config.MODEL_DINO:
        img_size = Config.IMG_SIZE_DINO
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        WrapperClass = TimmWrapper
    elif model_name == Config.MODEL_CONVNEXT:
        img_size = Config.IMG_SIZE_CONVNEXT
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        WrapperClass = TimmWrapper
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    transform = get_transforms(img_size=img_size, mean=mean, std=std)
    df = load_dataset(split)
    dataset = PawpularityDataset(
        df, Config.INPUT_DIR, transform=transform, return_id=True
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 3. Setup Model
    model = WrapperClass(model_name, device)

    # 4. Extraction Loop
    all_features = []
    all_ids = []
    all_targets = []
    all_meta = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images" if "images" in batch else "image"].to(device)
            meta = batch["meta"].numpy()
            ids = batch["id"]

            # Feature-Space Augmentation: Horizontal Flip
            # 1. Forward pass original
            feats_orig = model(images)

            # 2. Forward pass flipped
            images_flipped = torch.flip(images, dims=[3])
            feats_flip = model(images_flipped)

            # 3. Average embeddings
            feats_avg = (feats_orig + feats_flip) / 2.0

            # 4. Re-normalize if CLIP (to maintain cosine similarity validity)
            if model_name == Config.MODEL_CLIP:
                feats_avg = feats_avg / feats_avg.norm(p=2, dim=-1, keepdim=True)

            # 5. Post-process (e.g., add Zero-Shot Scores for CLIP)
            final_feats = model.post_process(feats_avg)

            all_features.append(final_feats.cpu().numpy())
            all_ids.extend(ids)
            all_meta.append(meta)

            if "target" in batch:
                all_targets.append(batch["target"].numpy())

    # 5. Aggregate and Save
    features_arr = np.concatenate(all_features, axis=0)
    meta_arr = np.concatenate(all_meta, axis=0)
    ids_arr = np.array(all_ids)

    save_array(paths["features"], features_arr)
    save_array(paths["ids"], ids_arr)
    save_array(paths["meta"], meta_arr)

    result = {
        "features": features_arr,
        "ids": ids_arr,
        "meta": meta_arr,
        "targets": None,
    }

    if all_targets:
        targets_arr = np.concatenate(all_targets, axis=0)
        save_array(paths["targets"], targets_arr)
        result["targets"] = targets_arr

    print(f"Saved features to {Config.WORKING_DIR} (Shape: {features_arr.shape})")

    return result
