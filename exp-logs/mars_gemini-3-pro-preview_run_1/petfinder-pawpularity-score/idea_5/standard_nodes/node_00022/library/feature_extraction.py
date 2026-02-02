import os
import torch
import torch.nn as nn
import numpy as np
import timm
from transformers import CLIPModel, AutoModel
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import PawpularityDataset
from library.utils import setup_logger

logger = setup_logger("FeatureExtraction")


class ModelFactory:
    """
    Factory class to instantiate and configure the specific backbone models.
    """

    @staticmethod
    def load_model(model_name: str, device: str = Config.DEVICE):
        if model_name not in Config.MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        cfg = Config.MODELS[model_name]
        lib = cfg["library"]
        name = cfg["name"]

        logger.info(f"Loading {model_name} ({name}) via {lib}...")

        model = None

        if model_name == "clip":
            # Load CLIP Vision Model
            # We use the full CLIPModel but will only access the vision tower or helper method
            model = CLIPModel.from_pretrained(name)

        elif model_name == "dinov2":
            # Load DINOv2
            model = AutoModel.from_pretrained(name)

        elif model_name == "convnext":
            # Load ConvNeXt via timm
            # num_classes=0 returns the global pool features
            model = timm.create_model(name, pretrained=True, num_classes=0)

        else:
            raise NotImplementedError(f"Model {model_name} logic not implemented.")

        model.to(device)
        model.eval()
        return model

    @staticmethod
    def forward_batch(model, model_name: str, images: torch.Tensor):
        """
        Performs the forward pass to extract embeddings based on the model type.
        """
        if model_name == "clip":
            # CLIP get_image_features returns the projected features (768 dim)
            # It expects 'pixel_values'
            return model.get_image_features(pixel_values=images)

        elif model_name == "dinov2":
            # DINOv2 output is BaseModelOutputWithPooling or similar
            # last_hidden_state is (B, Seq, D). CLS token is at index 0.
            outputs = model(pixel_values=images)
            last_hidden_state = outputs.last_hidden_state
            cls_token = last_hidden_state[:, 0, :]
            return cls_token

        elif model_name == "convnext":
            # timm models with num_classes=0 return pooled features directly
            return model(images)

        return None


def extract_features(
    model_name: str,
    metadata_path: str,
    mode: str = "test",
    load_cached_data: bool = True,
    device: str = Config.DEVICE,
):
    """
    Extracts features for a given dataset split using the specified model.
    Implements Feature-Space Augmentation (Original + Horizontal Flip).
    Handles caching of features to disk.

    Args:
        model_name (str): 'clip', 'dinov2', or 'convnext'.
        metadata_path (str): Path to the metadata CSV.
        mode (str): Dataset mode ('train', 'val', 'test'). Used for naming cache files.
                    Note: We force the Dataset to use deterministic transforms ('test' mode logic)
                    internally to ensure consistent center crops for flip averaging.
        load_cached_data (bool): If True, attempts to load from disk.
        device (str): Computation device.

    Returns:
        tuple: (features, meta_features, targets, ids)
    """

    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Construct filenames
    # Example: clip_train_features.npy
    f_path = os.path.join(cache_dir, f"{model_name}_{mode}_features.npy")
    m_path = os.path.join(cache_dir, f"{model_name}_{mode}_meta.npy")
    t_path = os.path.join(cache_dir, f"{model_name}_{mode}_targets.npy")
    i_path = os.path.join(cache_dir, f"{model_name}_{mode}_ids.npy")

    # 1. Try Loading Cache
    if load_cached_data:
        if (
            os.path.exists(f_path)
            and os.path.exists(m_path)
            and os.path.exists(t_path)
            and os.path.exists(i_path)
        ):
            logger.info(
                f"Loading cached features for {model_name} ({mode}) from {cache_dir}..."
            )
            features = np.load(f_path)
            meta = np.load(m_path)
            targets = np.load(t_path)
            ids = np.load(i_path, allow_pickle=True)  # IDs are strings
            return features, meta, targets, ids
        else:
            logger.info(f"Cache miss for {model_name} ({mode}). Computing features...")
    else:
        logger.info(f"Force re-compute for {model_name} ({mode})...")

    # 2. Setup Data
    # We use mode='test' for the Dataset to ensure deterministic preprocessing (Resize + CenterCrop)
    # regardless of whether we are processing the training or validation set.
    # We will handle the flip augmentation manually on the tensor.
    dataset = PawpularityDataset(
        metadata_path=metadata_path,
        model_name=model_name,
        mode="test",  # Enforce deterministic transforms
    )

    batch_size = Config.MODELS[model_name]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Setup Model
    model = ModelFactory.load_model(model_name, device)

    # 4. Extraction Loop
    all_features = []
    all_meta = []
    all_targets = []
    all_ids = []

    logger.info(f"Starting extraction for {len(dataset)} images...")

    with torch.no_grad():
        for i, (images, meta, targets, sample_ids) in enumerate(loader):
            images = images.to(device)

            # Original Forward Pass
            emb_orig = ModelFactory.forward_batch(model, model_name, images)

            # Flipped Forward Pass (Feature-Space Augmentation)
            # dim 3 is width (B, C, H, W)
            images_flip = torch.flip(images, dims=[3])
            emb_flip = ModelFactory.forward_batch(model, model_name, images_flip)

            # Average Embeddings
            emb_avg = (emb_orig + emb_flip) / 2.0

            # Move to CPU and collect
            all_features.append(emb_avg.cpu().numpy())
            all_meta.append(meta.numpy())
            all_targets.append(targets.numpy())
            all_ids.extend(sample_ids)

            if (i + 1) % 50 == 0:
                logger.info(f"Processed batch {i + 1}/{len(loader)}")

    # 5. Concatenate
    features = np.concatenate(all_features, axis=0)
    meta_features = np.concatenate(all_meta, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    ids = np.array(all_ids)

    logger.info(f"Extraction complete. Shape: {features.shape}")

    # 6. Save to Cache
    np.save(f_path, features)
    np.save(m_path, meta_features)
    np.save(t_path, targets)
    np.save(i_path, ids)
    logger.info(f"Saved features to {cache_dir}")

    return features, meta_features, targets, ids
