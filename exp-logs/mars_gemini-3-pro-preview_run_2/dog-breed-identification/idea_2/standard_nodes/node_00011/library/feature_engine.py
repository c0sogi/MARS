import os
import torch
import timm
import numpy as np
from torch.utils.data import DataLoader
from timm.data import resolve_data_config, create_transform

from library import config
from library import utils
from library import dataset


def get_backbone_transforms():
    """
    Creates the specific data transforms required for the CNN and ViT backbones.
    Uses timm to resolve the configuration from the pretrained models.

    Returns:
        dict: A dictionary with keys 'cnn' and 'vit' containing the respective transforms.
    """
    # Create temporary models just to read their config
    # We don't need to load weights here, just the architecture config for transforms
    cnn_model = timm.create_model(config.MODEL_CNN, pretrained=False, num_classes=0)
    vit_model = timm.create_model(config.MODEL_VIT, pretrained=False, num_classes=0)

    cnn_config = resolve_data_config({}, model=cnn_model)
    vit_config = resolve_data_config({}, model=vit_model)

    # Create transforms
    # is_training=False ensures deterministic validation/test transforms (center crop, etc.)
    cnn_transform = create_transform(**cnn_config, is_training=False)
    vit_transform = create_transform(**vit_config, is_training=False)

    return {"cnn": cnn_transform, "vit": vit_transform}


class FeatureExtractor:
    """
    Handles loading of a single frozen backbone and feature extraction with TTA.
    """

    def __init__(self, model_name):
        self.device = config.DEVICE
        print(f"Initializing FeatureExtractor for {model_name} on {self.device}...")

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_features(self, dataloader, img_key):
        """
        Iterates over the dataloader, applies TTA, and extracts embeddings.

        Args:
            dataloader: PyTorch DataLoader.
            img_key: Key to retrieve images from batch (e.g., 'cnn_img' or 'vit_img').

        Returns:
            tuple: (embeddings, labels, ids)
        """
        all_embeddings = []
        all_labels = []
        all_ids = []

        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch
                imgs = batch[img_key].to(self.device)
                labels = batch["label"].numpy()
                ids = batch["id"]  # tuple of strings

                # --- Test Time Augmentation (Horizontal Flip) ---
                # Create flipped versions
                imgs_flip = torch.flip(imgs, dims=[3])  # [B, C, H, W], flip W

                # Concatenate along batch dimension for efficient processing: [2B, C, H, W]
                batch_imgs = torch.cat([imgs, imgs_flip], dim=0)

                # Forward pass
                feats_batch = self.model(batch_imgs)

                # Split back
                batch_size = imgs.shape[0]
                feats_orig = feats_batch[:batch_size]
                feats_flip = feats_batch[batch_size:]

                # Average embeddings
                feats = (feats_orig + feats_flip) / 2.0

                # Store results
                all_embeddings.append(feats.cpu().numpy())
                all_labels.append(labels)
                all_ids.extend(ids)

        # Concatenate all batches
        if len(all_embeddings) > 0:
            all_embeddings = np.vstack(all_embeddings)
            all_labels = np.concatenate(all_labels)
            all_ids = np.array(all_ids)
        else:
            all_embeddings = np.array([])
            all_labels = np.array([])
            all_ids = np.array([])

        return all_embeddings, all_labels, all_ids


def extract_embeddings(
    metadata_path,
    embedding_path,
    label_path,
    id_path,
    model_name,
    img_key,
    load_cached_data=True,
    debug=False,
):
    """
    Main function to handle data loading, feature extraction, and caching.

    Args:
        metadata_path (str): Path to the metadata CSV.
        embedding_path (str): Path to save/load embeddings (.npy).
        label_path (str): Path to save/load labels (.npy).
        id_path (str): Path to save/load IDs (.npy).
        load_cached_data (bool): If True, attempts to load from disk first.
        debug (bool/int): If True, processes a subset of data.

    Returns:
        tuple: (embeddings, labels, ids)
    """
    # 1. Check Cache
    if load_cached_data:
        if (
            os.path.exists(embedding_path)
            and os.path.exists(label_path)
            and os.path.exists(id_path)
        ):
            print(f"Loading cached embeddings from {embedding_path}...")
            embeddings = utils.load_array(embedding_path)
            labels = utils.load_array(label_path)
            ids = utils.load_array(id_path)
            return embeddings, labels, ids
        else:
            print("Cache not found. Computing features from scratch...")

    # 2. Setup Data
    print(f"Setting up dataset from {metadata_path}...")

    # Get transforms
    transforms = get_backbone_transforms()

    # We need class mappings for consistency, though labels are handled numerically in dataset
    # if it's the training set. For test set, labels will be dummy (-1).
    # We'll rely on the dataset class to handle this, but we can pass the mapping if we have it
    # to ensure validation set matches training set mapping.
    # Ideally, we load the training mapping.
    class_to_idx = None
    if "train" not in metadata_path:
        # If processing val or test, try to use the mapping derived from train
        # This assumes train.csv exists.
        try:
            class_to_idx, _ = dataset.get_class_mappings(config.TRAIN_METADATA_PATH)
        except Exception:
            print(
                "Warning: Could not load class mapping from train.csv. Using default dataset logic."
            )

    ds = dataset.DogDataset(
        metadata_path=metadata_path,
        transforms=transforms,
        class_to_idx=class_to_idx,
        debug=debug,
    )

    loader = DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Extract Features
    extractor = FeatureExtractor(model_name=model_name)
    embeddings, labels, ids = extractor.extract_features(loader, img_key=img_key)

    # 4. Save to Cache
    print(f"Saving features to {embedding_path}...")
    utils.save_array(embeddings, embedding_path)
    utils.save_array(labels, label_path)
    utils.save_array(ids, id_path)

    return embeddings, labels, ids
