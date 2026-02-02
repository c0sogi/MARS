import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import library.config as config
import library.data_utils as data_utils


def load_backbone(stream_config):
    """
    Loads the pretrained backbone specified in the stream configuration
    and replaces the classification head with an Identity layer.

    Args:
        stream_config (dict): Configuration dictionary for the stream.

    Returns:
        model (nn.Module): The feature extractor model.
    """
    model_name = stream_config["model_name"]
    weights = stream_config["weights"]

    print(f"Loading model: {model_name} with weights: {weights}")

    if model_name == "convnext_large":
        # Load ConvNeXt Large
        # Structure: features -> avgpool -> classifier (Sequential: LayerNorm, Flatten, Linear)
        model = torchvision.models.convnext_large(weights=weights)
        # Replace the final Linear layer (index 2 in classifier) with Identity
        model.classifier[2] = nn.Identity()

    elif model_name == "regnet_y_128gf":
        # Load RegNetY 128GF
        # Structure: stem -> trunk -> avgpool -> fc
        model = torchvision.models.regnet_y_128gf(weights=weights)
        # Replace the final Fully Connected layer with Identity
        model.fc = nn.Identity()

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model.to(config.DEVICE)
    model.eval()
    return model


def extract_features(stream_config, split, view, load_cached_data=True):
    """
    Extracts features for a specific stream, data split, and view.
    Applies Test Time Augmentation (Horizontal Flip).
    Handles caching of embeddings.

    Args:
        stream_config (dict): Configuration dictionary for the stream.
        split (str): 'train', 'val', or 'test'.
        view (str): 'global', 'standard', or 'local'.
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        embeddings (np.ndarray): Extracted features (N, D).
        ids (np.ndarray): Image IDs (N,).
        labels (np.ndarray): Image labels (N,).
    """
    stream_name = stream_config["name"]

    # Define cache file paths
    cache_prefix = f"{stream_name}_{split}_{view}"
    emb_path = os.path.join(config.CACHE_DIR, f"{cache_prefix}_embeddings.npy")
    ids_path = os.path.join(config.CACHE_DIR, f"{cache_prefix}_ids.npy")
    lbl_path = os.path.join(config.CACHE_DIR, f"{cache_prefix}_labels.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(emb_path)
            and os.path.exists(ids_path)
            and os.path.exists(lbl_path)
        ):
            print(f"Loading cached features for {stream_name} | {split} | {view} ...")
            embeddings = np.load(emb_path)
            ids = np.load(ids_path, allow_pickle=True)
            labels = np.load(lbl_path)
            return embeddings, ids, labels
        else:
            print(
                f"Cache miss for {stream_name} | {split} | {view}. Computing features..."
            )

    # 2. Setup Data
    # Determine metadata file
    if split == "train":
        metadata_path = config.TRAIN_CSV
    elif split == "val":
        metadata_path = config.VAL_CSV
    elif split == "test":
        metadata_path = config.TEST_CSV
    else:
        raise ValueError(f"Unknown split: {split}")

    # Build transforms
    transform_dict = data_utils.build_stream_transforms(stream_config)

    # We only need the transform for the specific view requested
    # But DogDataset expects the full dict, or we can construct a dict with just the one we need.
    # The DogDataset implementation iterates over the dict keys.
    # To save compute, we can pass a dict with ONLY the relevant view.
    specific_transform_dict = {view: transform_dict[view]}

    # Load Label Map
    label_to_idx, _ = data_utils.get_label_map()

    dataset = data_utils.DogDataset(
        metadata_path=metadata_path,
        transform_dict=specific_transform_dict,
        label_to_idx=label_to_idx,
        return_label=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=stream_config["batch_size"],
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Setup Model
    model = load_backbone(stream_config)

    # 4. Inference Loop
    all_embeddings = []
    all_ids = []
    all_labels = []

    print(f"Extracting features: {stream_name} | {split} | {view}")

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch
            # batch['views'] is a dict: {view_name: Tensor}
            imgs = batch["views"][view].to(config.DEVICE)
            batch_ids = batch["id"]
            batch_labels = batch["label"].numpy()

            # TTA: Original
            features_orig = model(imgs)

            # TTA: Horizontal Flip
            imgs_flipped = torch.flip(imgs, dims=[3])  # N, C, H, W -> flip W
            features_flip = model(imgs_flipped)

            # Average features
            features_avg = (features_orig + features_flip) / 2.0

            # Move to CPU and store
            all_embeddings.append(features_avg.cpu().numpy())
            all_ids.extend(batch_ids)
            all_labels.append(batch_labels)

    # Concatenate results
    embeddings = np.concatenate(all_embeddings, axis=0)
    ids = np.array(all_ids)
    labels = np.concatenate(all_labels, axis=0)

    # 5. Save to Cache
    print(f"Saving features to {config.CACHE_DIR}...")
    np.save(emb_path, embeddings)
    np.save(ids_path, ids)
    np.save(lbl_path, labels)

    # Cleanup model to free GPU memory
    del model
    torch.cuda.empty_cache()

    return embeddings, ids, labels
