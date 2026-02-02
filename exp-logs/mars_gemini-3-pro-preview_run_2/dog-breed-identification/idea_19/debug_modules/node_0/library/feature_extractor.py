import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader, Subset
from library.config import Config, StreamConfig
from library.data_loader import DogDataset
from library.utils import save_array, load_array


def get_cache_subdir(stream_config: StreamConfig) -> str:
    """
    Determines the cache subdirectory name based on the stream configuration
    to prevent collisions and ensure organization.
    """
    if "convnext" in stream_config.arch:
        return "convnext_large_v1"
    elif "regnet" in stream_config.arch:
        return "regnet_y_128gf_swag"
    return stream_config.name


def load_backbone(stream_config: StreamConfig):
    """
    Loads the pre-trained backbone specified in the stream configuration,
    removes the classification head, and moves it to the configured device.
    """
    print(
        f"Loading backbone: {stream_config.arch} with weights {stream_config.weights}"
    )

    if stream_config.arch == "convnext_large":
        model = models.convnext_large(weights=stream_config.weights)
        # ConvNeXt Classifier structure:
        # (0): LayerNorm2d((1536,), eps=1e-06, elementwise_affine=True)
        # (1): Flatten(start_dim=1, end_dim=-1)
        # (2): Linear(in_features=1536, out_features=1000, bias=True)
        # We replace the Linear layer with Identity to get the flattened, normalized embeddings.
        model.classifier[2] = nn.Identity()

    elif stream_config.arch == "regnet_y_128gf":
        model = models.regnet_y_128gf(weights=stream_config.weights)
        # RegNet structure ends with:
        # (avgpool): AdaptiveAvgPool2d(output_size=1)
        # (flatten): Flatten(start_dim=1, end_dim=-1)
        # (fc): Linear(...)
        # We replace the fc layer with Identity.
        model.fc = nn.Identity()

    else:
        raise ValueError(f"Architecture {stream_config.arch} is not supported.")

    model = model.to(Config.DEVICE)
    model.eval()
    return model


def extract_features(model, dataloader, view_name: str):
    """
    Extracts features for a specific geometric view using the provided model.
    Implements Test-Time Augmentation (Horizontal Flip).
    """
    embeddings = []
    ids = []
    labels = []

    # Ensure model is in eval mode
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            # Extract data
            imgs = batch[view_name].to(Config.DEVICE)
            batch_ids = batch["id"]

            # Forward Pass - Original
            feats_orig = model(imgs)

            # Forward Pass - Horizontal Flip (TTA)
            imgs_flip = imgs.flip(-1)  # N, C, H, W -> flip W
            feats_flip = model(imgs_flip)

            # Average embeddings
            feats_avg = (feats_orig + feats_flip) / 2.0

            # Move to CPU and store
            embeddings.append(feats_avg.cpu().numpy())
            ids.extend(batch_ids)

            if "label" in batch:
                labels.append(batch["label"].numpy())

    # Concatenate all batches
    embeddings = np.concatenate(embeddings, axis=0)
    ids = np.array(ids)

    if labels:
        labels = np.concatenate(labels, axis=0)
    else:
        labels = None

    return embeddings, ids, labels


def get_concatenated_features(
    stream_config: StreamConfig,
    mode: str,
    load_cached_data: bool = True,
    max_samples: int = None,
):
    """
    Orchestrates the feature extraction process for a given stream and dataset mode (train/val/test).
    Manages caching, multi-view extraction, and concatenation.

    Args:
        stream_config: Configuration for the stream.
        mode: 'train', 'val', or 'test'.
        load_cached_data: Whether to attempt loading from cache.
        max_samples: Optional limit on number of samples (for debugging).

    Returns:
        X (np.ndarray): Concatenated embeddings (Global + Standard + Local).
        ids (np.ndarray): Image IDs.
        y (np.ndarray or None): Labels.
    """
    print(f"Preparing features for {stream_config.name} ({mode})...")

    # Determine metadata path
    if mode == "train":
        metadata_path = Config.TRAIN_METADATA
    elif mode == "val":
        metadata_path = Config.VAL_METADATA
    elif mode == "test":
        metadata_path = Config.TEST_METADATA
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Setup Cache Directory
    cache_subdir = get_cache_subdir(stream_config)
    # Ensure the subdirectory exists inside working dir
    full_cache_dir = os.path.join(Config.WORKING_DIR, cache_subdir)
    os.makedirs(full_cache_dir, exist_ok=True)

    views = ["global", "standard", "local"]
    view_embeddings = {}
    final_ids = None
    final_labels = None

    # Check if we need to run extraction
    model = None

    for view in views:
        # Define filenames
        # e.g. convnext_large_v1/train_global_embeddings.npy
        base_name = f"{mode}_{view}"
        emb_filename = os.path.join(cache_subdir, f"{base_name}_embeddings.npy")
        ids_filename = os.path.join(cache_subdir, f"{base_name}_ids.npy")
        lbl_filename = os.path.join(cache_subdir, f"{base_name}_labels.npy")

        # Check cache
        # Note: load_array takes relative path from WORKING_DIR
        cached_emb = load_array(emb_filename) if load_cached_data else None
        cached_ids = load_array(ids_filename) if load_cached_data else None
        cached_lbl = load_array(lbl_filename) if load_cached_data else None

        if cached_emb is not None and cached_ids is not None:
            # If labels are expected but missing, invalidate cache
            if mode in ["train", "val"] and cached_lbl is None:
                print(
                    f"  Cache incomplete for {view} (missing labels). Re-computing..."
                )
            else:
                print(f"  Loaded {view} features from cache.")
                view_embeddings[view] = cached_emb
                final_ids = cached_ids  # Assume consistency
                final_labels = cached_lbl
                continue

        # If not cached, we need to extract
        print(f"  Extracting {view} features...")

        # Initialize model and loader only if needed
        if model is None:
            model = load_backbone(stream_config)

            dataset = DogDataset(metadata_path, stream_config, mode=mode)
            if max_samples is not None:
                dataset = Subset(dataset, range(min(len(dataset), max_samples)))

            dataloader = DataLoader(
                dataset,
                batch_size=stream_config.batch_size,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

        # Extract
        emb, ids, lbl = extract_features(model, dataloader, view)

        # Save to cache
        save_array(emb_filename, emb)
        save_array(ids_filename, ids)
        if lbl is not None:
            save_array(lbl_filename, lbl)

        view_embeddings[view] = emb
        final_ids = ids
        final_labels = lbl

    # Clean up model to free GPU memory
    if model is not None:
        del model
        torch.cuda.empty_cache()

    # Concatenate Views: Global -> Standard -> Local
    # Shape: (N, D) -> (N, 3*D)
    print("  Concatenating views...")
    X = np.concatenate(
        [
            view_embeddings["global"],
            view_embeddings["standard"],
            view_embeddings["local"],
        ],
        axis=1,
    )

    return X, final_ids, final_labels
