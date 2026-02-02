import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Subset
import numpy as np
from library import config, data_utils


def build_model(device):
    """
    Constructs the ConvNeXt-Large model with IMAGENET1K_V1 weights.
    Replaces the final classification layer with an Identity layer to extract features,
    while retaining the native Global Average Pooling and Layer Normalization.

    Args:
        device (torch.device): The device to load the model onto.

    Returns:
        model (torch.nn.Module): The feature extractor model.
    """
    # Load pre-trained weights
    weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
    model = models.convnext_large(weights=weights)

    # The classifier block in ConvNeXt from torchvision is a Sequential:
    # (0): LayerNorm2d((1536,), eps=1e-06, elementwise_affine=True)
    # (1): Flatten(start_dim=1, end_dim=-1)
    # (2): Linear(in_features=1536, out_features=1000, bias=True)
    # We replace the Linear layer (index 2) with Identity to keep the LayerNorm and Flatten.
    model.classifier[2] = nn.Identity()

    model.to(device)
    model.eval()
    return model


def extract_features(loader, model, device):
    """
    Runs inference on the data loader using the provided model.
    Handles multi-view inputs by flattening, inferring, and then aggregating (averaging)
    the view embeddings for each image.

    Args:
        loader (DataLoader): The data loader yielding (images, labels, ids).
        model (nn.Module): The feature extractor.
        device (torch.device): Computation device.

    Returns:
        embeddings (np.ndarray): Aggregated features of shape (N, D).
        labels (np.ndarray): Labels of shape (N,).
        ids (np.ndarray): Image IDs of shape (N,).
    """
    all_embeddings = []
    all_labels = []
    all_ids = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, labels, batch_ids in loader:
            # images shape: (B, V, C, H, W)
            # V is the number of views (2 for global/standard, 10 for local)
            b, v, c, h, w = images.shape

            # Flatten batch and view dimensions: (B*V, C, H, W)
            inputs = images.view(-1, c, h, w).to(device)

            # Mixed precision inference
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                features = model(inputs)

            # features shape: (B*V, D)
            embedding_dim = features.shape[1]

            # Reshape back to separate batch and views: (B, V, D)
            features = features.view(b, v, embedding_dim)

            # Spatial/View Aggregation: Average across all views/crops
            # Result shape: (B, D)
            features_avg = features.mean(dim=1)

            # Store results
            all_embeddings.append(features_avg.float().cpu().numpy())
            all_labels.append(labels.numpy())
            all_ids.extend(batch_ids)

    # Concatenate all batches
    if len(all_embeddings) > 0:
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        ids = np.array(all_ids)
    else:
        embeddings = np.array([])
        labels = np.array([])
        ids = np.array([])

    return embeddings, labels, ids


def get_features(split, view_type, load_cached_data=True, debug_sample_size=None):
    """
    Orchestrates the feature extraction process with caching.

    Args:
        split (str): 'train', 'val', or 'test'.
        view_type (str): 'global', 'standard', or 'local'.
        load_cached_data (bool): If True, attempts to load from disk.
        debug_sample_size (int, optional): If set, processes only a subset of data.

    Returns:
        embeddings, labels, ids
    """
    # Determine cache directory and filenames
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    suffix = ""
    if debug_sample_size is not None:
        suffix = f"_debug_{debug_sample_size}"

    emb_filename = f"{split}_{view_type}_embeddings{suffix}.npy"
    lbl_filename = f"{split}_{view_type}_labels{suffix}.npy"
    id_filename = f"{split}_{view_type}_ids{suffix}.npy"

    emb_path = os.path.join(cache_dir, emb_filename)
    lbl_path = os.path.join(cache_dir, lbl_filename)
    id_path = os.path.join(cache_dir, id_filename)

    # Attempt to load from cache
    if (
        load_cached_data
        and os.path.exists(emb_path)
        and os.path.exists(lbl_path)
        and os.path.exists(id_path)
    ):
        print(f"Loading cached features for {split}/{view_type} from {emb_path}...")
        embeddings = np.load(emb_path)
        labels = np.load(lbl_path)
        ids = np.load(id_path)
        return embeddings, labels, ids

    print(
        f"Generating features for {split}/{view_type} (Debug: {debug_sample_size})..."
    )

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)

    # Get Dataset
    dataset = data_utils.get_dataset(split, view_type)

    # Handle Debug Subset
    if debug_sample_size is not None and debug_sample_size < len(dataset):
        indices = list(range(debug_sample_size))
        dataset = Subset(dataset, indices)

    # Create Loader
    # Shuffle is False to preserve order and match IDs deterministically
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Extract
    embeddings, labels, ids = extract_features(loader, model, device)

    # Save to cache (only if we have data)
    if len(embeddings) > 0:
        np.save(emb_path, embeddings)
        np.save(lbl_path, labels)
        np.save(id_path, ids)
        print(f"Saved features to {emb_path}")

    return embeddings, labels, ids
