import os
import numpy as np
import torch
from torchvision.models import convnext_large
import library.config as config
import library.utils as utils


def load_backbone(device):
    """
    Loads the ConvNeXt-Large model with specified weights and modifies it for feature extraction.

    Args:
        device (torch.device): The device to load the model onto.

    Returns:
        torch.nn.Module: The modified ConvNeXt model.
    """
    # Load weights defined in config (IMAGENET1K_V1)
    weights = config.WEIGHTS

    # Initialize model
    # torchvision supports passing the string name of the weights
    model = convnext_large(weights=weights)

    # Replace the classification head with Identity to extract features.
    # The ConvNeXt classifier block is a Sequential(LayerNorm, Flatten, Linear).
    # We replace the Linear layer (index 2) with Identity to keep the normalization and flattening,
    # resulting in the 1536-dim feature vector.
    model.classifier[2] = torch.nn.Identity()

    model.to(device)
    model.eval()

    return model


def extract_embeddings(model, loader, device):
    """
    Extracts embeddings from the dataloader using the model.
    Implements Feature-Level Test Time Augmentation (TTA) by averaging
    features from original and horizontally flipped images.

    Args:
        model (torch.nn.Module): The feature extractor model.
        loader (DataLoader): DataLoader providing images and targets/ids.
        device (torch.device): Device to perform computation on.

    Returns:
        tuple: (embeddings, targets)
               embeddings is a numpy array of shape (N, 1536)
               targets is a numpy array of labels (int) or IDs (str)
    """
    embeddings_list = []
    targets_list = []

    # Ensure model is in eval mode
    model.eval()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            # 1. Forward pass: Original images
            emb_orig = model(images)

            # 2. Forward pass: Horizontally flipped images (TTA)
            # NCHW format, flip on width dimension (dim 3)
            images_flipped = torch.flip(images, dims=[3])
            emb_flip = model(images_flipped)

            # 3. Average the embeddings
            emb_avg = (emb_orig + emb_flip) / 2.0

            embeddings_list.append(emb_avg.cpu().numpy())

            # Handle targets (Labels for train/val, IDs for test)
            if isinstance(targets, torch.Tensor):
                # Move tensor labels to CPU numpy
                targets_list.append(targets.cpu().numpy())
            else:
                # Assuming targets is a tuple/list of strings (IDs) from the dataset
                targets_list.extend(targets)

    # Concatenate embeddings
    if embeddings_list:
        embeddings = np.vstack(embeddings_list)
    else:
        embeddings = np.array([])

    # Concatenate targets
    if targets_list:
        if isinstance(targets_list[0], np.ndarray):
            # Numeric labels (Train/Val)
            targets = np.concatenate(targets_list)
        else:
            # String IDs (Test)
            targets = np.array(targets_list)
    else:
        targets = np.array([])

    return embeddings, targets


def get_embeddings(view_type, split_name, loader, load_cached_data=True, model=None):
    """
    Retrieves embeddings for a specific view and split.
    Uses caching mechanism to avoid re-computation.

    Args:
        view_type (str): 'standard', 'global', or 'local'.
        split_name (str): 'train', 'val', or 'test'.
        loader (DataLoader): The data loader for the split.
        load_cached_data (bool): Whether to attempt loading from cache.
        model (nn.Module, optional): Pre-loaded model. If None and cache miss, model is loaded locally.

    Returns:
        tuple: (embeddings, targets)
    """
    # Define cache file paths
    # Naming convention: {view}_{split}_embeddings.npy
    # Targets file: {view}_{split}_labels.npy (train/val) or {view}_{split}_ids.npy (test)
    target_suffix = "ids" if split_name == "test" else "labels"

    emb_file = os.path.join(
        config.WORKING_DIR, f"{view_type}_{split_name}_embeddings.npy"
    )
    target_file = os.path.join(
        config.WORKING_DIR, f"{view_type}_{split_name}_{target_suffix}.npy"
    )

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(emb_file) and os.path.exists(target_file):
            print(f"Loading cached embeddings for {view_type} ({split_name})...")
            embeddings = np.load(emb_file)
            targets = np.load(target_file)
            return embeddings, targets

    # 2. Compute from scratch
    print(f"Computing embeddings for {view_type} ({split_name})...")

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Manage model lifecycle
    local_model_created = False
    if model is None:
        # Load model locally if not provided
        model = load_backbone(config.DEVICE)
        local_model_created = True
        device = config.DEVICE
    else:
        # Use provided model's device
        device = next(model.parameters()).device

    # Extract features
    embeddings, targets = extract_embeddings(model, loader, device)

    # Save to cache
    np.save(emb_file, embeddings)
    np.save(target_file, targets)
    print(f"Saved embeddings to {emb_file}")

    # Cleanup if model was created locally to free memory
    if local_model_created:
        del model
        torch.cuda.empty_cache()

    return embeddings, targets
