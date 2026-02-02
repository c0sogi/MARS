import os
import torch
import timm
import numpy as np
import library.config as config


def load_backbone(stream_name: str, device: str = config.DEVICE):
    """
    Loads the backbone model for the specified stream.

    Args:
        stream_name (str): 'stream_a' or 'stream_b'.
        device (str): Device to load the model onto.

    Returns:
        torch.nn.Module: The loaded backbone model in eval mode.
    """
    kwargs = {}
    if stream_name == "stream_a":
        model_name = config.STREAM_A_MODEL_NAME
    elif stream_name == "stream_b":
        model_name = config.STREAM_B_MODEL_NAME
        # DINOv2 defaults to 518x518, but our pipeline uses 224x224.
        # Explicitly set img_size to trigger positional embedding interpolation.
        kwargs["img_size"] = config.IMG_SIZE
    else:
        raise ValueError(f"Unknown stream_name: {stream_name}")

    # Create model with no classifier (num_classes=0) to get embeddings.
    # This retains the final normalization layer (e.g. LayerNorm or Global Pool + Norm)
    # ensuring the features are properly scaled.
    model = timm.create_model(model_name, pretrained=True, num_classes=0, **kwargs)
    model.to(device)
    model.eval()

    return model


def extract_and_save_features(
    loader,
    model,
    save_dir: str,
    device: str = config.DEVICE,
    load_cached_data: bool = True,
):
    """
    Extracts features from the dataset using the provided model and saves them to disk.
    Implements Test Time Augmentation (Horizontal Flip) and caches results.

    Args:
        loader (DataLoader): DataLoader providing 'view_global', 'view_standard', 'view_local', 'id', 'label'.
        model (nn.Module): The backbone model.
        save_dir (str): Directory to save/load cached embeddings.
        device (str): Device for computation.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing 'global', 'standard', 'local' embeddings, 'ids', and 'labels'.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Define cache file paths
    files = {
        "global": os.path.join(save_dir, "embeddings_global.npy"),
        "standard": os.path.join(save_dir, "embeddings_standard.npy"),
        "local": os.path.join(save_dir, "embeddings_local.npy"),
        "ids": os.path.join(save_dir, "ids.npy"),
        "labels": os.path.join(save_dir, "labels.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(f) for f in files.values())

    if load_cached_data and cache_exists:
        print(f"Loading cached features from {save_dir}...")
        data = {}
        # Load arrays. allow_pickle=True is used to handle string arrays (ids) safely if needed,
        # though numpy handles unicode strings natively in recent versions.
        data["global"] = np.load(files["global"])
        data["standard"] = np.load(files["standard"])
        data["local"] = np.load(files["local"])
        data["ids"] = np.load(files["ids"], allow_pickle=True)
        data["labels"] = np.load(files["labels"])
        return data

    print(f"Extracting features to {save_dir}...")

    # Storage for features and metadata
    feats_global = []
    feats_standard = []
    feats_local = []
    all_ids = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            # Extract IDs and Labels
            # IDs are strings, Labels are tensors
            ids = batch["id"]
            labels = batch["label"].numpy()

            all_ids.extend(ids)
            all_labels.extend(labels)

            # Process each view independently
            # View names match keys in MultiViewDataset
            views_to_process = [
                ("view_global", feats_global),
                ("view_standard", feats_standard),
                ("view_local", feats_local),
            ]

            for view_name, storage in views_to_process:
                imgs = batch[view_name].to(device)

                # TTA: Create horizontally flipped version
                imgs_flip = torch.flip(imgs, dims=[3])

                # Forward Pass
                emb = model(imgs)
                emb_flip = model(imgs_flip)

                # Average embeddings (Cache Safety & TTA)
                emb_avg = (emb + emb_flip) / 2.0

                storage.append(emb_avg.cpu().numpy())

    # Concatenate lists into numpy arrays
    data = {
        "global": np.concatenate(feats_global, axis=0),
        "standard": np.concatenate(feats_standard, axis=0),
        "local": np.concatenate(feats_local, axis=0),
        "ids": np.array(all_ids),
        "labels": np.array(all_labels),
    }

    # Save to disk
    print(f"Saving features to {save_dir}...")
    np.save(files["global"], data["global"])
    np.save(files["standard"], data["standard"])
    np.save(files["local"], data["local"])
    np.save(files["ids"], data["ids"])
    np.save(files["labels"], data["labels"])

    return data
