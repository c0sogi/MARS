import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.models as models
from library.config import Config
from library.dataset import DogBreedDataset
from library.transforms import get_stream_transforms


def get_class_to_idx():
    """
    Generates a consistent class-to-index mapping based on the training data.
    """
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Train metadata not found at {Config.TRAIN_METADATA_PATH}"
        )

    df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    unique_breeds = sorted(df["breed"].unique())
    return {breed: idx for idx, breed in enumerate(unique_breeds)}


def load_model(stream_config):
    """
    Loads the specified model architecture and weights, removing the classification head
    to allow for feature extraction.
    """
    model_name = stream_config["name"]
    weights = stream_config["weights"]

    # Load model from torchvision
    if hasattr(models, model_name):
        model_fn = getattr(models, model_name)
        try:
            model = model_fn(weights=weights)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model {model_name} with weights {weights}: {e}"
            )
    else:
        raise ValueError(f"Model {model_name} not found in torchvision.models")

    # Remove classification head to output embeddings
    # ConvNeXt Architecture
    if "convnext" in model_name:
        # Classifier is typically: Sequential(LayerNorm2d, Flatten, Linear)
        # We replace the Linear layer (index 2) with Identity to keep the pooling/flattening
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            if len(model.classifier) >= 3:
                model.classifier[2] = nn.Identity()
            else:
                model.classifier = nn.Identity()
        else:
            model.classifier = nn.Identity()

    # Vision Transformer (ViT) Architecture
    elif "vit" in model_name:
        # Heads is typically: Sequential(Linear)
        if hasattr(model, "heads"):
            model.heads = nn.Identity()
        else:
            raise AttributeError(
                "ViT model does not have 'heads' attribute as expected."
            )

    # Move to device and set to eval mode
    model = model.to(Config.DEVICE)
    model.eval()

    return model


def extract_features(
    stream_config, dataset_type, load_cached_data=True, debug_subset_size=None
):
    """
    Extracts features for a specific stream and dataset split using the Multi-View TTA strategy.

    Args:
        stream_config (dict): Configuration for the stream (A or B).
        dataset_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug_subset_size (int): Number of samples to process for debugging.

    Returns:
        tuple: (embeddings, labels, ids)
               embeddings: np.ndarray of shape (N, embedding_dim * 3)
               labels: np.ndarray of shape (N,) or None for test
               ids: np.ndarray of shape (N,)
    """

    # Determine paths and cache filenames
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    prefix = stream_config["cache_prefix"]
    suffix = ""
    if debug_subset_size is not None:
        suffix = f"_debug_{debug_subset_size}"

    emb_path = os.path.join(
        cache_dir, f"{prefix}_{dataset_type}_embeddings{suffix}.npy"
    )
    lbl_path = os.path.join(cache_dir, f"{prefix}_{dataset_type}_labels{suffix}.npy")
    id_path = os.path.join(cache_dir, f"{prefix}_{dataset_type}_ids{suffix}.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        # Check if files exist
        files_exist = os.path.exists(emb_path) and os.path.exists(id_path)
        if dataset_type != "test":
            files_exist = files_exist and os.path.exists(lbl_path)

        if files_exist:
            print(f"Loading cached features for {prefix} - {dataset_type}...")
            embeddings = np.load(emb_path)
            ids = np.load(id_path, allow_pickle=True)

            labels = None
            if dataset_type != "test":
                labels = np.load(lbl_path)

            return embeddings, labels, ids

    print(f"Computing features for {prefix} - {dataset_type}...")

    # 2. Setup Metadata and Dataset
    if dataset_type == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        is_test = False
    elif dataset_type == "val":
        meta_path = Config.VAL_METADATA_PATH
        is_test = False
    elif dataset_type == "test":
        meta_path = Config.TEST_METADATA_PATH
        is_test = True
    else:
        raise ValueError("dataset_type must be 'train', 'val', or 'test'")

    df = pd.read_csv(meta_path)

    # Get transforms and class mapping
    transforms = get_stream_transforms(stream_config)
    class_to_idx = get_class_to_idx() if not is_test else None

    # Initialize Dataset
    dataset = DogBreedDataset(
        df=df,
        transforms=transforms,
        input_dir=Config.INPUT_DIR,
        class_to_idx=class_to_idx,
        is_test=is_test,
        debug_subset_size=debug_subset_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=stream_config["batch_size"],
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    model = load_model(stream_config)

    # 4. Extraction Loop
    all_embeddings = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            # Extract IDs
            batch_ids = batch["id"]
            all_ids.extend(batch_ids)

            # Extract Labels if present
            if not is_test:
                batch_labels = batch["label"].numpy()
                all_labels.extend(batch_labels)

            # Process Views: 'standard', 'global', 'local'
            view_embeddings = []

            for view_name in ["standard", "global", "local"]:
                imgs = batch[view_name].to(Config.DEVICE)  # (B, C, H, W)

                # Feature-Level TTA: Horizontal Flip
                imgs_flipped = torch.flip(imgs, dims=[3])

                # Combine original and flipped for efficient batch processing
                # Input shape becomes (2*B, C, H, W)
                combined_input = torch.cat([imgs, imgs_flipped], dim=0)

                # Forward pass to get embeddings
                features = model(combined_input)  # (2*B, Embedding_Dim)

                # Split back into original and flipped
                feat_orig, feat_flip = torch.split(features, imgs.size(0), dim=0)

                # Average the embeddings
                feat_avg = (feat_orig + feat_flip) / 2.0

                view_embeddings.append(feat_avg)

            # Intra-Stream Early Fusion: Concatenate views
            # Resulting Shape: (B, Embedding_Dim * 3)
            batch_final_emb = torch.cat(view_embeddings, dim=1)

            all_embeddings.append(batch_final_emb.cpu().numpy())

    # Concatenate all batches
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_ids = np.array(all_ids)

    if not is_test:
        final_labels = np.array(all_labels)
    else:
        final_labels = None

    # 5. Save to Cache
    print(f"Saving features to {cache_dir}...")
    np.save(emb_path, final_embeddings)
    np.save(id_path, final_ids)
    if final_labels is not None:
        np.save(lbl_path, final_labels)

    # Cleanup model to free GPU memory for the next stream
    del model
    torch.cuda.empty_cache()

    return final_embeddings, final_labels, final_ids
