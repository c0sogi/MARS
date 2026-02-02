import os
import torch
import numpy as np
from library.config import Config
from library.modeling import load_feature_extractor
from library.data import get_data_loaders
from library.utils import seed_everything


def extract_features_for_stream(dataloader, model, device):
    """
    Iterates through a DataLoader, extracting features for each image using the
    Multi-View TTA strategy.

    For each image:
    1. Processes 3 views: Global, Standard, Local.
    2. Applies TTA: Averages embeddings of original and horizontally flipped images.
    3. Concatenates embeddings from all 3 views.

    Args:
        dataloader (DataLoader): PyTorch DataLoader.
        model (nn.Module): The feature extractor model (eval mode).
        device (str): Computation device.

    Returns:
        tuple: (embeddings, labels, ids)
            - embeddings (np.ndarray): Shape (N, D_total)
            - labels (np.ndarray): Shape (N,)
            - ids (np.ndarray): Shape (N,)
    """
    model.eval()

    all_embeddings = []
    all_labels = []
    all_ids = []

    # Views to process in order
    view_names = [
        view["name"] for view in Config.VIEWS
    ]  # ['global', 'standard', 'local']

    with torch.no_grad():
        for batch in dataloader:
            # batch contains keys: 'id', 'label', 'global', 'standard', 'local'

            batch_ids = batch["id"]
            batch_labels = batch["label"].numpy()

            # List to store the averaged embedding for each view
            view_features = []

            for view_name in view_names:
                # Get images for this view and move to device
                images = batch[view_name].to(device)

                # Create horizontally flipped version for TTA
                # Images are (B, C, H, W), flip on last dim (width)
                images_flipped = torch.flip(images, dims=[3])

                # Extract features for original and flipped
                feats_orig = model(images)
                feats_flip = model(images_flipped)

                # Average the embeddings (TTA)
                feats_avg = (feats_orig + feats_flip) / 2.0

                view_features.append(feats_avg)

            # Intra-Stream Early Fusion: Concatenate views along feature dimension
            # Shape: (B, D_global + D_standard + D_local)
            fused_features = torch.cat(view_features, dim=1)

            # Move to CPU and store
            all_embeddings.append(fused_features.cpu().numpy())
            all_labels.append(batch_labels)
            all_ids.extend(batch_ids)

    # Aggregate results
    final_embeddings = np.vstack(all_embeddings)
    final_labels = np.concatenate(all_labels)
    final_ids = np.array(all_ids)

    return final_embeddings, final_labels, final_ids


def process_stream(stream_name, load_cached_data=True):
    """
    Orchestrates the feature extraction for a specific stream (Train, Val, Test).
    Handles caching of embeddings and labels/ids.

    Args:
        stream_name (str): 'stream_a' or 'stream_b'.
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        dict: Contains 'train', 'val', 'test' keys, each mapping to (embeddings, labels/ids).
              Structure:
              {
                  'train': (X_train, y_train),
                  'val': (X_val, y_val),
                  'test': (X_test, test_ids)
              }
    """
    # 1. Determine Cache Paths based on Stream
    if stream_name == "stream_a":
        model_name = Config.MODEL_A_NAME
        path_train_emb = Config.STREAM_A_TRAIN_EMB
        path_val_emb = Config.STREAM_A_VAL_EMB
        path_test_emb = Config.STREAM_A_TEST_EMB
    elif stream_name == "stream_b":
        model_name = Config.MODEL_B_NAME
        path_train_emb = Config.STREAM_B_TRAIN_EMB
        path_val_emb = Config.STREAM_B_VAL_EMB
        path_test_emb = Config.STREAM_B_TEST_EMB
    else:
        raise ValueError(f"Unknown stream name: {stream_name}")

    # Shared paths for labels/IDs
    path_train_lbl = Config.TRAIN_LABELS_CACHE
    path_val_lbl = Config.VAL_LABELS_CACHE
    path_test_ids = Config.TEST_IDS_CACHE

    # 2. Check Cache
    # We check if all embedding files exist. Labels/IDs are checked implicitly or generated if missing.
    cache_exists = (
        os.path.exists(path_train_emb)
        and os.path.exists(path_val_emb)
        and os.path.exists(path_test_emb)
        and os.path.exists(path_train_lbl)
        and os.path.exists(path_val_lbl)
        and os.path.exists(path_test_ids)
    )

    if load_cached_data and cache_exists:
        print(f"Loading cached features for {stream_name}...")
        try:
            train_emb = np.load(path_train_emb)
            val_emb = np.load(path_val_emb)
            test_emb = np.load(path_test_emb)

            train_lbl = np.load(path_train_lbl)
            val_lbl = np.load(path_val_lbl)
            test_ids = np.load(path_test_ids)

            return {
                "train": (train_emb, train_lbl),
                "val": (val_emb, val_lbl),
                "test": (test_emb, test_ids),
            }
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 3. Compute Features
    print(f"Starting feature extraction for {stream_name} using {model_name}...")
    seed_everything()

    # Ensure output directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load Model
    print(f"  Loading model: {model_name}...")
    model = load_feature_extractor(model_name, Config.DEVICE)

    # Get DataLoaders
    print(f"  Creating DataLoaders...")
    train_loader, val_loader, test_loader, _ = get_data_loaders(stream_name)

    # Extract Train
    print(f"  Extracting Training features...")
    train_emb, train_lbl, _ = extract_features_for_stream(
        train_loader, model, Config.DEVICE
    )

    # Extract Val
    print(f"  Extracting Validation features...")
    val_emb, val_lbl, _ = extract_features_for_stream(val_loader, model, Config.DEVICE)

    # Extract Test
    print(f"  Extracting Test features...")
    test_emb, _, test_ids = extract_features_for_stream(
        test_loader, model, Config.DEVICE
    )

    # 4. Save to Cache
    print(f"  Saving features to {Config.WORKING_DIR}...")
    np.save(path_train_emb, train_emb)
    np.save(path_val_emb, val_emb)
    np.save(path_test_emb, test_emb)

    # Save shared labels/ids if they don't exist (or overwrite to ensure consistency)
    np.save(path_train_lbl, train_lbl)
    np.save(path_val_lbl, val_lbl)
    np.save(path_test_ids, test_ids)

    # Clean up
    del model
    torch.cuda.empty_cache()

    print(f"Feature extraction for {stream_name} complete.")

    return {
        "train": (train_emb, train_lbl),
        "val": (val_emb, val_lbl),
        "test": (test_emb, test_ids),
    }
