import os
import torch
import numpy as np
from library import config
from library import dataset
from library import model_factory


def run_inference(loader, model, device):
    """
    Executes the inference loop over the dataloader using the provided model.
    Extracts features from Global, Standard, and Local views, aggregates them,
    and concatenates them into a single feature vector per image.
    """
    model.eval()

    all_features = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            # Extract batch metadata
            img_ids = batch["id"]
            lbls = batch["label"]  # (B,)

            # Determine current batch size (handles last incomplete batch)
            B = batch["global_view"].size(0)

            # -----------------------------------------------------------------
            # 1. Prepare Inputs
            # -----------------------------------------------------------------
            # Global View: (B, 2, 3, 224, 224) -> (B*2, 3, 224, 224)
            global_in = batch["global_view"].view(-1, 3, 224, 224).to(device)

            # Standard View: (B, 2, 3, 224, 224) -> (B*2, 3, 224, 224)
            standard_in = batch["standard_view"].view(-1, 3, 224, 224).to(device)

            # Local View: (B, 10, 3, 224, 224) -> (B*10, 3, 224, 224)
            local_in = batch["local_view"].view(-1, 3, 224, 224).to(device)

            # -----------------------------------------------------------------
            # 2. Model Forward Pass & Aggregation
            # -----------------------------------------------------------------

            # --- Global View ---
            global_out = model(global_in)
            # Reshape (B*2, D) -> (B, 2, D) -> Mean over views -> (B, D)
            g_s4 = global_out["stage4"].view(B, 2, -1).mean(dim=1)
            g_s3 = global_out["stage3"].view(B, 2, -1).mean(dim=1)

            # --- Standard View ---
            standard_out = model(standard_in)
            s_s4 = standard_out["stage4"].view(B, 2, -1).mean(dim=1)
            s_s3 = standard_out["stage3"].view(B, 2, -1).mean(dim=1)

            # --- Local View ---
            local_out = model(local_in)
            # Reshape (B*10, D) -> (B, 10, D) -> Mean over crops/views -> (B, D)
            l_s4 = local_out["stage4"].view(B, 10, -1).mean(dim=1)
            l_s3 = local_out["stage3"].view(B, 10, -1).mean(dim=1)

            # -----------------------------------------------------------------
            # 3. Concatenation
            # -----------------------------------------------------------------
            # Order: [Global_S4, Global_S3, Standard_S4, Standard_S3, Local_S4, Local_S3]
            # Dimensions: 1536 + 768 + 1536 + 768 + 1536 + 768 = 6912
            feats = torch.cat([g_s4, g_s3, s_s4, s_s3, l_s4, l_s3], dim=1)

            # Store results (move to CPU)
            all_features.append(feats.cpu().numpy())
            all_labels.append(lbls.numpy())
            all_ids.extend(img_ids)

    # Concatenate all batches
    features_arr = np.concatenate(all_features, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    ids_arr = np.array(all_ids)

    return features_arr, labels_arr, ids_arr


def extract_features(split, load_cached_data=True):
    """
    Main entry point for feature extraction.
    Handles caching, data loading, model instantiation, and saving results.

    Args:
        split (str): Dataset split to process ('train', 'val', 'test').
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        tuple: (features, labels, ids) as numpy arrays.
               Note: For 'test' split, labels will be dummy values (-1).
    """
    if split not in config.CACHE_FILES:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of {list(config.CACHE_FILES.keys())}"
        )

    cache_paths = config.CACHE_FILES[split]

    # -------------------------------------------------------------------------
    # 1. Check Cache
    # -------------------------------------------------------------------------
    files_exist = True
    for key, path in cache_paths.items():
        if not os.path.exists(path):
            files_exist = False
            break

    if load_cached_data and files_exist:
        print(f"Loading cached features for '{split}' split...")
        features = np.load(cache_paths["features"])
        ids = np.load(cache_paths["ids"])

        # Load labels if they exist in the cache definition (train/val)
        if "labels" in cache_paths:
            labels = np.load(cache_paths["labels"])
        else:
            labels = None

        return features, labels, ids

    # -------------------------------------------------------------------------
    # 2. Compute Features
    # -------------------------------------------------------------------------
    print(f"Computing features for '{split}' split...")

    # Determine CSV path based on split
    if split == "train":
        csv_path = config.TRAIN_CSV
    elif split == "val":
        csv_path = config.VAL_CSV
    elif split == "test":
        csv_path = config.TEST_CSV
    else:
        raise ValueError("Unknown split configuration.")

    # Initialize Data Loader
    loader = dataset.get_dataloader(
        csv_path=csv_path,
        mode=split,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # Initialize Model
    model = model_factory.get_feature_extractor(device=config.DEVICE)

    # Run Inference
    features, labels, ids = run_inference(loader, model, config.DEVICE)

    # -------------------------------------------------------------------------
    # 3. Save to Cache
    # -------------------------------------------------------------------------
    print(f"Saving features for '{split}' split to {config.WORKING_DIR}...")

    # Ensure working directory exists (redundant if config does it, but safe)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    np.save(cache_paths["features"], features)
    np.save(cache_paths["ids"], ids)

    if "labels" in cache_paths:
        np.save(cache_paths["labels"], labels)

    return features, labels, ids
