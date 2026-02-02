import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import library.config as config
import library.dataset as lib_dataset
import library.model_utils as lib_model


def extract_view_features(
    model, metadata_path, transform_type, class_to_idx=None, is_test=False, debug=False
):
    """
    Extracts Stage 3 and Stage 4 features for a specific view (transform_type).
    Applies Horizontal Flip TTA.
    Handles 5-crop aggregation for 'local' view.
    """

    # Initialize Dataset
    dataset = lib_dataset.DogDataset(
        metadata_path=metadata_path,
        transform_type=transform_type,
        class_to_idx=class_to_idx,
        is_test=is_test,
        debug=debug,
    )

    # Initialize DataLoader
    # Must be sequential (shuffle=False) to align with other views
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    s3_features_list = []
    s4_features_list = []
    targets_list = []

    # Determine if we need to handle 5-crop logic
    is_local = transform_type == "local"

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Minimal logging
            if i % 20 == 0:
                print(f"    Processing batch {i}/{len(dataloader)}")

            images, targets = batch

            # Move to device
            # images shape:
            #   Global/Standard: (B, 3, H, W)
            #   Local: (B, 5, 3, H, W)
            images = images.to(config.DEVICE)

            # Handle Input Shapes and Flipping
            if is_local:
                b, n_crops, c, h, w = images.shape
                # Flatten crops into batch dimension
                images_reshaped = images.view(-1, c, h, w)  # (B*5, 3, H, W)

                # Create Flipped Version
                images_flipped = torch.flip(images_reshaped, dims=[-1])

                # Forward Pass (Original)
                out_orig = model(images_reshaped)
                # Forward Pass (Flipped)
                out_flip = model(images_flipped)

                # Average TTA
                feat_s3_tta = (out_orig["stage3"] + out_flip["stage3"]) / 2.0
                feat_s4_tta = (out_orig["stage4"] + out_flip["stage4"]) / 2.0

                # Reshape back to (B, 5, D) and Average Crops
                # Stage 3
                feat_s3_tta = feat_s3_tta.view(b, n_crops, -1)
                feat_s3_final = feat_s3_tta.mean(dim=1)  # (B, D)

                # Stage 4
                feat_s4_tta = feat_s4_tta.view(b, n_crops, -1)
                feat_s4_final = feat_s4_tta.mean(dim=1)  # (B, D)

            else:
                # Global / Standard
                images_flipped = torch.flip(images, dims=[-1])

                out_orig = model(images)
                out_flip = model(images_flipped)

                feat_s3_final = (out_orig["stage3"] + out_flip["stage3"]) / 2.0
                feat_s4_final = (out_orig["stage4"] + out_flip["stage4"]) / 2.0

            # Collect
            s3_features_list.append(feat_s3_final.cpu().numpy())
            s4_features_list.append(feat_s4_final.cpu().numpy())

            if is_test:
                # targets are IDs (strings), convert to list
                targets_list.extend(targets)
            else:
                # targets are labels (tensors)
                targets_list.append(targets.numpy())

    # Concatenate batches
    s3_all = np.concatenate(s3_features_list, axis=0)
    s4_all = np.concatenate(s4_features_list, axis=0)

    if is_test:
        targets_all = np.array(targets_list)
    else:
        targets_all = np.concatenate(targets_list, axis=0)

    return s3_all, s4_all, targets_all, dataset.get_class_to_idx()


def get_dataset_embeddings(dataset_type, load_cached_data=True, debug=False):
    """
    Main entry point to get embeddings for a specific dataset split.
    Manages caching, multi-view extraction, and concatenation.

    Args:
        dataset_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from disk.
        debug (bool): Whether to run in debug mode (subset).

    Returns:
        tuple: (embeddings, targets)
               embeddings shape: (N, D_total)
               targets shape: (N,) - labels for train/val, ids for test
    """

    # Define paths based on split
    if dataset_type == "train":
        metadata_path = config.TRAIN_METADATA_PATH
        emb_path = config.TRAIN_EMBEDDINGS_PATH
        lbl_path = config.TRAIN_LABELS_PATH
        is_test = False
    elif dataset_type == "val":
        metadata_path = config.VAL_METADATA_PATH
        emb_path = config.VAL_EMBEDDINGS_PATH
        lbl_path = config.VAL_LABELS_PATH
        is_test = False
    elif dataset_type == "test":
        metadata_path = config.TEST_METADATA_PATH
        emb_path = config.TEST_EMBEDDINGS_PATH
        lbl_path = config.TEST_IDS_PATH
        is_test = True
    else:
        raise ValueError("dataset_type must be 'train', 'val', or 'test'")

    # Adjust paths for debug mode to avoid overwriting full cache
    if debug:
        base, ext = os.path.splitext(emb_path)
        emb_path = f"{base}_debug{ext}"
        base, ext = os.path.splitext(lbl_path)
        lbl_path = f"{base}_debug{ext}"

    # Check Cache
    if load_cached_data and os.path.exists(emb_path) and os.path.exists(lbl_path):
        print(f"Loading cached {dataset_type} embeddings from {emb_path}...")
        embeddings = np.load(emb_path)
        targets = np.load(
            lbl_path, allow_pickle=True
        )  # allow_pickle needed for string IDs in test
        return embeddings, targets

    print(f"Generating {dataset_type} embeddings (Debug={debug})...")

    # Load Model
    model = lib_model.build_feature_extractor()

    # Prepare Class Mapping
    # For validation, we need to ensure we use the same mapping as training
    class_to_idx = None
    if dataset_type == "val":
        print("Loading training metadata to establish class mapping...")
        # Instantiate a temporary train dataset just to get the class mapping
        temp_train = lib_dataset.DogDataset(
            config.TRAIN_METADATA_PATH, transform_type="standard"
        )
        class_to_idx = temp_train.get_class_to_idx()

    # --- View 1: Global ---
    print(f"Processing View: Global...")
    g_s3, g_s4, targets, _ = extract_view_features(
        model, metadata_path, "global", class_to_idx, is_test, debug
    )

    # --- View 2: Standard ---
    print(f"Processing View: Standard...")
    s_s3, s_s4, targets_check, _ = extract_view_features(
        model, metadata_path, "standard", class_to_idx, is_test, debug
    )
    # Sanity check alignment
    if not np.array_equal(targets, targets_check):
        raise RuntimeError("Mismatch in targets between Global and Standard views!")

    # --- View 3: Local ---
    print(f"Processing View: Local...")
    l_s3, l_s4, targets_check, _ = extract_view_features(
        model, metadata_path, "local", class_to_idx, is_test, debug
    )
    if not np.array_equal(targets, targets_check):
        raise RuntimeError("Mismatch in targets between Standard and Local views!")

    # --- Concatenation ---
    # Order: [Global_S4, Global_S3, Standard_S4, Standard_S3, Local_S4, Local_S3]
    print("Concatenating features...")
    embeddings = np.concatenate([g_s4, g_s3, s_s4, s_s3, l_s4, l_s3], axis=1)

    # --- Save to Cache ---
    print(f"Saving to {emb_path}...")
    np.save(emb_path, embeddings)
    np.save(lbl_path, targets)

    return embeddings, targets
