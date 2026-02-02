import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import load_data_to_memory, CactusDataset, get_transforms
from library.model import MetadataFusedRepVGG
from library.utils import seed_everything


def run_inference():
    """
    Executes the inference pipeline:
    1. Loads training stats for normalization.
    2. Loads test data.
    3. Loads the best/SWA model and fuses it for deployment.
    4. Performs 4-view Test Time Augmentation (TTA).
    5. Saves the submission file.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Initializing inference on {device}...")

    # 2. Load Training Stats for Metadata Normalization
    # We need to normalize test file sizes using the training set's mean and std
    print("Loading training metadata to compute normalization statistics...")
    _, _, train_filesizes, _ = load_data_to_memory(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_IMGS,
        Config.CACHE_TRAIN_FILESIZES,
        Config.CACHE_TRAIN_LABELS,
        load_cached_data=True,
    )

    fs_mean = train_filesizes.mean()
    fs_std = train_filesizes.std()
    print(f"Training Filesize Stats - Mean: {fs_mean:.4f}, Std: {fs_std:.4f}")

    # 3. Load Test Data
    print("Loading test data into memory...")
    test_imgs, _, test_filesizes, test_ids = load_data_to_memory(
        Config.TEST_METADATA_PATH,
        Config.CACHE_TEST_IMGS,
        Config.CACHE_TEST_FILESIZES,
        cache_id_path=Config.CACHE_TEST_IDS,
        load_cached_data=True,
    )

    # 4. Create Dataset and Loader
    # Use 'val' transforms (ToTensorV2 only) as we handle TTA manually or via batch manipulation
    test_dataset = CactusDataset(
        images=test_imgs,
        filesizes=test_filesizes,
        labels=None,  # No labels needed for inference
        transform=get_transforms(mode="val"),
        filesize_mean=fs_mean,
        filesize_std=fs_std,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Load and Prepare Model
    print("Initializing model architecture...")
    model = MetadataFusedRepVGG(num_classes=Config.NUM_CLASSES, deploy=False)

    # Determine which weights to load
    if os.path.exists(Config.FINAL_SWA_MODEL_PATH):
        checkpoint_path = Config.FINAL_SWA_MODEL_PATH
        print(f"Loading SWA model from {checkpoint_path}")
    else:
        checkpoint_path = Config.BEST_MODEL_PATH
        print(f"SWA model not found. Loading Best model from {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)

    # Handle potential 'module.' prefix from AveragedModel or DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Structural Re-parameterization (Fusion)
    print("Switching model to deploy mode (fusing RepVGG blocks)...")
    model.switch_to_deploy()

    # 6. Inference Loop with TTA
    print("Starting inference with 4-view Test Time Augmentation...")
    all_preds = []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            # inputs: ((img_tensor, meta_tensor), dummy_label)
            imgs, metas = inputs
            imgs = imgs.to(device)
            metas = metas.to(device)

            # --- TTA Strategy ---
            # 1. Original
            imgs_orig = imgs
            # 2. Horizontal Flip
            imgs_h = torch.flip(imgs, [3])
            # 3. Vertical Flip
            imgs_v = torch.flip(imgs, [2])
            # 4. Rotate 180 (Horizontal + Vertical Flip)
            imgs_hv = torch.flip(imgs, [2, 3])

            # Stack all views into a single batch: (4*B, C, H, W)
            tta_imgs = torch.cat([imgs_orig, imgs_h, imgs_v, imgs_hv], dim=0)

            # Repeat metadata to match the new batch size: (4*B)
            tta_metas = metas.repeat(4)

            # Forward pass
            logits = model((tta_imgs, tta_metas))
            probs = torch.sigmoid(logits)  # Shape: (4*B, 1)

            # Reshape to separate views: (4, B, 1)
            # The batch was concatenated as [Orig, H, V, HV], so we split into 4 chunks
            batch_size = imgs.size(0)
            probs_views = probs.view(4, batch_size, -1)

            # Average predictions across the 4 views
            avg_probs = probs_views.mean(dim=0)  # Shape: (B, 1)

            # Store results
            all_preds.extend(avg_probs.cpu().numpy().flatten())

    # 7. Generate Submission File
    print(f"Generating submission file at {Config.SUBMISSION_PATH}...")

    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": all_preds})

    # Ensure output directory exists (handled by Config, but good practice)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved. Total predictions: {len(submission_df)}")
