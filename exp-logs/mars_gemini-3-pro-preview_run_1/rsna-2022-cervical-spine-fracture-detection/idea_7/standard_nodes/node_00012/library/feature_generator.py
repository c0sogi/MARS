import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2

from library.config import Config
from library.models import SegmentationUNet, FractureEncoder
from library.utils import load_scan_volume


def generate_patient_features(
    splits=["train", "val", "test"],
    load_cached_data=True,
    batch_size=32,
    limit_count=None,
):
    """
    Generates and caches feature sequences for patients in the specified splits.

    The feature vector for each slice t is a concatenation of:
    1. Local Fracture Embedding (512) - From Stage 2 (High-Res Crop)
    2. Global Context Vector (512)  - From Stage 1 (Downsampled Full Slice)
    3. Anatomical Probabilities (8) - From Stage 1

    Total Dimension: 1032

    Args:
        splits (list): List of splits to process ['train', 'val', 'test'].
        load_cached_data (bool): If True, skip patients with existing feature files.
        batch_size (int): Batch size for inference.
        limit_count (int, optional): Limit number of patients per split (for debugging).
    """
    # 1. Setup
    Config.setup()
    device = torch.device(Config.DEVICE)

    feature_dir = os.path.join(Config.WORKING_DIR, "cache", "features")
    os.makedirs(feature_dir, exist_ok=True)

    print(f"Starting Feature Generation. Output dir: {feature_dir}")

    # 2. Load Models
    print("Loading models...")

    # Stage 1: Segmentation U-Net
    model_stage1 = SegmentationUNet().to(device)
    stage1_ckpt = os.path.join(Config.CHECKPOINT_DIR, "stage1_unet.pth")
    if os.path.exists(stage1_ckpt):
        model_stage1.load_state_dict(torch.load(stage1_ckpt, map_location=device))
        print(f"Loaded Stage 1 weights from {stage1_ckpt}")
    else:
        print("Warning: Stage 1 checkpoint not found. Using random weights.")
    model_stage1.eval()

    # Stage 2: Fracture Encoder
    model_stage2 = FractureEncoder().to(device)
    stage2_ckpt = os.path.join(Config.CHECKPOINT_DIR, "stage2_encoder.pth")
    if os.path.exists(stage2_ckpt):
        # The checkpoint saves the encoder state dict directly
        model_stage2.load_state_dict(torch.load(stage2_ckpt, map_location=device))
        print(f"Loaded Stage 2 weights from {stage2_ckpt}")
    else:
        print("Warning: Stage 2 checkpoint not found. Using random weights.")
    model_stage2.eval()

    # 3. Process Splits
    for split in splits:
        print(f"\nProcessing split: {split}")

        # Load Metadata
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = Config.VAL_METADATA_PATH
        elif split == "test":
            meta_path = Config.TEST_METADATA_PATH
        else:
            print(f"Unknown split {split}, skipping.")
            continue

        if not os.path.exists(meta_path):
            print(f"Metadata file {meta_path} not found, skipping.")
            continue

        df = pd.read_csv(meta_path)

        # Get unique studies
        unique_studies = df["StudyInstanceUID"].unique()

        if limit_count:
            unique_studies = unique_studies[:limit_count]
            print(f"Limiting to first {limit_count} studies.")

        print(f"Total studies to process: {len(unique_studies)}")

        for i, uid in enumerate(unique_studies):
            save_path = os.path.join(feature_dir, f"{uid}.npy")

            # Check Cache
            if load_cached_data and os.path.exists(save_path):
                continue

            # Determine image directory
            if split == "test":
                img_dir = Config.TEST_IMAGES_DIR
            else:
                img_dir = Config.TRAIN_IMAGES_DIR

            # Load Volume (Original Size 512x512)
            # Shape: (D, H, W)
            volume = load_scan_volume(
                uid, img_dir, size=Config.ORIGINAL_SIZE, load_cached_data=True
            )

            if volume.shape[0] == 0:
                print(f"Warning: Empty volume for {uid}. Skipping.")
                continue

            num_slices = volume.shape[0]

            # ---------------------------------------------------------
            # Stage 1 Inference
            # ---------------------------------------------------------
            # Resize for Stage 1 (256x256)
            # We process in batches

            stage1_global_ctx = []
            stage1_anat_probs = []
            stage1_masks = []  # We need these for Stage 2 cropping and input

            for b_start in range(0, num_slices, batch_size):
                b_end = min(b_start + batch_size, num_slices)
                batch_imgs = volume[b_start:b_end]  # (B, 512, 512)

                # Resize to 256x256
                batch_resized = []
                for img in batch_imgs:
                    img_r = cv2.resize(
                        img,
                        (Config.STAGE1_IMAGE_SIZE[1], Config.STAGE1_IMAGE_SIZE[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    batch_resized.append(img_r)

                batch_tensor = (
                    torch.tensor(np.array(batch_resized), dtype=torch.float32)
                    .unsqueeze(1)
                    .to(device)
                )  # (B, 1, 256, 256)

                with torch.no_grad():
                    logits, glob_ctx, anat_prob = model_stage1(batch_tensor)

                    # Store features
                    stage1_global_ctx.append(glob_ctx.cpu().numpy())
                    stage1_anat_probs.append(anat_prob.cpu().numpy())

                    # Get mask for Stage 2
                    # Argmax to get class indices, then binarize (any vertebra > 0)
                    preds = torch.argmax(logits, dim=1)  # (B, 256, 256)
                    stage1_masks.append(preds.cpu().numpy())

            stage1_global_ctx = np.concatenate(stage1_global_ctx, axis=0)  # (D, 512)
            stage1_anat_probs = np.concatenate(stage1_anat_probs, axis=0)  # (D, 8)
            stage1_masks = np.concatenate(stage1_masks, axis=0)  # (D, 256, 256)

            # ---------------------------------------------------------
            # Calculate ROIs (Center of Mass)
            # ---------------------------------------------------------
            # We need ROIs relative to the original 512x512 image.
            # Masks are 256x256. Scale factor = 2.

            rois = []  # List of (x, y) centers
            default_center = (
                Config.ORIGINAL_SIZE[1] // 2,
                Config.ORIGINAL_SIZE[0] // 2,
            )

            for z in range(num_slices):
                mask = stage1_masks[z]
                # Binarize: any vertebra
                binary_mask = (mask > 0).astype(np.uint8)

                M = cv2.moments(binary_mask)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Scale to 512
                    rois.append((cX * 2, cY * 2))
                else:
                    # If no spine detected, use previous valid or default
                    if len(rois) > 0:
                        rois.append(rois[-1])
                    else:
                        rois.append(default_center)

            # ---------------------------------------------------------
            # Stage 2 Inference
            # ---------------------------------------------------------
            # Input: 3 slices (z-1, z, z+1) + Mask
            # Crop 256x256 from 512x512 volume

            stage2_embeddings = []

            crop_h, crop_w = Config.STAGE2_CROP_SIZE

            for b_start in range(0, num_slices, batch_size):
                b_end = min(b_start + batch_size, num_slices)

                batch_inputs = []

                for z in range(b_start, b_end):
                    # 1. Construct 3-slice stack
                    stack = []
                    for offset in [-1, 0, 1]:
                        idx = max(0, min(z + offset, num_slices - 1))
                        stack.append(volume[idx])  # 512x512

                    img_stack = np.stack(stack, axis=-1)  # (512, 512, 3)

                    # 2. Get Mask (upsample back to 512 or use 256 and resize? Upsample is better)
                    # Mask is (256, 256).
                    m = stage1_masks[z]
                    m_bin = (m > 0).astype(np.float32)
                    m_up = cv2.resize(
                        m_bin,
                        (Config.ORIGINAL_SIZE[1], Config.ORIGINAL_SIZE[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    # Combine
                    full_input = np.dstack([img_stack, m_up])  # (512, 512, 4)

                    # 3. Crop
                    cx, cy = rois[z]
                    x1 = int(cx - crop_w / 2)
                    y1 = int(cy - crop_h / 2)

                    # Clamp
                    x1 = max(0, min(x1, Config.ORIGINAL_SIZE[1] - crop_w))
                    y1 = max(0, min(y1, Config.ORIGINAL_SIZE[0] - crop_h))

                    crop = full_input[y1 : y1 + crop_h, x1 : x1 + crop_w, :]

                    # Transpose to (C, H, W)
                    batch_inputs.append(crop.transpose(2, 0, 1))

                batch_tensor = torch.tensor(
                    np.array(batch_inputs), dtype=torch.float32
                ).to(device)

                with torch.no_grad():
                    embeddings = model_stage2(batch_tensor)
                    stage2_embeddings.append(embeddings.cpu().numpy())

            stage2_embeddings = np.concatenate(stage2_embeddings, axis=0)  # (D, 512)

            # ---------------------------------------------------------
            # Aggregate and Save
            # ---------------------------------------------------------
            # Concatenate: [Local(512) | Global(512) | Probs(8)]
            final_features = np.concatenate(
                [stage2_embeddings, stage1_global_ctx, stage1_anat_probs], axis=1
            )

            # Save
            np.save(save_path, final_features.astype(np.float32))

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(unique_studies)} studies.")

    print("Feature Generation Completed.")
