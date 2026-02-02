import os
import torch
import numpy as np
import pandas as pd
import cv2
import torch.nn.functional as F

from library.config import Config
from library.models import SegmentationUNet, FractureEncoder, DualStreamRNN
from library.utils import load_scan_volume


def run_inference(load_cached_data=True, limit_count=None):
    """
    Runs the full inference pipeline on the test set.

    Args:
        load_cached_data (bool): If True, attempts to load intermediate features from cache.
                                 If False, re-computes features from raw DICOMs.
        limit_count (int, optional): Limits the number of studies processed (for debugging).
    """
    # 1. Setup
    Config.setup()
    device = torch.device(Config.DEVICE)

    # Ensure cache directory for features exists
    feature_cache_dir = os.path.join(Config.WORKING_DIR, "cache", "test_features")
    os.makedirs(feature_cache_dir, exist_ok=True)

    print(f"Starting Inference. Device: {device}")

    # 2. Load Models
    print("Loading models...")

    # Stage 1: Segmentation U-Net
    model_stage1 = SegmentationUNet().to(device)
    ckpt_s1 = os.path.join(Config.CHECKPOINT_DIR, "stage1_unet.pth")
    if os.path.exists(ckpt_s1):
        model_stage1.load_state_dict(torch.load(ckpt_s1, map_location=device))
    else:
        print(f"Warning: {ckpt_s1} not found. Using random weights.")
    model_stage1.eval()

    # Stage 2: Fracture Encoder
    model_stage2 = FractureEncoder().to(device)
    ckpt_s2 = os.path.join(Config.CHECKPOINT_DIR, "stage2_encoder.pth")
    if os.path.exists(ckpt_s2):
        model_stage2.load_state_dict(torch.load(ckpt_s2, map_location=device))
    else:
        print(f"Warning: {ckpt_s2} not found. Using random weights.")
    model_stage2.eval()

    # Stage 3: Aggregator
    # Global context dim is 512 (from ResNet18 bottleneck)
    model_stage3 = DualStreamRNN(global_context_dim=512).to(device)
    ckpt_s3 = os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth")
    if os.path.exists(ckpt_s3):
        model_stage3.load_state_dict(torch.load(ckpt_s3, map_location=device))
    else:
        print(f"Warning: {ckpt_s3} not found. Using random weights.")
    model_stage3.eval()

    # 3. Load Test Information
    test_csv_path = os.path.join(Config.INPUT_DIR, "test.csv")
    if not os.path.exists(test_csv_path):
        print("test.csv not found. Cannot generate submission.")
        return

    df_test = pd.read_csv(test_csv_path)
    unique_studies = df_test["StudyInstanceUID"].unique()

    if limit_count:
        unique_studies = unique_studies[:limit_count]
        print(f"Limiting inference to {limit_count} studies.")

    print(f"Total unique studies to process: {len(unique_studies)}")

    # Dictionary to store predictions: {uid: {'C1': 0.1, ..., 'patient_overall': 0.5}}
    study_predictions = {}

    # 4. Inference Loop
    for i, uid in enumerate(unique_studies):
        feature_path = os.path.join(feature_cache_dir, f"{uid}.npy")

        # ---------------------------------------------------------
        # Part A: Feature Extraction (Stage 1 & 2)
        # ---------------------------------------------------------
        final_features = None

        # Try loading from cache
        if load_cached_data and os.path.exists(feature_path):
            try:
                final_features = np.load(feature_path)
            except Exception:
                pass  # Corrupt file, recompute

        if final_features is None:
            # Compute features from scratch
            volume = load_scan_volume(
                uid,
                Config.TEST_IMAGES_DIR,
                size=Config.ORIGINAL_SIZE,
                load_cached_data=load_cached_data,
            )

            if volume.shape[0] == 0:
                # Fallback for empty/missing volume
                # Create dummy features to allow pipeline to continue
                # SeqLen=1, Dim=1032
                final_features = np.zeros((1, 1032), dtype=np.float32)
            else:
                num_slices = volume.shape[0]
                batch_size = Config.STAGE1_BATCH_SIZE

                # --- Stage 1 Inference ---
                s1_global_ctx = []
                s1_anat_probs = []
                s1_masks = []

                for b_start in range(0, num_slices, batch_size):
                    b_end = min(b_start + batch_size, num_slices)
                    batch_imgs = volume[b_start:b_end]  # (B, 512, 512)

                    # Resize to 256x256 for Stage 1
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
                    )

                    with torch.no_grad():
                        logits, glob_ctx, anat_prob = model_stage1(batch_tensor)
                        s1_global_ctx.append(glob_ctx.cpu().numpy())
                        s1_anat_probs.append(anat_prob.cpu().numpy())

                        # Get masks for ROI (B, 256, 256)
                        preds = torch.argmax(logits, dim=1)
                        s1_masks.append(preds.cpu().numpy())

                s1_global_ctx = np.concatenate(s1_global_ctx, axis=0)
                s1_anat_probs = np.concatenate(s1_anat_probs, axis=0)
                s1_masks = np.concatenate(s1_masks, axis=0)

                # --- ROI Calculation ---
                rois = []
                default_center = (
                    Config.ORIGINAL_SIZE[1] // 2,
                    Config.ORIGINAL_SIZE[0] // 2,
                )

                for z in range(num_slices):
                    mask = s1_masks[z]
                    binary_mask = (mask > 0).astype(np.uint8)
                    M = cv2.moments(binary_mask)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # Scale from 256 back to 512
                        rois.append((cX * 2, cY * 2))
                    else:
                        if len(rois) > 0:
                            rois.append(rois[-1])
                        else:
                            rois.append(default_center)

                # --- Stage 2 Inference ---
                s2_embeddings = []
                crop_h, crop_w = Config.STAGE2_CROP_SIZE

                for b_start in range(0, num_slices, batch_size):
                    b_end = min(b_start + batch_size, num_slices)
                    batch_inputs = []

                    for z in range(b_start, b_end):
                        # Stack 3 slices
                        stack = []
                        for offset in [-1, 0, 1]:
                            idx = max(0, min(z + offset, num_slices - 1))
                            stack.append(volume[idx])
                        img_stack = np.stack(stack, axis=-1)  # (512, 512, 3)

                        # Upsample mask
                        m = s1_masks[z]
                        m_bin = (m > 0).astype(np.float32)
                        m_up = cv2.resize(
                            m_bin,
                            (Config.ORIGINAL_SIZE[1], Config.ORIGINAL_SIZE[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )

                        full_input = np.dstack([img_stack, m_up])  # (512, 512, 4)

                        # Crop
                        cx, cy = rois[z]
                        x1 = int(cx - crop_w / 2)
                        y1 = int(cy - crop_h / 2)
                        x1 = max(0, min(x1, Config.ORIGINAL_SIZE[1] - crop_w))
                        y1 = max(0, min(y1, Config.ORIGINAL_SIZE[0] - crop_h))

                        crop = full_input[y1 : y1 + crop_h, x1 : x1 + crop_w, :]
                        batch_inputs.append(crop.transpose(2, 0, 1))

                    batch_tensor = torch.tensor(
                        np.array(batch_inputs), dtype=torch.float32
                    ).to(device)

                    with torch.no_grad():
                        embeddings = model_stage2(batch_tensor)
                        s2_embeddings.append(embeddings.cpu().numpy())

                s2_embeddings = np.concatenate(s2_embeddings, axis=0)

                # Concatenate all features
                final_features = np.concatenate(
                    [s2_embeddings, s1_global_ctx, s1_anat_probs], axis=1
                )

                # Save to cache
                np.save(feature_path, final_features.astype(np.float32))

        # ---------------------------------------------------------
        # Part B: Sequence Aggregation (Stage 3)
        # ---------------------------------------------------------
        # Prepare input tensors
        features_tensor = (
            torch.from_numpy(final_features).float().unsqueeze(0)
        )  # (1, SeqLen, 1032)

        # Split features
        local_emb = features_tensor[:, :, : Config.STAGE2_EMBEDDING_DIM].to(device)
        global_ctx = features_tensor[
            :, :, Config.STAGE2_EMBEDDING_DIM : Config.STAGE2_EMBEDDING_DIM + 512
        ].to(device)
        anat_probs = features_tensor[:, :, -8:].to(device)

        with torch.no_grad():
            logits = model_stage3(local_emb, global_ctx, anat_probs)
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # (8,)

        # Store predictions
        # Output order is [C1, C2, C3, C4, C5, C6, C7, Patient]
        study_predictions[uid] = {
            "C1": probs[0],
            "C2": probs[1],
            "C3": probs[2],
            "C4": probs[3],
            "C5": probs[4],
            "C6": probs[5],
            "C7": probs[6],
            "patient_overall": probs[7],
        }

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(unique_studies)} studies.")

    # 5. Generate Submission File
    print("Generating submission file...")
    submission_rows = []

    # Iterate through the requested rows in test.csv
    for _, row in df_test.iterrows():
        row_id = row["row_id"]
        uid = row["StudyInstanceUID"]
        pred_type = row["prediction_type"]  # e.g., "C1", "patient_overall"

        # Default probability if study failed or wasn't processed
        prob = 0.5

        if uid in study_predictions:
            if pred_type in study_predictions[uid]:
                prob = study_predictions[uid][pred_type]
            else:
                # Should not happen if pred_type is standard
                pass

        submission_rows.append({"row_id": row_id, "fractured": prob})

    df_submission = pd.DataFrame(submission_rows)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(df_submission.head())
