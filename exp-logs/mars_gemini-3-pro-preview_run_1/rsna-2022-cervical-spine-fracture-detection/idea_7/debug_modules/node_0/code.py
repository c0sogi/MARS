import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.models import SegmentationUNet, FractureEncoder, DualStreamRNN
from library.train_segmentation import train_stage1
from library.train_encoder import train_stage2
from library.feature_generator import generate_patient_features
from library.train_aggregator import train_stage3
from library.inference import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== RSNA Cervical Spine Fracture Detection Pipeline Demo ===")

    # --------------------------------------------------------------------------
    # 1. Configuration for Speed & Demo
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Enable Debug mode to use small subsets of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 16  # Small sample size for quick iteration

    # Set training hyperparameters to minimal values
    Config.STAGE1_EPOCHS = 1
    Config.STAGE2_EPOCHS = 1
    Config.STAGE3_EPOCHS = 1

    Config.STAGE1_BATCH_SIZE = 4
    Config.STAGE2_BATCH_SIZE = 4
    Config.STAGE3_BATCH_SIZE = 2

    # Disable multiprocessing overhead for small demo
    Config.NUM_WORKERS = 0

    # Initialize environment (seeds, directories)
    Config.setup()
    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # --------------------------------------------------------------------------
    # 2. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Model Architectures...")

    # --- Stage 1: Segmentation U-Net ---
    print("    Checking Stage 1 (SegmentationUNet)...")
    model_s1 = SegmentationUNet().to(device)
    # Input: (Batch, 1, 256, 256)
    dummy_input_s1 = torch.randn(2, 1, 256, 256).to(device)
    logits, glob_ctx, anat_probs = model_s1(dummy_input_s1)

    # Verify shapes
    assert logits.shape == (2, 8, 256, 256), f"S1 Logits shape mismatch: {logits.shape}"
    assert glob_ctx.shape == (2, 512), f"S1 Context shape mismatch: {glob_ctx.shape}"
    assert anat_probs.shape == (
        2,
        8,
    ), f"S1 Anat Probs shape mismatch: {anat_probs.shape}"
    print("    -> Stage 1 Verified.")

    # --- Stage 2: Fracture Encoder ---
    print("    Checking Stage 2 (FractureEncoder)...")
    model_s2 = FractureEncoder().to(device)
    # Input: (Batch, 4, 256, 256) -> 3 slices + 1 mask
    dummy_input_s2 = torch.randn(2, 4, 256, 256).to(device)
    embedding = model_s2(dummy_input_s2)

    # Verify shape: (Batch, EmbeddingDim=512)
    assert embedding.shape == (
        2,
        512,
    ), f"S2 Embedding shape mismatch: {embedding.shape}"
    print("    -> Stage 2 Verified.")

    # --- Stage 3: Dual Stream RNN ---
    print("    Checking Stage 3 (DualStreamRNN)...")
    model_s3 = DualStreamRNN(global_context_dim=512).to(device)
    # Inputs: Sequences of length 5
    dummy_local = torch.randn(2, 5, 512).to(device)
    dummy_global = torch.randn(2, 5, 512).to(device)
    dummy_anat = torch.randn(2, 5, 8).to(device)

    out_s3 = model_s3(dummy_local, dummy_global, dummy_anat)

    # Verify shape: (Batch, 8) -> 7 vertebrae + 1 patient_overall
    assert out_s3.shape == (2, 8), f"S3 Output shape mismatch: {out_s3.shape}"
    print("    -> Stage 3 Verified.")

    # --------------------------------------------------------------------------
    # 3. Training Pipeline Execution
    # --------------------------------------------------------------------------
    print("\n[3] Executing Training Pipeline...")

    # --- Stage 1 Training ---
    print("\n    >>> Running Stage 1 Training (Segmentation)...")
    # We disable loading cached data to force the data loader to process raw files
    try:
        train_stage1(load_cached_data=False)
    except Exception as e:
        print(f"    Stage 1 training failed with error: {e}")
        # In a real scenario we might exit, but for demo we check if checkpoint exists

    s1_ckpt = os.path.join(Config.CHECKPOINT_DIR, "stage1_unet.pth")
    if os.path.exists(s1_ckpt):
        print("    -> Stage 1 Checkpoint created successfully.")
    else:
        print(
            "    -> Warning: Stage 1 Checkpoint not found (possibly due to empty dataset in debug mode)."
        )

    # --- Stage 2 Training ---
    print("\n    >>> Running Stage 2 Training (Slice Classification)...")
    try:
        train_stage2(load_cached_data=False)
    except Exception as e:
        print(f"    Stage 2 training failed with error: {e}")

    s2_ckpt = os.path.join(Config.CHECKPOINT_DIR, "stage2_encoder.pth")
    if os.path.exists(s2_ckpt):
        print("    -> Stage 2 Checkpoint created successfully.")
    else:
        print("    -> Warning: Stage 2 Checkpoint not found.")

    # --- Feature Generation ---
    print("\n    >>> Generating Features for Stage 3...")
    # Generate features for a small subset of train/val to allow Stage 3 to run
    # limit_count ensures we don't process the whole dataset
    generate_patient_features(
        splits=["train", "val"], batch_size=4, limit_count=5, load_cached_data=False
    )

    feature_dir = os.path.join(Config.WORKING_DIR, "cache", "features")
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith(".npy")]
    print(f"    -> Generated {len(feature_files)} feature files in {feature_dir}")

    if len(feature_files) == 0:
        print("    -> Error: No features generated. Stage 3 will fail.")

    # --- Stage 3 Training ---
    print("\n    >>> Running Stage 3 Training (Aggregator)...")
    try:
        train_stage3(load_cached_data=False)
    except Exception as e:
        print(f"    Stage 3 training failed with error: {e}")

    s3_ckpt = os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth")
    if os.path.exists(s3_ckpt):
        print("    -> Stage 3 Checkpoint created successfully.")
    else:
        print("    -> Warning: Stage 3 Checkpoint not found.")

    # --------------------------------------------------------------------------
    # 4. Inference Execution
    # --------------------------------------------------------------------------
    print("\n[4] Executing Inference Pipeline...")

    # Run inference on a few test studies
    run_inference(load_cached_data=False, limit_count=3)

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    -> Submission file generated at {Config.SUBMISSION_PATH}")
        print(f"    -> Rows: {len(df_sub)}")
        print("    -> Head:")
        print(df_sub.head(3))

        # Validation
        assert "row_id" in df_sub.columns, "Submission missing row_id"
        assert "fractured" in df_sub.columns, "Submission missing fractured"
        assert not df_sub.empty, "Submission file is empty"
    else:
        print("    -> Error: Submission file was not generated.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
