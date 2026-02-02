import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config, seed_everything
from library.models import UNetLocalizer, DualStreamEncoder, AnatomicalGRU
from library.data import (
    get_segmentation_dataloader,
    get_slice_classification_dataloader,
    get_sequence_dataloader,
    prepare_segmentation_cache,
    prepare_slice_classification_metadata,
)
from library.training import (
    train_stage1_localizer,
    train_stage2_encoder,
    train_stage3_aggregator,
    extract_features_and_cache,
)
from library.inference import InferencePipeline


def run_demo():
    print("Starting RSNA Cervical Spine Fracture Detection Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    print("\n[1/6] Configuring environment...")
    seed_everything(42)

    # Override Config for rapid execution
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5  # Process only 5 studies
    Config.STAGE1_EPOCHS = 1
    Config.STAGE2_EPOCHS = 1
    Config.STAGE3_EPOCHS = 1
    Config.STAGE1_BATCH_SIZE = 2
    Config.STAGE2_BATCH_SIZE = 2
    Config.STAGE3_BATCH_SIZE = 2
    Config.NUM_WORKERS = 2

    # Ensure working directory is clean-ish (optional, but good for demo)
    if os.path.exists(Config.CACHE_DIR):
        # We don't delete it to assume some cache might exist,
        # but for a pure run we might want to.
        # Given constraints, we leave it be, code handles existing cache.
        pass

    # -------------------------------------------------------------------------
    # 2. Model Logic Verification
    # -------------------------------------------------------------------------
    print("\n[2/6] Verifying Model Architectures...")
    device = torch.device("cpu")  # Use CPU for quick shape checks

    # A. UNetLocalizer
    print("  - Testing UNetLocalizer...")
    model_s1 = UNetLocalizer(num_classes=8, pretrained=False).to(device)
    dummy_input_s1 = torch.randn(2, 1, 256, 256).to(device)
    with torch.no_grad():
        out_s1 = model_s1(dummy_input_s1)
    assert out_s1.shape == (2, 8, 256, 256), f"Stage 1 Output Mismatch: {out_s1.shape}"

    # B. DualStreamEncoder
    print("  - Testing DualStreamEncoder...")
    model_s2 = DualStreamEncoder(pretrained=False).to(device)
    # Local branch takes 2 channels (Image + Mask), Global takes 1
    dummy_local = torch.randn(2, 2, 256, 256).to(device)
    dummy_global = torch.randn(2, 1, 256, 256).to(device)
    with torch.no_grad():
        out_s2 = model_s2(dummy_local, dummy_global)
    # Output is fused features (512+512=1024)
    assert out_s2.shape == (2, 1024), f"Stage 2 Output Mismatch: {out_s2.shape}"

    # C. AnatomicalGRU
    print("  - Testing AnatomicalGRU...")
    # Input dim = 1024 (visual) + 8 (anatomical) = 1032
    model_s3 = AnatomicalGRU(input_dim=1032).to(device)
    dummy_seq = torch.randn(2, 10, 1032).to(device)  # Batch=2, SeqLen=10
    with torch.no_grad():
        out_s3 = model_s3(dummy_seq)
    # Output: 7 vertebrae + 1 patient = 8 classes
    assert out_s3.shape == (2, 8), f"Stage 3 Output Mismatch: {out_s3.shape}"

    print("  -> All models verified.")

    # -------------------------------------------------------------------------
    # 3. Data Loader Verification
    # -------------------------------------------------------------------------
    print("\n[3/6] Verifying Data Loaders...")

    # A. Segmentation Loader
    # Note: This requires .nii files in input/segmentations.
    # If none exist for the debug subset, this might return None or empty.
    # The provided dataset info shows some .nii files exist.
    print("  - Testing Segmentation DataLoader...")
    loader_s1 = get_segmentation_dataloader(batch_size=2, split="train")
    if loader_s1 is not None and len(loader_s1) > 0:
        batch_imgs, batch_masks = next(iter(loader_s1))
        assert batch_imgs.shape == (
            2,
            1,
            256,
            256,
        ), f"Seg Image shape error: {batch_imgs.shape}"
        assert batch_masks.shape == (
            2,
            256,
            256,
        ), f"Seg Mask shape error: {batch_masks.shape}"
    else:
        print(
            "    (Skipping assertion: No segmentation data available in debug subset)"
        )

    # B. Slice Classification Loader
    print("  - Testing Slice Classification DataLoader...")
    loader_s2 = get_slice_classification_dataloader(batch_size=2, split="train")
    if loader_s2 is not None and len(loader_s2) > 0:
        batch_data = next(iter(loader_s2))
        assert batch_data["local"].shape == (2, 1, 256, 256)
        assert batch_data["global"].shape == (2, 1, 256, 256)
        assert batch_data["label"].shape == (2,)
    else:
        print(
            "    (Skipping assertion: No slice classification data available in debug subset)"
        )

    # -------------------------------------------------------------------------
    # 4. Training Pipeline Simulation
    # -------------------------------------------------------------------------
    print("\n[4/6] Simulating Training Pipeline...")

    # A. Train Stage 1
    print("  - Training Stage 1 (Localizer)...")
    model_s1_trained = train_stage1_localizer(debug=True)
    assert os.path.exists(Config.STAGE1_CHECKPOINT_PATH), "Stage 1 Checkpoint missing"

    # B. Train Stage 2
    print("  - Training Stage 2 (Encoder)...")
    model_s2_trained = train_stage2_encoder(debug=True)
    assert os.path.exists(Config.STAGE2_CHECKPOINT_PATH), "Stage 2 Checkpoint missing"

    # C. Extract Features (Crucial for Stage 3)
    print("  - Extracting Features for Stage 3...")
    # We extract for 'train' to allow Stage 3 training to run
    extract_features_and_cache(
        model_s1_trained, model_s2_trained, split="train", debug=True
    )
    # Verify cache creation
    feature_dir = os.path.join(Config.CACHE_DIR, "features")
    assert os.path.exists(feature_dir), "Feature directory not created"
    # Check if at least one feature file exists (if data was available)
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith(".npy")]
    if len(feature_files) == 0:
        print(
            "    Warning: No feature files created. Stage 3 might fail or use dummy data."
        )
        # Create a dummy feature file to ensure Stage 3 runs for demo purposes
        # The library's FeatureSequenceDataset has a fallback, but let's be safe regarding dimensions.
        # The fallback in library/data.py creates (100, 512), but model expects 1032.
        # We must manually fix this for the demo if extraction yielded nothing.
        dummy_uid = "1.2.826.0.1.3680043.10051"  # Example from metadata
        dummy_feat = np.zeros((10, 1032), dtype=np.float32)
        np.save(os.path.join(feature_dir, f"{dummy_uid}.npy"), dummy_feat)

    # D. Train Stage 3
    print("  - Training Stage 3 (Aggregator)...")
    # Ensure we have data for the loader
    model_s3_trained = train_stage3_aggregator(debug=True)
    assert os.path.exists(Config.STAGE3_CHECKPOINT_PATH), "Stage 3 Checkpoint missing"

    # -------------------------------------------------------------------------
    # 5. Inference Pipeline Simulation
    # -------------------------------------------------------------------------
    print("\n[5/6] Simulating Inference Pipeline...")

    pipeline = InferencePipeline()
    # Run pipeline (includes feature extraction for test set and prediction)
    pipeline.run(debug=True)

    # -------------------------------------------------------------------------
    # 6. Final Validation
    # -------------------------------------------------------------------------
    print("\n[6/6] Validating Submission...")

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"  Submission generated with {len(df_sub)} rows.")
        print("  First 5 rows:")
        print(df_sub.head())

        # Check columns
        assert "row_id" in df_sub.columns
        assert "fractured" in df_sub.columns

        # Check probability range
        probs = df_sub["fractured"].values
        assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"

        print("  Submission format verified.")
    else:
        # If no predictions were generated (e.g. empty test set in debug), raise error
        raise FileNotFoundError("Submission file was not generated.")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
