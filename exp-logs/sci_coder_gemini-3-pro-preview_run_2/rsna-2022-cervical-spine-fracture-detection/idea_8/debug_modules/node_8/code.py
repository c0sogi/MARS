import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data import get_loaders, get_test_loader, RSNADataset
from library.model import DualAttentionNetwork
from library.loss import TriLevelLoss
from library.engine import Trainer, inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting RSNA Cervical Spine Fracture Detection Demo ===")

    # 1. Setup Demo Configuration
    # We subclass Config to override settings for a fast, lightweight demonstration.
    class DemoConfig(Config):
        # Paths
        WORKING_DIR = "./working/demo_execution"
        OUTPUT_DIR = os.path.join(WORKING_DIR, "output")
        CACHE_DIR = WORKING_DIR

        TRAIN_METADATA = os.path.join(WORKING_DIR, "train_metadata.csv")
        VAL_METADATA = os.path.join(WORKING_DIR, "val_metadata.csv")
        TEST_METADATA = os.path.join(WORKING_DIR, "test_metadata.csv")
        SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

        # Model & Data Params (Reduced for Speed)
        IMAGE_SIZE = (128, 128)  # Smaller images
        SEQ_LEN = 8  # Shorter sequence
        BACKBONE = "tf_efficientnet_b0_ns"  # Smaller backbone
        BATCH_SIZE = 2
        NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny demo

        # Training Params
        EPOCHS = 1
        ACCUMULATION_STEPS = 1
        LEARNING_RATE = 1e-4

        # Ensure directories exist
        os.makedirs(WORKING_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = DemoConfig()
    seed_everything(config.SEED)
    print(f"Configuration loaded. Working dir: {config.WORKING_DIR}")

    # Apply configuration overrides globally so imported modules (like data.py) see them.
    # Cite debug_lesson_2: Prevent Stale Configuration by updating the global class used by other modules.
    Config.TRAIN_METADATA = config.TRAIN_METADATA
    Config.VAL_METADATA = config.VAL_METADATA
    Config.TEST_METADATA = config.TEST_METADATA
    Config.WORKING_DIR = config.WORKING_DIR
    Config.IMAGE_SIZE = config.IMAGE_SIZE
    Config.SEQ_LEN = config.SEQ_LEN
    Config.BACKBONE = config.BACKBONE
    Config.EPOCHS = config.EPOCHS
    Config.BATCH_SIZE = config.BATCH_SIZE
    Config.NUM_WORKERS = config.NUM_WORKERS
    Config.PREFETCH_FACTOR = (
        None if config.NUM_WORKERS == 0 else getattr(config, "PREFETCH_FACTOR", 2)
    )
    Config.CACHE_DIR = config.CACHE_DIR

    # 2. Prepare Subset Data
    # We create small metadata files pointing to actual images to test the loading pipeline.
    print("\n[Step 1] Preparing Subset Metadata...")

    # Load original metadata
    orig_train = pd.read_csv(Config.TRAIN_METADATA)
    orig_test = pd.read_csv(Config.TEST_METADATA)

    # Select a few studies that actually exist on disk
    # We check the first few to find valid ones
    valid_train_studies = []
    for uid in orig_train["StudyInstanceUID"].unique():
        path = os.path.join(Config.TRAIN_IMAGES_DIR, uid)
        if os.path.exists(path) and len(os.listdir(path)) > 0:
            valid_train_studies.append(uid)
            if len(valid_train_studies) >= 4:
                break

    if len(valid_train_studies) < 1:
        raise RuntimeError("Not enough valid training data found for demo.")

    # Split into demo train/val
    mid_point = len(valid_train_studies) // 2
    if mid_point == 0 and len(valid_train_studies) > 0:
        mid_point = 1

    demo_train_df = orig_train[
        orig_train["StudyInstanceUID"].isin(valid_train_studies[:mid_point])
    ].copy()
    demo_val_df = orig_train[
        orig_train["StudyInstanceUID"].isin(valid_train_studies[mid_point:])
    ].copy()

    # Save to working dir
    demo_train_df.to_csv(config.TRAIN_METADATA, index=False)
    demo_val_df.to_csv(config.VAL_METADATA, index=False)

    # Prepare demo test data
    valid_test_studies = []
    for uid in orig_test["StudyInstanceUID"].unique():
        path = os.path.join(Config.TEST_IMAGES_DIR, uid)
        if os.path.exists(path):
            valid_test_studies.append(uid)
            if len(valid_test_studies) >= 2:
                break

    demo_test_df = orig_test[
        orig_test["StudyInstanceUID"].isin(valid_test_studies)
    ].copy()
    demo_test_df.to_csv(config.TEST_METADATA, index=False)

    print(f"  Train subset: {len(demo_train_df)} rows")
    print(f"  Val subset:   {len(demo_val_df)} rows")
    print(f"  Test subset:  {len(demo_test_df)} rows")

    # 3. Test Data Loading
    print("\n[Step 2] Verifying Data Loading...")
    train_loader, val_loader = get_loaders(
        train_meta_path=config.TRAIN_METADATA,
        val_meta_path=config.VAL_METADATA,
        bbox_path=config.BOUNDING_BOXES,
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    images = batch["image"]
    targets_study = batch["label_study"]
    targets_slice = batch["label_slice"]
    targets_spatial = batch["label_spatial"]

    print(f"  Batch Shapes:")
    print(f"    Images: {images.shape} (Expected: B, Seq, 3, H, W)")
    print(f"    Study Targets: {targets_study.shape} (Expected: B, 8)")
    print(f"    Slice Targets: {targets_slice.shape} (Expected: B, Seq)")
    print(f"    Spatial Targets: {targets_spatial.shape} (Expected: B, Seq, 1, H, W)")

    # Assertions
    assert images.ndim == 5, "Images should be 5D tensor (B, S, C, H, W)"
    assert (
        images.shape[1] == config.SEQ_LEN
    ), f"Sequence length mismatch. Got {images.shape[1]}, expected {config.SEQ_LEN}"
    assert images.shape[3] == config.IMAGE_SIZE[0], "Image height mismatch"
    assert targets_study.shape[1] == 8, "Study targets should have 8 classes"

    # 4. Test Model Forward Pass
    print("\n[Step 3] Verifying Model Architecture...")
    model = DualAttentionNetwork(config)
    model.to(config.DEVICE)

    # Move batch to device
    images = images.to(config.DEVICE)

    # Forward
    outputs = model(images)

    study_logits = outputs["study_logits"]
    slice_logits = outputs["slice_logits"]
    spatial_logits = outputs["spatial_logits"]

    print(f"  Output Shapes:")
    print(f"    Study Logits: {study_logits.shape}")
    print(f"    Slice Logits: {slice_logits.shape}")
    print(f"    Spatial Logits: {spatial_logits.shape}")

    assert study_logits.shape == (images.shape[0], 8)
    assert slice_logits.shape == (images.shape[0], config.SEQ_LEN)
    # Spatial logits are (B, S, 1, H_feat, W_feat). H_feat/W_feat depend on backbone stride (usually /16 or /32)
    assert spatial_logits.ndim == 5

    # 5. Test Loss Calculation
    print("\n[Step 4] Verifying Loss Function...")
    criterion = TriLevelLoss(config)
    criterion.to(config.DEVICE)

    # Prepare targets dict on device
    targets_dict = {
        "label_study": targets_study.to(config.DEVICE),
        "label_slice": targets_slice.to(config.DEVICE),
        "label_spatial": targets_spatial.to(config.DEVICE),
    }

    loss = criterion(outputs, targets_dict)
    print(f"  Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # 6. Test Training Loop (Engine)
    print("\n[Step 5] Running Training Loop (Trainer.fit)...")
    # We need to monkey-patch the get_loaders call inside Trainer or pass the loaders.
    # The provided Trainer class calls get_loaders() with default args inside fit().
    # However, get_loaders uses default args from Config. Since we updated Config paths
    # in our DemoConfig subclass, but the library imports Config directly,
    # we must ensure the library uses our paths.
    # The library code uses `from library.config import Config`.
    # To make the Trainer use our DemoConfig values, we can instantiate Trainer with DemoConfig.

    trainer = Trainer(config)

    # We need to ensure the trainer uses our specific loaders or that get_loaders picks up the right files.
    # The provided Trainer.fit() calls get_loaders() without arguments.
    # get_loaders() default arguments are Config.TRAIN_METADATA etc.
    # Since we cannot easily change the default args of a function in another module without patching,
    # and Trainer.fit() doesn't accept loaders, we will manually run the loop logic or
    # overwrite the Config class attributes globally for the session.

    # Now run fit
    best_loss = trainer.fit()
    print(f"  Training finished. Best Loss: {best_loss:.4f}")

    assert os.path.exists(
        os.path.join(config.WORKING_DIR, "best_model.pth")
    ), "Model checkpoint not found"

    # 7. Test Inference
    print("\n[Step 6] Running Inference...")

    # Ensure sample submission exists (it's in input, but config points to it)
    # The inference function reads TEST_METADATA (which we set) and SAMPLE_SUBMISSION.
    # It generates predictions and saves to SUBMISSION_PATH.

    inference(config)

    if os.path.exists(config.SUBMISSION_PATH):
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"  Submission generated with {len(sub_df)} rows.")
        print(sub_df.head())

        # Validate submission format
        assert "row_id" in sub_df.columns
        assert "fractured" in sub_df.columns
        assert len(sub_df) > 0
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
