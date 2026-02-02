import os
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, load_hierarchy_mappings
from library.data import get_dataloaders
from library.model import HierarchicalEfficientNet
from library.train import fit
from library.inference import predict_tta, generate_submission


def run_demonstration():
    print("==== Starting Library Demonstration ====")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Rapid Execution
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demo...")

    # Override Config settings to run quickly on a tiny subset
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 64  # Use only 64 images
    Config.STAGE1_EPOCHS = 1  # Train for only 1 epoch
    Config.STAGE2_EPOCHS = 1
    Config.STAGE1_BATCH_SIZE = 8  # Small batch size
    Config.STAGE2_BATCH_SIZE = 8
    Config.NUM_WORKERS = 2  # Reduce worker overhead

    # Redirect outputs to a demo directory to avoid clutter/overwriting
    Config.WORK_DIR = "./working/demo_run"
    Config.CHECKPOINT_DIR_STAGE1 = os.path.join(Config.WORK_DIR, "stage_1")
    Config.CHECKPOINT_DIR_STAGE2 = os.path.join(Config.WORK_DIR, "stage_2")
    Config.SUBMISSION_PATH = os.path.join(Config.WORK_DIR, "submission.csv")
    Config.HIERARCHY_CACHE_PATH = os.path.join(
        Config.WORK_DIR, "hierarchy_mappings.parquet"
    )

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR_STAGE1, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR_STAGE2, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("    Configuration updated: DEBUG=True, Epochs=1, SampleSize=64")

    # -------------------------------------------------------------------------
    # 2. Test Hierarchy Mapping
    # -------------------------------------------------------------------------
    print("\n[2] Testing Hierarchy Mapping Logic...")

    # Force re-computation by setting load_cached_data=False initially if desired,
    # but here we test the standard flow.
    hierarchy_df = load_hierarchy_mappings(
        Config.TRAIN_METADATA_JSON, Config.HIERARCHY_CACHE_PATH, load_cached_data=False
    )

    # Assertions to verify correctness
    assert not hierarchy_df.empty, "Hierarchy DataFrame should not be empty."
    assert "category_id" in hierarchy_df.columns, "Missing 'category_id' column."
    assert "genus_id" in hierarchy_df.columns, "Missing 'genus_id' column."
    assert "family_id" in hierarchy_df.columns, "Missing 'family_id' column."
    assert os.path.exists(
        Config.HIERARCHY_CACHE_PATH
    ), "Parquet cache file was not created."

    print(f"    Hierarchy loaded successfully. Mapped {len(hierarchy_df)} categories.")

    # -------------------------------------------------------------------------
    # 3. Test Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Testing Data Loading (Debug Mode)...")

    # Load dataloaders with debug=True (uses Config.DEBUG_SAMPLE_SIZE)
    train_loader, val_loader, test_loader = get_dataloaders(
        img_size=224, batch_size=Config.STAGE1_BATCH_SIZE, debug=True
    )

    # Fetch a single batch to verify structure
    images, targets = next(iter(train_loader))

    # Assertions
    assert images.shape == (
        Config.STAGE1_BATCH_SIZE,
        3,
        224,
        224,
    ), f"Expected image shape (8, 3, 224, 224), got {images.shape}"
    assert "species" in targets, "Targets dictionary missing 'species'."
    assert "genus" in targets, "Targets dictionary missing 'genus'."
    assert "family" in targets, "Targets dictionary missing 'family'."
    assert (
        targets["species"].shape[0] == Config.STAGE1_BATCH_SIZE
    ), "Target batch size mismatch."

    print(f"    Data loaded successfully. Batch shape: {images.shape}")

    # -------------------------------------------------------------------------
    # 4. Test Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Testing Model Architecture...")

    # Initialize model (pretrained=False for speed/offline execution)
    model = HierarchicalEfficientNet(pretrained=False)
    model.to(Config.DEVICE)

    # Perform a forward pass with the batch from step 3
    images = images.to(Config.DEVICE)
    outputs = model(images)

    # Assertions on output
    assert "species" in outputs, "Model output missing 'species' logits."
    assert outputs["species"].shape == (
        Config.STAGE1_BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Incorrect species logits shape: {outputs['species'].shape}"
    assert outputs["genus"].shape == (
        Config.STAGE1_BATCH_SIZE,
        Config.NUM_GENERA,
    ), "Incorrect genus logits shape."

    print("    Model forward pass successful. Output shapes verified.")

    # -------------------------------------------------------------------------
    # 5. Test Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Testing Training Loop (Stage 1)...")

    # Run the fit function. It handles the optimizer, scheduler, and validation loop.
    trained_model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.STAGE1_EPOCHS,
        checkpoint_dir=Config.CHECKPOINT_DIR_STAGE1,
        stage_name="Demo_Stage1",
    )

    # Verify checkpoint creation
    expected_checkpoint = os.path.join(Config.CHECKPOINT_DIR_STAGE1, "best_model.pth")
    assert os.path.exists(
        expected_checkpoint
    ), f"Checkpoint not found at {expected_checkpoint}"

    print("    Training loop completed. Checkpoint saved.")

    # -------------------------------------------------------------------------
    # 6. Test Inference
    # -------------------------------------------------------------------------
    print("\n[6] Testing Inference...")

    # Use the trained model to predict on the test set
    ids, preds = predict_tta(trained_model, test_loader, Config.DEVICE)

    # Assertions
    assert (
        len(ids) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} predictions, got {len(ids)}"
    assert len(ids) == len(preds), "Mismatch between ID count and prediction count."

    print(f"    Inference successful. Generated {len(preds)} predictions.")

    # -------------------------------------------------------------------------
    # 7. Test Submission Generation
    # -------------------------------------------------------------------------
    print("\n[7] Testing Submission Generation...")

    generate_submission(ids, preds, Config.SUBMISSION_PATH)

    # Verify file existence and content format
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(df_sub.columns) == [
        "Id",
        "Predicted",
    ], f"Submission columns mismatch. Got {list(df_sub.columns)}"
    assert len(df_sub) == Config.DEBUG_SAMPLE_SIZE, "Submission row count mismatch."

    print(f"    Submission saved to {Config.SUBMISSION_PATH}")
    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    run_demonstration()
