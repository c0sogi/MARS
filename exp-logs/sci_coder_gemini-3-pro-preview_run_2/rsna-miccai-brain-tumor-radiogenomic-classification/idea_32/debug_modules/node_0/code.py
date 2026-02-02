import os
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# Import provided library modules
from library import config, utils, data_loader, model, train

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Creates a small subset of the metadata to speed up the demo.
    Overrides config paths to point to this subset.
    """
    print("--- Setting up Demo Environment ---")

    # Define demo directories
    demo_dir = os.path.join(config.WORKING_DIR, "demo_run")
    demo_metadata_dir = os.path.join(demo_dir, "metadata")
    demo_cache_dir = os.path.join(demo_dir, "working")
    demo_submission_dir = os.path.join(demo_dir, "submission")

    os.makedirs(demo_metadata_dir, exist_ok=True)
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    # Slice metadata files (take top 5 rows)
    # We read the original metadata files provided in the environment
    splits = {
        "train": config.TRAIN_METADATA_PATH,
        "val": config.VAL_METADATA_PATH,
        "test": config.TEST_METADATA_PATH,
    }

    new_paths = {}

    for split_name, original_path in splits.items():
        if os.path.exists(original_path):
            df = pd.read_csv(original_path)
            # Take a small subset. Ensure we have at least a few samples.
            subset = df.head(8).copy()
            save_path = os.path.join(demo_metadata_dir, f"{split_name}.csv")
            subset.to_csv(save_path, index=False)
            new_paths[split_name] = save_path
            print(f"Created demo {split_name} metadata with {len(subset)} samples.")
        else:
            raise FileNotFoundError(f"Original metadata not found at {original_path}")

    # Override config paths
    config.TRAIN_METADATA_PATH = new_paths["train"]
    config.VAL_METADATA_PATH = new_paths["val"]
    config.TEST_METADATA_PATH = new_paths["test"]

    # Override output paths
    config.CACHE_DIR = demo_cache_dir
    config.SUBMISSION_DIR = demo_submission_dir
    config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "demo_submission.csv")

    # Override Hyperparameters for speed
    config.BATCH_SIZE = 4
    config.NUM_EPOCHS = 1
    config.EARLY_STOPPING_PATIENCE = 1
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny demo

    print("Config paths and hyperparameters updated for demo.")
    return demo_dir


def demonstrate_utils():
    print("\n--- Demonstrating Utils ---")

    # 1. Set Seed
    utils.set_seed(123)
    print("Random seed set to 123.")

    # 2. Check Device
    device = utils.get_device()
    print(f"Device detected: {device}")

    # 3. AverageMeter
    meter = utils.AverageMeter()
    meter.update(10, n=2)
    meter.update(20, n=2)
    assert (
        meter.avg == 15.0
    ), f"AverageMeter logic failed. Expected 15.0, got {meter.avg}"
    print("AverageMeter logic verified.")

    # 4. Checkpoint Logic
    dummy_state = {"epoch": 1, "val": 0.99}
    ckpt_dir = os.path.join(config.CACHE_DIR, "ckpt_test")
    utils.save_checkpoint(dummy_state, is_best=True, checkpoint_dir=ckpt_dir)

    assert os.path.exists(
        os.path.join(ckpt_dir, "best_model.pth")
    ), "Best model not saved."
    print("Checkpoint saving verified.")


def demonstrate_data_loader():
    print("\n--- Demonstrating Data Loader ---")

    # This will trigger ROI calculation on the small demo subset
    # Since we overrode the metadata paths, this should be fast.
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        load_cached_data=False
    )

    print(f"Train Loader Batches: {len(train_loader)}")

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Validation
    # Shape: (Batch, Channels, Height, Width)
    # Channels = 4 modalities * 3 slices = 12
    expected_shape = (config.BATCH_SIZE, 12, config.IMG_SIZE, config.IMG_SIZE)

    assert (
        images.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {images.shape}"
    assert labels.shape == (
        config.BATCH_SIZE,
    ), f"Label shape mismatch. Expected {(config.BATCH_SIZE,)}, got {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32"

    print(f"Batch loaded successfully. Image Shape: {images.shape}")
    return images, labels


def demonstrate_model(sample_batch):
    print("\n--- Demonstrating Model ---")

    device = utils.get_device()
    net = model.AsymmetricEfficientNet().to(device)

    inputs = sample_batch.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = net(inputs)

    # Validation
    # Output should be (Batch, 1) logits
    assert outputs.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Got {outputs.shape}"

    print("Model forward pass successful.")
    return net


def demonstrate_training_pipeline():
    print("\n--- Demonstrating Full Training Pipeline ---")

    # We use the run_training function which orchestrates everything.
    # We enable debug=True to ensure it truncates even further if needed (though we already sliced metadata).
    # We use load_cached_data=True so it picks up the ROI cache generated in the data_loader step above.

    try:
        train.run_training(
            num_epochs=1,
            batch_size=config.BATCH_SIZE,
            debug=True,
            load_cached_data=True,
        )
        print("Training run completed without errors.")
    except Exception as e:
        print(f"Training run failed with error: {e}")
        raise e

    # Verify Submission
    if os.path.exists(config.SUBMISSION_PATH):
        df_sub = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission file generated at {config.SUBMISSION_PATH}")
        print(f"Submission rows: {len(df_sub)}")

        # Basic check on submission content
        assert "BraTS21ID" in df_sub.columns
        assert "MGMT_value" in df_sub.columns
        assert not df_sub.isnull().values.any(), "Submission contains NaNs"
    else:
        raise FileNotFoundError("Submission file was not generated.")


def cleanup(demo_dir):
    print("\n--- Cleanup ---")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
        print(f"Removed temporary directory: {demo_dir}")


if __name__ == "__main__":
    demo_dir = None
    try:
        # 1. Setup
        demo_dir = setup_demo_environment()

        # 2. Utils
        demonstrate_utils()

        # 3. Data Loader
        sample_images, sample_labels = demonstrate_data_loader()

        # 4. Model
        demonstrate_model(sample_images)

        # 5. Full Pipeline
        demonstrate_training_pipeline()

        print("\nAll demonstrations passed successfully.")

    except Exception as e:
        print(f"\nDemonstration FAILED: {e}")
        raise e
    finally:
        # Cleanup to leave workspace clean
        if demo_dir:
            cleanup(demo_dir)
