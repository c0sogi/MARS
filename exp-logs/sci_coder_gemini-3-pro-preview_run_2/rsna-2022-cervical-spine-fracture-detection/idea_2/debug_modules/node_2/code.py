import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import from the provided library
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
import library.train as train


def main():
    print("=== Starting RSNA Cervical Spine Fracture Detection Demo ===\n")

    # 1. Configuration Overrides for Speed
    # We monkey-patch the config to run a very short cycle
    config.EPOCHS = 1
    config.DEBUG_SIZE = 12  # Small enough for 3 batches of size 4
    config.BATCH_SIZE = 4
    config.SEQ_LEN = 16  # Reduced sequence length for faster forward pass

    print(f"Configuration:")
    print(f"  Device: {config.DEVICE}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Debug Size: {config.DEBUG_SIZE}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Seq Len: {config.SEQ_LEN}")

    # Set seeds
    utils.seed_everything(config.SEED)
    print("\n[Step 1] Seeds set successfully.")

    # 2. Verify Loss Function
    print("\n[Step 2] Verifying WeightedMultiLabelLoss...")
    criterion = utils.WeightedMultiLabelLoss()
    # Create dummy data: Batch=2, Classes=8
    dummy_logits = torch.randn(2, 8)
    dummy_targets = torch.randint(0, 2, (2, 8)).float()

    loss = criterion(dummy_logits, dummy_targets)

    assert isinstance(loss, torch.Tensor), "Loss must be a tensor"
    assert loss.ndim == 0, "Loss must be a scalar"
    assert not torch.isnan(loss), "Loss resulted in NaN"
    print(f"  Loss check passed. Value: {loss.item():.4f}")

    # 3. Verify Data Loading Components
    print("\n[Step 3] Verifying Data Loading...")

    # Load metadata manually to check dataset instantiation
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {config.TRAIN_METADATA_PATH}")

    train_df = pd.read_csv(config.TRAIN_METADATA_PATH).iloc[: config.DEBUG_SIZE]

    # Generate path map (using cache logic from library)
    path_map = data.get_study_paths_map(
        train_df,
        config.TRAIN_IMAGES_DIR,
        "demo_train_paths",
        load_cached_data=False,  # Force re-scan for demo purposes on small subset
    )

    # Instantiate Dataset
    ds = data.RSNADataset(
        train_df,
        path_map,
        config.TRAIN_IMAGES_DIR,
        transform=None,  # Skip heavy transforms for shape check
        seq_len=config.SEQ_LEN,
        is_train=True,
    )

    # Check __getitem__
    volume, label = ds[0]

    # Expected shape: (Seq_Len, 3, H, W)
    # H, W are defined in config.IMG_SIZE (224, 224)
    expected_shape = (config.SEQ_LEN, 3, config.IMG_SIZE[0], config.IMG_SIZE[1])

    print(f"  Sample Volume Shape: {volume.shape}")
    print(f"  Sample Label Shape: {label.shape}")

    assert (
        volume.shape == expected_shape
    ), f"Volume shape mismatch. Expected {expected_shape}, got {volume.shape}"
    assert label.shape == (
        8,
    ), f"Label shape mismatch. Expected (8,), got {label.shape}"

    # Check DataLoader generation
    train_loader, val_loader, test_loader = data.get_dataloaders(
        batch_size=config.BATCH_SIZE, load_cached_data=True, debug=True
    )

    # Fetch one batch
    images, targets = next(iter(train_loader))
    print(f"  Batch Images Shape: {images.shape}")
    print(f"  Batch Targets Shape: {targets.shape}")

    assert images.shape == (
        config.BATCH_SIZE,
        *expected_shape,
    ), "Batch image shape mismatch"
    assert targets.shape == (config.BATCH_SIZE, 8), "Batch target shape mismatch"
    print("  Data Loading verification passed.")

    # 4. Verify Model Architecture
    print("\n[Step 4] Verifying Model Architecture...")

    # Instantiate model
    # We use pretrained=False to avoid downloading weights during this quick demo if possible,
    # but the library code defaults to True. We'll stick to the library default.
    net = model.CervicalSpineSeqModel(pretrained=False)
    net.to(config.DEVICE)
    net.eval()

    # Move dummy batch to device
    images = images.to(config.DEVICE)

    with torch.no_grad():
        logits = net(images)

    print(f"  Model Output Logits Shape: {logits.shape}")

    assert logits.shape == (
        config.BATCH_SIZE,
        8,
    ), f"Model output shape mismatch. Expected {(config.BATCH_SIZE, 8)}, got {logits.shape}"
    print("  Model verification passed.")

    # 5. Verify Full Training Pipeline
    print("\n[Step 5] Running Full Training Pipeline (Debug Mode)...")

    # Ensure working directory exists (handled by config, but good to ensure)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Run the training function provided in library.train
    # This handles training loop, validation, saving model, and generating submission
    try:
        train.train_model(debug=True)
    except Exception as e:
        print(f"Training pipeline failed with error: {e}")
        raise e

    # 6. Verify Submission Output
    print("\n[Step 6] Verifying Submission...")

    if os.path.exists(config.SUBMISSION_PATH):
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"  Submission file found with {len(sub_df)} rows.")
        print("  First 5 rows:")
        print(sub_df.head())

        # Basic validation of submission format
        assert "row_id" in sub_df.columns, "Submission missing row_id column"
        assert "fractured" in sub_df.columns, "Submission missing fractured column"
        assert not sub_df.empty, "Submission file is empty"
        print("  Submission format verification passed.")
    else:
        raise FileNotFoundError(
            f"Submission file not generated at {config.SUBMISSION_PATH}"
        )

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
