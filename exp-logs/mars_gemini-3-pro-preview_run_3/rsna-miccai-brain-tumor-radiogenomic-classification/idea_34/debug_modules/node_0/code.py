import os
import shutil
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library files
from library.utils import seed_everything
from library.model import GNHRNet
from library.data_loader import get_dataloaders
from library.train_eval import run_training, predict_and_submit

# Constants for the demonstration
DEMO_WORK_DIR = "./working/idea_34/"
DEMO_SUBMISSION_DIR = "./working/submission_demo"
LIMIT_DATA_COUNT = 6  # Small number of samples for speed
BATCH_SIZE = 2
EPOCHS = 1


def clean_working_directory():
    """Cleans up the working directory to ensure a fresh run for the demo."""
    if os.path.exists(DEMO_WORK_DIR):
        print(f"Cleaning existing cache at {DEMO_WORK_DIR}...")
        shutil.rmtree(DEMO_WORK_DIR)
    os.makedirs(DEMO_WORK_DIR, exist_ok=True)

    if os.path.exists(DEMO_SUBMISSION_DIR):
        shutil.rmtree(DEMO_SUBMISSION_DIR)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)


def demo_data_loading():
    print("\n=== Demo 1: Data Loading & Processing ===")

    # We set load_cached_data=False to force the processing logic to run
    # We limit data to a small number to make this fast
    print(f"Initializing DataLoaders with limit_data={LIMIT_DATA_COUNT}...")
    dataloaders = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=2,
        load_cached_data=False,
        limit_data=LIMIT_DATA_COUNT,
    )

    train_loader = dataloaders["train"]

    # Fetch one batch
    print("Fetching one batch from Train DataLoader...")
    inputs, targets, ids = next(iter(train_loader))

    # Verification
    print(f"Input batch shape: {inputs.shape}")
    print(f"Target batch shape: {targets.shape}")
    print(f"IDs: {ids}")

    # Expected shape: (Batch, Channels, Height, Width)
    # Channels = 16 slices * 4 modalities = 64
    # Height/Width = 320
    expected_shape = (BATCH_SIZE, 64, 320, 320)
    assert (
        inputs.shape == expected_shape
    ), f"Expected input shape {expected_shape}, got {inputs.shape}"
    assert targets.shape == (
        BATCH_SIZE,
    ), f"Expected target shape {(BATCH_SIZE,)}, got {targets.shape}"

    print("Data Loading verification passed.")


def demo_model_instantiation():
    print("\n=== Demo 2: Model Architecture & GroupNorm Replacement ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model
    print("Instantiating GNHRNet (EfficientNet-B0 backbone)...")
    model = GNHRNet(
        model_name="efficientnet_b0",
        pretrained=False,  # False for speed in demo (avoid downloading weights)
        in_chans=64,
        num_classes=1,
    ).to(device)

    # Verify GroupNorm replacement
    bn_count = 0
    gn_count = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            bn_count += 1
        elif isinstance(m, torch.nn.GroupNorm):
            gn_count += 1

    print(f"Layer check - BatchNorm2d layers: {bn_count}")
    print(f"Layer check - GroupNorm layers: {gn_count}")

    # We expect BatchNorm to be replaced by GroupNorm
    assert bn_count == 0, "Error: BatchNorm layers still present in the model."
    assert gn_count > 0, "Error: No GroupNorm layers found."

    # Verify Forward Pass
    print("Running forward pass with dummy input...")
    dummy_input = torch.randn(BATCH_SIZE, 64, 320, 320).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    assert output.shape == (
        BATCH_SIZE,
        1,
    ), f"Expected output shape {(BATCH_SIZE, 1)}, got {output.shape}"

    print("Model verification passed.")


def demo_training_pipeline():
    print("\n=== Demo 3: Training Pipeline ===")

    # Run a short training session
    # We reuse the cache generated in Demo 1 if available, or it regenerates quickly
    print(f"Starting training run for {EPOCHS} epoch(s)...")

    best_auc = run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=1e-3,
        patience=1,
        num_workers=0,  # 0 workers often safer for short scripts/demos to avoid overhead
        load_cached_data=True,  # Use the cache we just made/verified
        limit_data=LIMIT_DATA_COUNT,
        seed=42,
    )

    print(f"Training completed. Best AUC: {best_auc}")

    # Verify checkpoint creation
    expected_checkpoint = os.path.join(DEMO_WORK_DIR, "best_model.pth")
    assert os.path.exists(
        expected_checkpoint
    ), f"Checkpoint not found at {expected_checkpoint}"
    print(f"Checkpoint verified at {expected_checkpoint}")


def demo_inference():
    print("\n=== Demo 4: Inference & Submission ===")

    # The predict_and_submit function loads the best_model.pth from the CHECKPOINT_DIR
    # which is hardcoded in library.train_eval to "./working/idea_34/"
    # It will process the test set. Since the test set is small (59 samples), we let it run.

    print("Running inference on test set...")
    predict_and_submit(
        batch_size=BATCH_SIZE,
        num_workers=2,
        load_cached_data=False,  # Force processing of test set
        output_dir=DEMO_SUBMISSION_DIR,
    )

    # Verify submission file
    submission_path = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    df = pd.read_csv(submission_path)
    print(f"Submission file loaded. Rows: {len(df)}")
    print(df.head())

    expected_cols = ["BraTS21ID", "MGMT_value"]
    assert (
        list(df.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(df.columns)}"
    assert len(df) > 0, "Submission file is empty."

    print("Inference verification passed.")


if __name__ == "__main__":
    # 1. Set Seed
    seed_everything(42)

    # 2. Cleanup environment for the demo
    clean_working_directory()

    # 3. Run Demos
    try:
        demo_data_loading()
        demo_model_instantiation()
        demo_training_pipeline()
        demo_inference()
        print("\nAll demonstrations completed successfully.")
    except AssertionError as e:
        print(f"\n[FAILED] Verification failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
