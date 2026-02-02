import os
import sys
import pandas as pd
import torch
import torch.optim as optim
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.data import generate_instance_metadata, get_dataloader
from library.model import SILNet
from library.engine import train_model, predict


def run_demo():
    print("Initializing Demo Execution...")

    # ==========================================
    # 1. Configuration Overrides for Demo Speed
    # ==========================================
    # We modify the Config class attributes directly to affect the behavior
    # of the subsequent library calls.
    Config.DEBUG = True
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.DEBUG_DATASET_SIZE = 10  # Use only 10 subjects per split
    Config.WORKING_DIR = "./working/demo_execution"  # Separate dir for demo outputs

    # Create the working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. Data Loading & Preparation
    # ==========================================
    print("\n[Step 1] Loading and Subsetting Metadata...")

    # Load original metadata
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Subset data for speed
    df_train_meta = df_train_meta.head(Config.DEBUG_DATASET_SIZE)
    df_val_meta = df_val_meta.head(Config.DEBUG_DATASET_SIZE)
    df_test_meta = df_test_meta.head(Config.DEBUG_DATASET_SIZE)

    print(
        f"Subset sizes - Train: {len(df_train_meta)}, Val: {len(df_val_meta)}, Test: {len(df_test_meta)}"
    )

    # ==========================================
    # 3. Instance Generation (2.5D Logic)
    # ==========================================
    print("\n[Step 2] Generating Instance Metadata (Slices)...")

    # We disable cache loading to ensure we generate fresh metadata for our subset
    df_train_inst = generate_instance_metadata(
        df_train_meta, "train", load_cached_data=False
    )
    df_val_inst = generate_instance_metadata(df_val_meta, "val", load_cached_data=False)

    # Verify expansion: 10 subjects * 3 slices/subject = 30 instances (approx, depending on file availability)
    print(f"Generated Training Instances: {len(df_train_inst)}")
    assert (
        len(df_train_inst)
        <= Config.DEBUG_DATASET_SIZE * Config.NUM_INSTANCES_PER_SUBJECT
    )
    assert len(df_train_inst) > 0, "No training instances generated. Check input paths."

    # ==========================================
    # 4. DataLoader Creation & Verification
    # ==========================================
    print("\n[Step 3] Creating DataLoaders...")

    train_loader = get_dataloader(df_train_inst, "train", batch_size=Config.BATCH_SIZE)
    val_loader = get_dataloader(df_val_inst, "val", batch_size=Config.BATCH_SIZE)

    # Fetch one batch to verify shapes
    images, targets = next(iter(train_loader))

    print(f"Batch Shapes - Images: {images.shape}, Targets: {targets.shape}")

    # Assertions for correctness
    # Expected: (Batch, Channels=3, H=224, W=224)
    expected_image_shape = (
        Config.BATCH_SIZE,
        Config.CHANNELS,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )
    assert (
        images.shape == expected_image_shape
    ), f"Expected {expected_image_shape}, got {images.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Expected ({Config.BATCH_SIZE},), got {targets.shape}"
    assert images.dtype == torch.float32

    # ==========================================
    # 5. Model Initialization
    # ==========================================
    print("\n[Step 4] Initializing SILNet Model...")

    model = SILNet(
        pretrained=False
    )  # Disable pretrained download for speed/offline safety if needed
    model = model.to(Config.DEVICE)

    # Optimizer setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # ==========================================
    # 6. Training Loop
    # ==========================================
    print("\n[Step 5] Starting Training (1 Epoch)...")

    # Train the model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=Config.DEVICE,
        num_epochs=Config.NUM_EPOCHS,
        patience=1,  # Aggressive early stopping for demo
    )

    # Verify model artifact creation
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved."
    print(f"Model saved successfully at {best_model_path}")

    # ==========================================
    # 7. Inference & Submission Generation
    # ==========================================
    print("\n[Step 6] Running Inference on Test Set...")

    # Prepare test data
    df_test_inst = generate_instance_metadata(
        df_test_meta, "test", load_cached_data=False
    )
    test_loader = get_dataloader(df_test_inst, "test", batch_size=Config.BATCH_SIZE)

    # Run prediction
    submission_df = predict(trained_model, test_loader, Config.DEVICE)

    print("\nSubmission DataFrame Head:")
    print(submission_df.head())

    # ==========================================
    # 8. Final Output Verification
    # ==========================================
    print("\n[Step 7] Verifying Output Format...")

    required_cols = ["BraTS21ID", "MGMT_value"]
    for col in required_cols:
        assert col in submission_df.columns, f"Missing column: {col}"

    # Check value range
    if not submission_df.empty:
        assert submission_df["MGMT_value"].min() >= 0.0
        assert submission_df["MGMT_value"].max() <= 1.0
        # Check that we have one prediction per subject ID in our subset
        assert len(submission_df) == len(
            df_test_meta
        ), f"Expected {len(df_test_meta)} predictions, got {len(submission_df)}"

    # Save demo submission
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Demo submission saved to {submission_path}")

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    run_demo()
