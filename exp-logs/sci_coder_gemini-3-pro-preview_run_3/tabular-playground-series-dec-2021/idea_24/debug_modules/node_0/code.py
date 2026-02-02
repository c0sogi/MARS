import os
import shutil
import numpy as np
import pandas as pd
import torch
import sys

# ------------------------------------------------------------------------------
# 1. Setup and Configuration Override
# ------------------------------------------------------------------------------
# We import Config first to modify it before other modules use it.
from library.config import Config
from library.utils import seed_everything, get_device

# Define a temporary directory for this demonstration
DEMO_DIR = "./working/demo_pipeline"
DEMO_DATA_DIR = os.path.join(DEMO_DIR, "data")
DEMO_CACHE_DIR = os.path.join(DEMO_DIR, "cache")
DEMO_SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

# Clean up previous runs if they exist
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)

os.makedirs(DEMO_DATA_DIR, exist_ok=True)
os.makedirs(DEMO_CACHE_DIR, exist_ok=True)
os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

print(f"Setting up demo environment in {DEMO_DIR}...")

# Override Config paths to point to our demo environment
Config.WORKING_DIR = DEMO_CACHE_DIR
Config.SUBMISSION_DIR = DEMO_SUBMISSION_DIR
Config.SUBMISSION_PATH = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")

# Override Hyperparameters for speed
Config.BATCH_SIZE = 256
Config.EPOCHS = 2
Config.HIDDEN_DIM = 128  # Reduced for speed
Config.NUM_BLOCKS = 2  # Reduced for speed

# ------------------------------------------------------------------------------
# 2. Create Mini Datasets
# ------------------------------------------------------------------------------
# We create small subsets of the original data to verify logic quickly.
print("Creating mini datasets for fast verification...")

# Paths to original metadata
orig_train_path = "./metadata/train.parquet"
orig_val_path = "./metadata/val.parquet"
orig_test_path = "./metadata/test.parquet"

# Load head of original files
df_train_mini = pd.read_parquet(orig_train_path).head(2000)
df_val_mini = pd.read_parquet(orig_val_path).head(500)
df_test_mini = pd.read_parquet(orig_test_path).head(500)

# Save mini files to demo data directory
mini_train_path = os.path.join(DEMO_DIR, "mini_train.parquet")
mini_val_path = os.path.join(DEMO_DIR, "mini_val.parquet")
mini_test_path = os.path.join(DEMO_DIR, "mini_test.parquet")

df_train_mini.to_parquet(mini_train_path, index=False)
df_val_mini.to_parquet(mini_val_path, index=False)
df_test_mini.to_parquet(mini_test_path, index=False)

# Update Config to point to mini files
Config.TRAIN_DATA_PATH = mini_train_path
Config.VAL_DATA_PATH = mini_val_path
Config.TEST_DATA_PATH = mini_test_path

print("Mini datasets created.")

# ------------------------------------------------------------------------------
# 3. Import Library Modules
# ------------------------------------------------------------------------------
# Now that Config is patched, we import the rest of the library.
from library.data_loader import get_dataloaders
from library.model import ParallelDCNResNet
from library.train import Trainer
from library.infer import run_inference

# ------------------------------------------------------------------------------
# 4. Demonstration Functions
# ------------------------------------------------------------------------------


def demo_data_loading():
    print("\n=== Demo: Data Loading & Processing ===")

    # Force reload to ensure processing logic runs on new mini data
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Force processing
        num_workers=0,  # Avoid multiprocessing overhead for small demo
    )

    # Verification
    print("Verifying DataLoader shapes...")

    # Check Train Loader
    X_batch, y_batch = next(iter(train_loader))
    print(f"Train Batch Shape: X={X_batch.shape}, y={y_batch.shape}")
    assert X_batch.shape[0] <= Config.BATCH_SIZE
    assert len(X_batch.shape) == 2

    # Check Test IDs
    print(f"Test IDs count: {len(test_ids)}")
    assert len(test_ids) == 500, f"Expected 500 test IDs, got {len(test_ids)}"

    # Check Feature Engineering (Input dim should be > raw features)
    # Raw continuous (10) + Binary (44) + Engineered (5) = 59 features expected
    # Note: Aspect is kept, Aspect_Sin/Cos added.
    # Raw continuous: 10. Engineered: 5. Binary: 44. Total: 59.
    input_dim = X_batch.shape[1]
    print(f"Input Feature Dimension: {input_dim}")
    assert input_dim == 59, f"Expected 59 input features, got {input_dim}"

    print("Data loading verification passed.")
    return train_loader, val_loader, test_loader, input_dim


def demo_model_architecture(input_dim):
    print("\n=== Demo: Model Architecture ===")
    device = get_device()

    model = ParallelDCNResNet(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_BLOCKS,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT,
    ).to(device)

    print(f"Model initialized on {device}.")

    # Create dummy input
    dummy_input = torch.randn(16, input_dim).to(device)

    # Forward pass
    output = model(dummy_input)
    print(f"Forward pass output shape: {output.shape}")

    # Assertions
    assert output.shape == (
        16,
        Config.NUM_CLASSES,
    ), f"Expected output shape (16, {Config.NUM_CLASSES}), got {output.shape}"

    print("Model architecture verification passed.")
    return model


def demo_training(model, train_loader, val_loader):
    print("\n=== Demo: Training Loop ===")
    device = get_device()

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        patience=2,  # Short patience for demo
    )

    # Run training for limited epochs
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_model = trainer.fit(epochs=Config.EPOCHS)

    # Check if best model file exists
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "best_model.pth was not saved."
    print(f"Training complete. Best model saved at {model_path}.")

    return best_model


def demo_inference():
    print("\n=== Demo: Inference Pipeline ===")

    # Run inference using the library function
    # This function internally loads the best model from Config.WORKING_DIR
    # and generates predictions on the test set defined in Config.
    run_inference(batch_size=Config.BATCH_SIZE, num_workers=0)

    # Verify Submission
    sub_path = Config.SUBMISSION_PATH
    assert os.path.exists(sub_path), f"Submission file not found at {sub_path}"

    df_sub = pd.read_csv(sub_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    # Assertions
    assert df_sub.shape == (500, 2), f"Expected (500, 2), got {df_sub.shape}"
    assert list(df_sub.columns) == [
        "Id",
        "Cover_Type",
    ], "Incorrect columns in submission."
    assert df_sub["Cover_Type"].isnull().sum() == 0, "Null values found in predictions."

    # Check value range (Original classes are 1-7)
    unique_preds = df_sub["Cover_Type"].unique()
    print(f"Unique predicted classes: {unique_preds}")
    assert np.all(
        (unique_preds >= 1) & (unique_preds <= 7)
    ), "Predictions out of range [1, 7]"

    print("Inference verification passed.")


# ------------------------------------------------------------------------------
# 5. Main Execution
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    seed_everything(Config.SEED)

    try:
        # 1. Data
        train_loader, val_loader, test_loader, input_dim = demo_data_loading()

        # 2. Model
        model = demo_model_architecture(input_dim)

        # 3. Train
        trained_model = demo_training(model, train_loader, val_loader)

        # 4. Inference
        demo_inference()

        print("\nAll demonstrations completed successfully!")

    except AssertionError as e:
        print(f"\n[FAILED] Verification failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
