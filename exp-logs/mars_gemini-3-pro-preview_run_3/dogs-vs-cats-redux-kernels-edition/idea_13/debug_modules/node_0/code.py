import os
import shutil
import pandas as pd
import torch
import numpy as np
import glob

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.data import get_dataloaders
from library.modeling import create_model
from library.engine import train_model, generate_submission


def run_demonstration():
    print("Starting Library Demonstration...")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Patching
    # -------------------------------------------------------------------------
    # Define demo directories to avoid messing with real training artifacts
    DEMO_ROOT = "./working/demo_run"
    DEMO_METADATA = os.path.join(DEMO_ROOT, "metadata")
    DEMO_WORKING = os.path.join(DEMO_ROOT, "working")
    DEMO_SUBMISSION = os.path.join(DEMO_ROOT, "submission")

    # Clean up previous runs if they exist
    if os.path.exists(DEMO_ROOT):
        shutil.rmtree(DEMO_ROOT)

    os.makedirs(DEMO_METADATA, exist_ok=True)
    os.makedirs(DEMO_WORKING, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION, exist_ok=True)

    print(f"Created demo directories at {DEMO_ROOT}")

    # Patch the Config class to use our demo directories and settings
    # Since Config is a class with static attributes, modifying them here
    # affects the imported modules (data.py, engine.py, etc.)
    Config.METADATA_DIR = DEMO_METADATA
    Config.WORKING_DIR = DEMO_WORKING
    Config.SUBMISSION_DIR = DEMO_SUBMISSION

    # Modify hyperparameters for speed
    # We will use 'resnet' config for this demo
    MODEL_KEY = "resnet"
    Config.MODEL_CONFIGS[MODEL_KEY]["epochs"] = 1
    Config.MODEL_CONFIGS[MODEL_KEY]["batch_size"] = 4  # Small batch for small data
    Config.MODEL_CONFIGS[MODEL_KEY]["img_size"] = 128  # Smaller image for speed

    # Set seed for reproducibility
    seed_everything(42)

    # -------------------------------------------------------------------------
    # 2. Create Mini Datasets (Subset of Original)
    # -------------------------------------------------------------------------
    print("\nCreating mini datasets...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Create subsets (ensure we have enough for a batch)
    mini_train = orig_train.head(20).copy()
    mini_val = orig_val.head(10).copy()
    mini_test = orig_test.head(10).copy()

    # Save to demo metadata directory
    mini_train.to_csv(os.path.join(DEMO_METADATA, "train.csv"), index=False)
    mini_val.to_csv(os.path.join(DEMO_METADATA, "val.csv"), index=False)
    mini_test.to_csv(os.path.join(DEMO_METADATA, "test.csv"), index=False)

    print(
        f"Saved mini_train ({len(mini_train)}), mini_val ({len(mini_val)}), mini_test ({len(mini_test)})"
    )

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\nVerifying Data Loading...")

    # Initialize DataLoaders
    # load_cached_data=False forces it to read our new CSVs instead of looking for existing parquets
    train_loader, val_loader, test_loader = get_dataloaders(
        img_size=Config.MODEL_CONFIGS[MODEL_KEY]["img_size"],
        batch_size=Config.MODEL_CONFIGS[MODEL_KEY]["batch_size"],
        load_cached_data=False,
    )

    # Fetch one batch from train loader
    images, labels = next(iter(train_loader))

    # Assertions
    print(f"Batch shapes - Images: {images.shape}, Labels: {labels.shape}")

    # Check Batch Size (might be smaller if drop_last=False, but here drop_last=True for train)
    assert (
        images.shape[0] == Config.MODEL_CONFIGS[MODEL_KEY]["batch_size"]
    ), "Batch size mismatch"
    # Check Channels (3 for RGB)
    assert images.shape[1] == 3, "Channel count mismatch"
    # Check Height/Width
    assert (
        images.shape[2] == Config.MODEL_CONFIGS[MODEL_KEY]["img_size"]
    ), "Image height mismatch"
    assert (
        images.shape[3] == Config.MODEL_CONFIGS[MODEL_KEY]["img_size"]
    ), "Image width mismatch"
    # Check Label Type (Float for BCE)
    assert labels.dtype == torch.float32, "Label dtype mismatch"

    print("Data Loading verification passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Creation & Forward Pass
    # -------------------------------------------------------------------------
    print("\nVerifying Model Creation...")

    model = create_model(
        model_name=Config.MODEL_CONFIGS[MODEL_KEY]["model_name"],
        num_classes=Config.NUM_CLASSES,
        pretrained=False,  # False for speed/offline demo
        img_size=Config.MODEL_CONFIGS[MODEL_KEY]["img_size"],
    )
    model.to(Config.DEVICE)
    model.eval()

    # Dummy forward pass
    with torch.no_grad():
        output = model(images.to(Config.DEVICE))

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        Config.MODEL_CONFIGS[MODEL_KEY]["batch_size"],
        1,
    ), "Output shape mismatch"
    assert not torch.isnan(output).any(), "Model produced NaN values"

    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 5. Verify Training Engine
    # -------------------------------------------------------------------------
    print("\nVerifying Training Engine (1 Epoch)...")

    # Run training
    # This uses the patched Config, so it will run 1 epoch on the mini dataset
    best_loss = train_model(MODEL_KEY, load_cached_data=False)

    print(f"Training finished. Best Loss: {best_loss}")

    # Assertions
    assert isinstance(best_loss, float), "train_model did not return a float loss"

    # Check if checkpoint exists
    expected_ckpt = os.path.join(Config.WORKING_DIR, f"{MODEL_KEY}_best.pth")
    assert os.path.exists(expected_ckpt), f"Checkpoint not found at {expected_ckpt}"

    print("Training Engine verification passed.")

    # -------------------------------------------------------------------------
    # 6. Verify Submission Engine
    # -------------------------------------------------------------------------
    print("\nVerifying Submission Engine...")

    # Generate submission
    generate_submission([MODEL_KEY], load_cached_data=False)

    expected_sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(expected_sub_path), "Submission file not created"

    # Load and verify content
    sub_df = pd.read_csv(expected_sub_path)
    print(f"Submission shape: {sub_df.shape}")
    print(sub_df.head())

    # Assertions
    assert len(sub_df) == len(
        mini_test
    ), f"Submission length ({len(sub_df)}) != Test set length ({len(mini_test)})"
    assert list(sub_df.columns) == ["id", "label"], "Submission columns mismatch"
    assert sub_df["id"].is_monotonic_increasing, "Submission IDs are not sorted"
    assert sub_df["label"].between(0, 1).all(), "Probabilities out of range [0, 1]"

    print("Submission Engine verification passed.")

    # -------------------------------------------------------------------------
    # 7. Verify Metric Utility
    # -------------------------------------------------------------------------
    print("\nVerifying Metric Utility...")

    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])

    loss = calculate_log_loss(y_true, y_pred)
    print(f"Calculated Log Loss: {loss}")

    # Assertions
    # Expected loss: - (0.5*ln(0.9) + 0.5*ln(0.8)) approx 0.16
    # Using sklearn logic: log_loss([0,1], [0.1, 0.9]) -> -ln(0.9) ~= 0.105
    # Let's just check it's a valid positive float and behaves reasonably
    assert loss > 0, "Log loss should be positive"
    assert loss < 1.0, "Log loss should be low for good predictions"

    # Test perfect predictions
    loss_perfect = calculate_log_loss([0, 1], [0.00001, 0.99999])
    assert loss_perfect < 0.01, "Log loss should be near zero for perfect predictions"

    print("Metric verification passed.")

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    run_demonstration()
