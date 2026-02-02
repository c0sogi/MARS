import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_class_weights
from library.dataset import AppleDataset, get_transforms, load_data_frames
from library.modeling import AppleNet, train_fold, predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== Starting Apple Disease Detection Pipeline Demo ====\n")

    # 1. Configuration Overrides for Speed and Demonstration
    print("[1] Configuring Demo Environment...")
    Config.SEED = 42
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.IMG_SIZE = 128  # Reduce image size for speed
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce workers
    Config.EXP_NAME = "demo_execution"
    Config.WORKING_DIR = os.path.join("./working", Config.EXP_NAME)
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Use only one backbone for the demo to save time
    # We use the EfficientNet backbone defined in the original config
    Config.BACKBONES = [Config.BACKBONES[0]]

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")
    print(f"    Backbone: {Config.BACKBONES[0]}")
    print("    Configuration complete.\n")

    # 2. Data Loading and Subsetting
    print("[2] Preparing Data Subsets...")
    # Load full dataframes
    train_df_full, val_df_full, test_df_full = load_data_frames(load_cached_data=False)

    # Create tiny subsets for the demo (e.g., 50 train, 20 val, 20 test)
    train_subset = train_df_full.head(50).reset_index(drop=True)
    val_subset = val_df_full.head(20).reset_index(drop=True)
    test_subset = test_df_full.head(20).reset_index(drop=True)

    print(f"    Training Subset: {len(train_subset)} samples")
    print(f"    Validation Subset: {len(val_subset)} samples")
    print(f"    Test Subset: {len(test_subset)} samples")

    # Create Datasets
    train_dataset = AppleDataset(
        train_subset, transform=get_transforms("train"), test=False
    )
    val_dataset = AppleDataset(
        val_subset, transform=get_transforms("valid"), test=False
    )
    test_dataset = AppleDataset(
        test_subset, transform=get_transforms("valid"), test=True
    )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    print("    DataLoaders created successfully.\n")

    # 3. Verification: Dataset and Model
    print("[3] Verifying Components...")

    # Verify Dataset Output
    images, labels = next(iter(train_loader))
    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Label Shape: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image tensor shape"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect label tensor shape"

    # Verify Model Architecture
    print("    Initializing Model...")
    model = AppleNet(
        model_name=Config.BACKBONES[0], num_classes=Config.NUM_CLASSES, pretrained=False
    )
    model.to(Config.DEVICE)

    # Forward pass check
    with torch.no_grad():
        images = images.to(Config.DEVICE)
        logits = model(images)

    print(f"    Model Output Shape: {logits.shape}")
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    print("    Component verification passed.\n")

    # 4. Training Simulation
    print("[4] Running Training Loop (1 Epoch)...")

    # We use the train_fold function from the library
    # This handles optimizer, scheduler, loss, AMP, EMA, and saving
    best_model_path, best_auc = train_fold(
        model_name=Config.BACKBONES[0],
        train_loader=train_loader,
        val_loader=val_loader,
        train_df=train_subset,
        fold_idx=0,
    )

    print(f"    Training complete.")
    print(f"    Best AUC: {best_auc:.4f}")
    print(f"    Saved Model Path: {best_model_path}")

    assert os.path.exists(best_model_path), "Model checkpoint file was not created."
    print("    Training loop verification passed.\n")

    # 5. Inference and Submission
    print("[5] Generating Submission...")

    # Use the predict_and_submit function from the library
    # This handles TTA and ensemble averaging (though we only have 1 model here)
    predict_and_submit(
        models_paths=[best_model_path],
        test_loader=test_loader,
        output_path=Config.SUBMISSION_PATH,
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission Shape: {submission_df.shape}")
    print(f"    Submission Columns: {submission_df.columns.tolist()}")

    # Check rows (should match test subset size)
    assert len(submission_df) == len(
        test_subset
    ), f"Submission row count mismatch. Expected {len(test_subset)}, got {len(submission_df)}"

    # Check columns
    expected_cols = ["image_id"] + Config.CLASS_LABELS
    assert all(
        col in submission_df.columns for col in expected_cols
    ), "Missing columns in submission file"

    # Check values are probabilities (0-1)
    # Note: Due to TTA and floating point, values might slightly exceed 0-1 bounds or sum != 1 depending on implementation,
    # but softmax usually guarantees 0-1.
    pred_cols = Config.CLASS_LABELS
    preds = submission_df[pred_cols].values
    assert (preds >= 0).all() and (
        preds <= 1.0 + 1e-5
    ).all(), "Predictions are not valid probabilities"

    print("    Submission verification passed.\n")

    print("==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
