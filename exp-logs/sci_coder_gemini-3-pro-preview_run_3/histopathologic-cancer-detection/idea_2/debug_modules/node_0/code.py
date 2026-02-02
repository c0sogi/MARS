import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import load_metadata, PathologyDataset, get_transforms
from library.models import get_model
from library.engine import train_one_epoch, validate_with_tta
from library.inference import predict_ensemble


def main():
    print("--- Starting Library Usage Demonstration ---")

    # 1. Setup & Configuration Overrides
    # We override Config settings to ensure the demo runs quickly and uses a temporary directory.
    seed_everything(Config.SEED)

    # Define a temporary working directory for this run
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    # Override Global Config
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Limit to a single model for speed
    Config.MODELS = ["resnet18"]

    # Training hyperparameters for demo
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Selected Model: {Config.MODELS[0]}")

    # 2. Data Preparation (Subsetting)
    # We create small subsets of the original metadata to avoid processing the full dataset.
    print("\n[Data] Creating data subsets for demonstration...")

    # Load original metadata (bypassing cache to ensure we get fresh dataframes)
    # Note: We access the original files first before overriding paths
    df_train_full = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_full = pd.read_csv(Config.VAL_META_PATH)
    df_test_full = pd.read_csv(Config.TEST_META_PATH)

    # Create subsets
    df_train_sub = df_train_full.head(32).copy()  # 4 batches of 8
    df_val_sub = df_val_full.head(16).copy()  # 2 batches of 8
    df_test_sub = df_test_full.head(16).copy()  # 2 batches of 8

    # Save subsets to the demo directory
    train_sub_path = os.path.join(demo_dir, "train_subset.csv")
    val_sub_path = os.path.join(demo_dir, "val_subset.csv")
    test_sub_path = os.path.join(demo_dir, "test_subset.csv")

    df_train_sub.to_csv(train_sub_path, index=False)
    df_val_sub.to_csv(val_sub_path, index=False)
    df_test_sub.to_csv(test_sub_path, index=False)

    # IMPORTANT: Override Config paths so library functions use our subsets
    Config.TRAIN_META_PATH = train_sub_path
    Config.VAL_META_PATH = val_sub_path
    Config.TEST_META_PATH = test_sub_path

    print(
        f"Subset sizes - Train: {len(df_train_sub)}, Val: {len(df_val_sub)}, Test: {len(df_test_sub)}"
    )

    # 3. Dataset & DataLoader Instantiation
    print("\n[Data] Initializing Datasets and Loaders...")

    # Create Datasets
    train_dataset = PathologyDataset(
        df=df_train_sub, phase="train", transform=get_transforms("train")
    )

    val_dataset = PathologyDataset(
        df=df_val_sub, phase="val", transform=get_transforms("val")
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Verification
    sample_img, sample_lbl = next(iter(train_loader))
    assert sample_img.shape == (
        Config.BATCH_SIZE,
        3,
        Config.CROP_SIZE,
        Config.CROP_SIZE,
    ), f"Unexpected image shape: {sample_img.shape}"
    assert sample_lbl.shape == (
        Config.BATCH_SIZE,
    ), f"Unexpected label shape: {sample_lbl.shape}"
    print("DataLoader verification successful.")

    # 4. Model Initialization & Training Loop
    print("\n[Training] Initializing Model and running 1 epoch...")

    device = Config.DEVICE
    # Using pretrained=False to avoid download overhead/errors in demo environment
    model = get_model(Config.MODELS[0], pretrained=False)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Run Training
    train_metrics = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        epoch=1,
    )

    # Verify Metrics
    assert "Loss" in train_metrics, "Training metrics missing Loss"
    assert "AUC" in train_metrics, "Training metrics missing AUC"
    print(f"Epoch 1 Metrics: {train_metrics}")

    # 5. Validation
    print("\n[Validation] Running Validation with TTA...")

    val_metrics = validate_with_tta(model=model, loader=val_loader, device=device)

    assert "Loss" in val_metrics, "Validation metrics missing Loss"
    print(f"Validation Metrics: {val_metrics}")

    # 6. Saving Model for Inference
    print("\n[Checkpoint] Saving model weights...")
    # The inference engine expects weights at {WORKING_DIR}/{model_name}_best.pth
    weights_path = os.path.join(Config.WORKING_DIR, f"{Config.MODELS[0]}_best.pth")
    torch.save(model.state_dict(), weights_path)

    assert os.path.exists(weights_path), "Failed to save model weights"
    print(f"Saved weights to {weights_path}")

    # 7. Ensemble Inference
    print("\n[Inference] Running Ensemble Prediction...")

    # predict_ensemble handles data loading internally using Config.TEST_META_PATH
    # It iterates through Config.MODELS and loads weights from Config.WORKING_DIR
    predict_ensemble(
        output_path=Config.SUBMISSION_PATH,
        device=device,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
    )

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_submission = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {df_submission.shape}")

    assert len(df_submission) == len(
        df_test_sub
    ), f"Submission row count mismatch. Expected {len(df_test_sub)}, got {len(df_submission)}"

    assert list(df_submission.columns) == [
        "id",
        "label",
    ], f"Invalid columns in submission: {df_submission.columns}"

    # Check if probabilities are valid
    probs = df_submission["label"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of [0, 1] range"

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    main()
