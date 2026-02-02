import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
import library.config as config
import library.data_utils as data_utils
from library.dataset import RMSHDDataset
from library.model import RMSHDNet
from library.engine import run_training, predict


def run_demo():
    # 1. Setup and Configuration
    print("Initializing Demo...")
    config.seed_everything(42)

    # Override working directory for demo purposes to keep it isolated
    DEMO_WORKING_DIR = "./working/demo_execution"
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Monkey-patch the config working directory so data_utils saves caches there
    config.WORKING_DIR = DEMO_WORKING_DIR

    # Define device
    device = config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Preparation (Load Metadata and Subset)
    print("\nLoading Metadata...")
    train_meta = pd.read_parquet(config.TRAIN_META_PATH)
    val_meta = pd.read_parquet(config.VAL_META_PATH)
    test_meta = pd.read_parquet(config.TEST_META_PATH)

    # SUBSETTING FOR SPEED:
    # We take only 4 samples from each set to demonstrate the pipeline quickly.
    # In a real run, you would use the full dataframes.
    demo_train_df = train_meta.head(4).copy()
    demo_val_df = val_meta.head(4).copy()
    demo_test_df = test_meta.head(4).copy()

    print(f"Demo Train Size: {len(demo_train_df)}")
    print(f"Demo Val Size:   {len(demo_val_df)}")
    print(f"Demo Test Size:  {len(demo_test_df)}")

    # 3. Dataset Instantiation & Verification
    print("\nInstantiating Datasets...")

    # We use unique subset names so they cache to files like 'cached_demo_train_X.npy'
    # load_cached_data=False forces processing from scratch for this demo to prove it works
    train_dataset = RMSHDDataset(
        df=demo_train_df, subset_name="demo_train", load_cached_data=False
    )
    val_dataset = RMSHDDataset(
        df=demo_val_df, subset_name="demo_val", load_cached_data=False
    )
    test_dataset = RMSHDDataset(
        df=demo_test_df, subset_name="demo_test", load_cached_data=False
    )

    # Verification: Check shapes
    sample_img, sample_target = train_dataset[0]
    print(f"Sample Image Shape: {sample_img.shape}")
    print(f"Sample Target: {sample_target}")

    # Expected shape: (128, 224, 224) -> 32 slices * 4 modalities
    expected_shape = (128, 224, 224)
    if sample_img.shape != expected_shape:
        raise AssertionError(
            f"Dataset yielded wrong shape. Expected {expected_shape}, got {sample_img.shape}"
        )

    if not isinstance(sample_target, torch.Tensor):
        raise AssertionError("Target is not a Tensor")

    # 4. Model Initialization & Verification
    print("\nInitializing Model...")
    model = RMSHDNet()
    model.to(device)

    # Verification: Dummy Forward Pass
    # Create a batch of size 2
    dummy_input = torch.randn(2, 128, 224, 224).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")
    if dummy_output.shape != (2, 1):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, 1), got {dummy_output.shape}"
        )

    # 5. Training Loop Demonstration
    print("\nStarting Training Loop...")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # Small batch for demo
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for tiny demo
    )
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

    # Setup Training Components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    save_path = os.path.join(DEMO_WORKING_DIR, "best_model.pth")

    # Invalidate stale artifacts to prevent loading ghost models (Cite debug_lesson_8)
    if os.path.exists(save_path):
        os.remove(save_path)

    # Run Training
    # We limit to 2 epochs for speed
    trained_model = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=2,
        patience=1,
        save_path=save_path,
    )

    # 6. Inference & Submission
    print("\nRunning Inference on Test Set...")
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    predictions = predict(trained_model, test_loader, device)

    print(f"Predictions generated: {len(predictions)}")
    if len(predictions) != len(demo_test_df):
        raise AssertionError(
            "Number of predictions does not match number of test samples."
        )

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"BraTS21ID": test_dataset.get_ids(), "MGMT_value": predictions}
    )

    # Save Submission
    submission_path = "./demo_submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print("\n" + "=" * 30)
    print("DEMO COMPLETED SUCCESSFULLY")
    print(f"Submission saved to: {submission_path}")
    print("Sample Submission Content:")
    print(submission_df.head())
    print("=" * 30)


if __name__ == "__main__":
    run_demo()
