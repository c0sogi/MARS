import os
import sys
import shutil
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import LungDataset, get_transforms
from library.network import PriorPreservingDualAxisNet
from library.engine import run_training, predict_and_submit


def create_subset_metadata(working_dir):
    """
    Creates small subsets of the metadata files to speed up the demonstration.
    """
    print("Creating metadata subsets for demonstration...")

    # Load original metadata
    train_full = pd.read_csv(Config.TRAIN_META_PATH)
    val_full = pd.read_csv(Config.VAL_META_PATH)
    test_full = pd.read_csv(Config.TEST_META_PATH)

    # Select a few patients for training (e.g., 4 patients)
    train_patients = train_full["Patient"].unique()[:4]
    train_subset = train_full[train_full["Patient"].isin(train_patients)].copy()

    # Select a few patients for validation (e.g., 2 patients)
    val_patients = val_full["Patient"].unique()[:2]
    val_subset = val_full[val_full["Patient"].isin(val_patients)].copy()

    # Select a few rows for testing (e.g., 5 rows)
    test_subset = test_full.head(5).copy()

    # Define paths
    train_path = os.path.join(working_dir, "train_subset.csv")
    val_path = os.path.join(working_dir, "val_subset.csv")
    test_path = os.path.join(working_dir, "test_subset.csv")

    # Save subsets
    train_subset.to_csv(train_path, index=False)
    val_subset.to_csv(val_path, index=False)
    test_subset.to_csv(test_path, index=False)

    print(
        f"Subsets created: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}"
    )
    return train_path, val_path, test_path


def main():
    # 1. Setup Environment
    seed_everything(42)

    # Override Config for Speed and Demo purposes
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    demo_model_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    # Clean up any previous demo runs
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 2. Prepare Data
    train_csv, val_csv, test_csv = create_subset_metadata(Config.WORKING_DIR)

    print("\n--- Initializing Datasets (Triggers Image Processing) ---")
    # This step will process DICOMs and save .npy files to Config.CACHE_DIR
    train_dataset = LungDataset(
        metadata_path=train_csv,
        mode="train",
        transform=get_transforms("train"),
        load_cached_data=True,
    )

    val_dataset = LungDataset(
        metadata_path=val_csv,
        mode="val",
        transform=get_transforms("val"),
        load_cached_data=True,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model
    print("\n--- Initializing Model ---")
    model = PriorPreservingDualAxisNet()
    model.to(Config.DEVICE)

    # Verify Model Output Shape
    # Create dummy batch to check forward pass
    dummy_axial = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(
        Config.DEVICE
    )
    dummy_coronal = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(
        Config.DEVICE
    )
    dummy_tabular = torch.randn(2, 5).to(Config.DEVICE)

    with torch.no_grad():
        dummy_out = model(dummy_axial, dummy_coronal, dummy_tabular)

    # Expected output: (Batch_Size, 3) -> [Alpha, Sigma_Base, Sigma_Growth]
    assert dummy_out.shape == (
        2,
        3,
    ), f"Model output shape mismatch. Expected (2, 3), got {dummy_out.shape}"
    print("Model forward pass verification successful.")

    # 4. Setup Training Components
    optimizer = optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 5. Run Training
    print("\n--- Starting Training Loop ---")
    best_score = run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
        patience=2,
        save_path=demo_model_path,
    )

    print(f"Training finished. Best Validation Score: {best_score}")

    # 6. Run Inference / Prediction
    print("\n--- Running Prediction on Test Subset ---")
    # Load best model weights
    model.load_state_dict(torch.load(demo_model_path))

    predict_and_submit(
        model=model,
        device=Config.DEVICE,
        metadata_path=test_csv,
        output_path=Config.SUBMISSION_PATH,
    )

    # 7. Verify Submission
    print("\n--- Verifying Submission File ---")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(submission_df.columns)}"

    # Check length
    test_df = pd.read_csv(test_csv)
    assert len(submission_df) == len(
        test_df
    ), f"Submission length mismatch. Expected {len(test_df)}, got {len(submission_df)}"

    # Check values
    # Confidence should be >= 70 (clipped in predict_and_submit)
    assert (
        submission_df["Confidence"] >= 70
    ).all(), "Found confidence values < 70 in submission."

    print("Submission verification successful.")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
