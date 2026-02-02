import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library components
import library.config as config
import library.utils as utils
import library.data_processing as data_processing
import library.dataset as dataset_module
import library.model as model_module
import library.train as train_module
import library.inference as inference_module


def run_demo():
    print("=== Starting Glioblastoma Subtype Prediction Demo ===")

    # 1. Setup & Configuration Overrides for Speed
    # ---------------------------------------------
    utils.seed_everything(42)

    # Define a temporary working directory for this demo
    demo_dir = os.path.join(config.WORKING_DIR, "demo_execution_custom")
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Working directory: {demo_dir}")

    # Create mini metadata files to speed up processing (subsetting to 4 samples)
    # This prevents the Dataset class from trying to process 500+ subjects
    print("Creating mini datasets for rapid verification...")

    # Train Metadata Subset
    full_train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    mini_train_df = full_train_df.head(4).copy()
    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_train_df.to_csv(mini_train_path, index=False)

    # Test Metadata Subset
    full_test_df = pd.read_csv(config.TEST_METADATA_PATH)
    mini_test_df = full_test_df.head(4).copy()
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")
    mini_test_df.to_csv(mini_test_path, index=False)

    # Monkey-patch the paths in library.data_processing to point to our mini files
    # The BraTSDataset calls get_processed_dataframe in data_processing, which uses these variables.
    data_processing.TRAIN_METADATA_PATH = mini_train_path
    data_processing.TEST_METADATA_PATH = mini_test_path

    # Redirect cache to a fresh temp dir to ensure we verify the computation logic
    # and don't rely on pre-existing large caches.
    data_processing.CACHE_DIR = os.path.join(demo_dir, "cache")
    os.makedirs(data_processing.CACHE_DIR, exist_ok=True)

    # 2. Dataset Verification
    # -----------------------
    print("\n[Step 1/5] Verifying Dataset Logic...")

    # Instantiate dataset (this triggers stats computation for the 4 samples)
    train_ds = dataset_module.BraTSDataset(
        split="train", transform=dataset_module.get_transforms("train")
    )

    # Verify length
    print(f"  Dataset Length: {len(train_ds)}")
    assert len(train_ds) == 4, f"Expected 4 samples, got {len(train_ds)}"

    # Verify Item Loading
    img, label = train_ds[0]
    print(f"  Sample Image Shape: {img.shape}")
    print(f"  Sample Label: {label}")

    # Assertions
    # Expected shape: (9 channels, 224, 224)
    assert img.shape == (
        9,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), f"Expected shape (9, {config.IMG_SIZE}, {config.IMG_SIZE}), got {img.shape}"
    assert isinstance(label, torch.Tensor), "Label must be a torch.Tensor"

    # Create DataLoader
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)

    # 3. Model Verification
    # ---------------------
    print("\n[Step 2/5] Verifying Model Architecture...")
    device = config.DEVICE
    model = model_module.CASIVNet().to(device)

    # Fetch a batch
    batch_imgs, batch_labels = next(iter(train_loader))
    batch_imgs = batch_imgs.to(device)

    # Forward Pass
    with torch.no_grad():
        logits = model(batch_imgs)

    print(f"  Batch Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (2, 1), f"Expected logits shape (2, 1), got {logits.shape}"

    # 4. Training Loop Verification
    # -----------------------------
    print("\n[Step 3/5] Verifying Training & Validation Loop...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Run one epoch of training
    avg_loss = train_module.train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    print(f"  Train Loss (1 epoch): {avg_loss:.4f}")
    assert avg_loss > 0, "Training loss should be positive"

    # Run validation
    val_loss, val_auc = train_module.validate(model, train_loader, criterion, device)
    print(f"  Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
    assert val_loss > 0, "Validation loss should be positive"

    # 5. Inference Pipeline Verification
    # ----------------------------------
    print("\n[Step 4/5] Verifying Inference & Submission Generation...")

    # Create a dummy model directory to simulate a trained ensemble
    model_dir = os.path.join(demo_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Save the current model state as if it were the best model for all 5 folds
    state_dict = model.state_dict()
    for i in range(config.N_FOLDS):
        save_path = os.path.join(model_dir, f"best_model_fold{i}.pth")
        torch.save({"state_dict": state_dict}, save_path)

    print(f"  Saved dummy ensemble checkpoints to {model_dir}")

    submission_path = os.path.join(demo_dir, "submission.csv")

    # Run the inference function
    # This uses the patched TEST_METADATA_PATH (mini_test.csv)
    inference_module.predict_test_set(
        model_dir=model_dir,
        output_path=submission_path,
        device=device,
        batch_size=2,
        num_workers=0,  # Avoid multiprocessing overhead for demo
    )

    # 6. Submission Verification
    # --------------------------
    print("\n[Step 5/5] Verifying Submission File...")

    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    print(f"  Submission Rows: {len(sub_df)}")
    print(sub_df.head())

    # Assertions
    assert len(sub_df) == 4, "Submission should have 4 rows (matching mini test set)"
    assert "BraTS21ID" in sub_df.columns, "Missing BraTS21ID column"
    assert "MGMT_value" in sub_df.columns, "Missing MGMT_value column"
    assert (
        sub_df["MGMT_value"].between(0, 1).all()
    ), "Probabilities must be between 0 and 1"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
