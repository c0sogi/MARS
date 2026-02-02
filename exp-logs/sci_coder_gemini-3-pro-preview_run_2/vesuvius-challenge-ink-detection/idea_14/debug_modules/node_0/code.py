import os
import pandas as pd
import torch
import numpy as np
import shutil
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Import from provided library files
from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.dataset import InkDataset
from library.model import SiameseSegFormer
from library.losses import BCEDiceLoss
from library.train import train_one_epoch, validate
from library.inference import InferenceEngine

if __name__ == "__main__":
    print("--- Starting Vesuvius Ink Detection Demo ---")

    # 1. Setup Environment
    # Define paths for demo
    DEMO_WORKING_DIR = "./working/demo_execution"
    DEMO_META_DIR = os.path.join(DEMO_WORKING_DIR, "metadata")
    os.makedirs(DEMO_META_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(42)

    # 2. Prepare Demo Metadata (Subset of real metadata)
    print("Preparing demo metadata...")

    # Read original metadata
    orig_train_df = pd.read_csv("./metadata/train.csv")
    orig_test_df = pd.read_csv("./metadata/test.csv")

    # Create subsets (4 samples for train/val to allow batch_size=2)
    demo_train_df = orig_train_df.head(4).copy()
    demo_val_df = (
        orig_train_df.iloc[4:8].copy()
        if len(orig_train_df) > 4
        else orig_train_df.head(4).copy()
    )
    demo_test_df = orig_test_df.head(1).copy()

    # Save demo metadata
    demo_train_path = os.path.join(DEMO_META_DIR, "train.csv")
    demo_val_path = os.path.join(DEMO_META_DIR, "validation.csv")
    demo_test_path = os.path.join(DEMO_META_DIR, "test.csv")

    demo_train_df.to_csv(demo_train_path, index=False)
    demo_val_df.to_csv(demo_val_path, index=False)
    demo_test_df.to_csv(demo_test_path, index=False)

    # 3. Override Config for Demo
    print("Configuring parameters...")
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VALID_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path
    Config.CHECKPOINT_PATH = os.path.join(DEMO_WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = "./submission.csv"

    # Speed optimizations
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.PRETRAINED = False  # Skip downloading weights
    Config.setup()

    # 4. Verify Dataset
    print("Verifying Dataset...")
    dataset = InkDataset(demo_train_df, mode="train", load_cached_data=True)

    # Fetch one sample
    inputs, target = dataset[0]

    # Check keys
    assert "view_1" in inputs
    assert "view_2" in inputs
    assert "view_3" in inputs

    # Check shapes (Channels, H, W)
    # Config.TILE_SIZE is 512
    expected_shape = (3, 512, 512)
    assert (
        inputs["view_1"].shape == expected_shape
    ), f"View 1 shape mismatch: {inputs['view_1'].shape}"
    assert target.shape == (1, 512, 512), f"Target shape mismatch: {target.shape}"

    # Check value ranges (Normalization [0, 1])
    assert (
        inputs["view_1"].min() >= 0.0 and inputs["view_1"].max() <= 1.0
    ), "Input normalization failed"

    print("Dataset verification successful.")

    # 5. Verify Model and Loss
    print("Verifying Model and Loss...")
    device = Config.DEVICE
    model = SiameseSegFormer(num_classes=1, pretrained=False)
    model.to(device)

    # Create a batch
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch_inputs, batch_targets = next(iter(loader))

    v1 = batch_inputs["view_1"].to(device)
    v2 = batch_inputs["view_2"].to(device)
    v3 = batch_inputs["view_3"].to(device)
    targets = batch_targets.to(device)

    # Forward pass
    outputs = model(v1, v2, v3)

    # Check output shape (Batch, 1, 512, 512)
    assert outputs.shape == (
        2,
        1,
        512,
        512,
    ), f"Model output shape mismatch: {outputs.shape}"

    # Loss calculation
    criterion = BCEDiceLoss()
    loss = criterion(outputs, targets)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    print(
        f"Model forward pass and loss calculation successful. Loss: {loss.item():.4f}"
    )

    # 6. Verify Training Loop
    print("Verifying Training Loop (1 Epoch)...")
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Run one epoch
    train_loss = train_one_epoch(model, loader, optimizer, criterion, device)
    print(f"Train Loss: {train_loss:.4f}")

    # Run validation
    val_loss, val_score = validate(model, loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val F0.5: {val_score:.4f}")

    # Save Checkpoint manually to ensure it exists for inference step
    # (The training loop in train.py only saves if score > baseline, which might not happen in 1 step)
    save_checkpoint(model, optimizer, None, 1, val_score, Config.CHECKPOINT_PATH)
    assert os.path.exists(Config.CHECKPOINT_PATH), "Checkpoint file was not created."

    print("Training loop verification successful.")

    # 7. Verify Inference
    print("Verifying Inference Engine...")

    # Initialize Engine (loads the checkpoint we just saved)
    engine = InferenceEngine(checkpoint_path=Config.CHECKPOINT_PATH)

    # Run generation
    # This processes the test fragment defined in demo_test_df
    engine.generate_submission()

    # Check output
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "Id" in submission_df.columns and "Predicted" in submission_df.columns
    ), "Submission columns missing."
    assert len(submission_df) == len(demo_test_df), "Submission row count mismatch."

    print("Inference verification successful.")
    print("--- Demo Completed Successfully ---")
