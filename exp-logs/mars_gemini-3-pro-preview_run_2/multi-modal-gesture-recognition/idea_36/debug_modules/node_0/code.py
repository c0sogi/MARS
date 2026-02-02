import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
# We need to import config first to patch it before other modules use the paths
import library.config as config
from library.utils import set_seed
from library.data_loader import GestureDataset, collate_fn
from library.model import DCHGNet
from library.loss import DCHGLoss
from library.train import Trainer
from library.inference import Predictor

# =============================================================================
# CONFIGURATION OVERRIDE FOR DEMO
# =============================================================================
# We override the working directories to isolate this run and ensure we don't
# overwrite actual experiment data.
DEMO_DIR = "./working/demo_run"
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)

config.WORKING_DIR = DEMO_DIR
config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

# Ensure directories exist
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

# Reduce hyperparameters for speed
BATCH_SIZE = 2
NUM_EPOCHS = 2
MAX_SAMPLES = 4  # Very small subset for demonstration

# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_data_loader():
    print("\n=== Testing Data Loader ===")

    # Initialize Dataset with a small subset
    # We set load_cached_data=False to verify the processing logic works from scratch
    dataset = GestureDataset(
        split="train", load_cached_data=False, max_samples=MAX_SAMPLES, augment=True
    )

    print(f"Dataset size: {len(dataset)}")
    assert len(dataset) > 0, "Dataset should not be empty."
    assert len(dataset) <= MAX_SAMPLES, f"Dataset size should be <= {MAX_SAMPLES}"

    # Test __getitem__
    sample = dataset[0]
    features = sample["features"]
    targets = sample["targets"]
    boundaries = sample["boundaries"]

    print(f"Sample feature shape: {features.shape}")
    print(f"Sample target shape: {targets.shape}")

    # Check dimensions
    # Features: (Time, InputDim=85)
    # InputDim = 72 (Skeleton) + 13 (Audio) = 85
    assert features.dim() == 2, "Features should be (T, D)"
    assert (
        features.size(1) == config.INPUT_DIM
    ), f"Feature dim should be {config.INPUT_DIM}, got {features.size(1)}"
    assert targets.dim() == 1, "Targets should be (T,)"
    assert boundaries.dim() == 1, "Boundaries should be (T,)"
    assert (
        features.size(0) == targets.size(0) == boundaries.size(0)
    ), "Time dimension mismatch"

    # Test Collate Function
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"Batch keys: {batch.keys()}")
    assert "features" in batch
    assert "mask" in batch

    b_features = batch["features"]
    b_mask = batch["mask"]

    print(f"Batch features shape: {b_features.shape}")
    print(f"Batch mask shape: {b_mask.shape}")

    assert b_features.size(0) == BATCH_SIZE or b_features.size(0) == len(
        dataset
    ), "Batch size mismatch"
    assert b_features.size(2) == config.INPUT_DIM
    assert b_mask.dtype == torch.bool

    return loader


def test_model_and_loss(loader):
    print("\n=== Testing Model and Loss ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCHGNet().to(device)
    criterion = DCHGLoss()

    # Get a batch
    batch = next(iter(loader))
    features = batch["features"].to(device)
    targets = batch["targets"].to(device)
    boundaries = batch["boundaries"].to(device)
    mask = batch["mask"].to(device)

    # Forward Pass
    outputs = model(features, mask)

    print("Model outputs keys:", outputs.keys())
    required_keys = ["stage1_cls", "stage1_bnd", "stage2_cls", "stage3_cls"]
    for k in required_keys:
        assert k in outputs, f"Missing output key: {k}"

    # Check Output Shapes
    # Logits: (B, T, NumClasses)
    s3_cls = outputs["stage3_cls"]
    print(f"Stage 3 CLS shape: {s3_cls.shape}")
    assert s3_cls.size(0) == features.size(0)
    assert s3_cls.size(1) == features.size(1)
    assert s3_cls.size(2) == config.NUM_CLASSES

    # Loss Calculation
    loss, metrics = criterion(outputs, targets, boundaries, mask)
    print(f"Calculated Loss: {loss.item()}")
    print("Metrics:", metrics)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    return model, criterion


def test_training_loop(model, criterion, train_loader):
    print("\n=== Testing Training Loop ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=config.CHECKPOINT_DIR,
    )

    # Use the same loader for val to save time
    val_loader = train_loader

    best_loss = trainer.fit(train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=2)

    print(f"Best Validation Loss: {best_loss}")

    # Verify Checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created"
    print(f"Checkpoint verified at: {checkpoint_path}")


def test_inference():
    print("\n=== Testing Inference ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    # Initialize Predictor
    predictor = Predictor(checkpoint_path, device)

    # Create a Test Loader (using val/train data for demo purposes since we don't have ground truth for test to verify easily,
    # but the code expects 'test' split to use test metadata. We will use 'val' split but treat it as inference input)
    # Note: Predictor.predict() just takes a dataloader.

    dataset = GestureDataset(
        split="val", load_cached_data=False, max_samples=MAX_SAMPLES, augment=False
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    sequences = predictor.predict(loader)

    print(f"Number of predicted sequences: {len(sequences)}")
    assert len(sequences) == len(dataset)

    print("Sample prediction:", sequences[0])
    assert isinstance(sequences[0], list), "Prediction should be a list of gesture IDs"
    # Check if all elements are ints
    if sequences[0]:
        assert isinstance(sequences[0][0], int), "Gesture IDs should be integers"

    # Generate Submission File
    output_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Mock sample IDs
    sample_ids = [f"Sample{i:05d}" for i in range(len(sequences))]

    with open(output_path, "w") as f:
        for sid, seq in zip(sample_ids, sequences):
            labels_str = ",".join(map(str, seq))
            f.write(f"{sid},{labels_str}\n")

    assert os.path.exists(output_path), "Submission file not created"
    print(f"Submission file created at: {output_path}")

    # Read back to verify
    with open(output_path, "r") as f:
        lines = f.readlines()
        print(f"Submission lines: {len(lines)}")
        print(f"First line: {lines[0].strip()}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    set_seed(42)

    try:
        # 1. Data Loading
        train_loader = test_data_loader()

        # 2. Model & Loss
        model, criterion = test_model_and_loss(train_loader)

        # 3. Training
        test_training_loop(model, criterion, train_loader)

        # 4. Inference
        test_inference()

        print("\nAll tests passed successfully!")

    except Exception as e:
        print(f"\nTest Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
