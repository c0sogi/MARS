import os
import sys
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import numpy as np

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.model import BAMPNet
from library.train import train_epoch, validate, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo execution.
    Creates a subset of the metadata to run the pipeline quickly.
    """
    print("Setting up demo environment...")

    # 1. Define Demo Paths
    demo_root = "./working/demo_execution"
    if os.path.exists(demo_root):
        shutil.rmtree(demo_root)
    os.makedirs(demo_root, exist_ok=True)

    demo_metadata_dir = os.path.join(demo_root, "metadata")
    os.makedirs(demo_metadata_dir, exist_ok=True)

    # 2. Override Config to use this directory
    Config.WORKING_DIR = demo_root
    Config.CACHE_DIR = os.path.join(demo_root, "cache")
    Config.CHECKPOINT_DIR = os.path.join(demo_root, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(demo_root, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")

    # Create necessary dirs
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 3. Create Subset Metadata (Mini Datasets)
    # We read the original CSVs and take the top N rows
    subset_size = 5

    # Train
    orig_train = pd.read_csv(os.path.join("./metadata", "train.csv"))
    mini_train = orig_train.head(subset_size)
    mini_train_path = os.path.join(demo_metadata_dir, "train.csv")
    mini_train.to_csv(mini_train_path, index=False)
    Config.TRAIN_CSV = mini_train_path

    # Val
    orig_val = pd.read_csv(os.path.join("./metadata", "val.csv"))
    mini_val = orig_val.head(subset_size)
    mini_val_path = os.path.join(demo_metadata_dir, "val.csv")
    mini_val.to_csv(mini_val_path, index=False)
    Config.VAL_CSV = mini_val_path

    # Test
    orig_test = pd.read_csv(os.path.join("./metadata", "test.csv"))
    mini_test = orig_test.head(subset_size)
    mini_test_path = os.path.join(demo_metadata_dir, "test.csv")
    mini_test.to_csv(mini_test_path, index=False)
    Config.TEST_CSV = mini_test_path

    # 4. Override Hyperparameters for Speed
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 2
    Config.NUM_WORKERS = (
        0  # Use 0 for simple debugging/demo to avoid multiprocessing overhead
    )

    print(f"Demo environment configured at {demo_root}")
    print(f"Using subset size: {subset_size}")


def verify_data_loading():
    """
    Verifies that the data loader produces batches with correct shapes.
    """
    print("\n--- Verifying Data Loading ---")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Fetch one batch
    batch = next(iter(train_loader))

    skeleton = batch["skeleton"]
    audio = batch["audio"]
    labels = batch["labels"]
    boundaries = batch["boundaries"]
    mask = batch["mask"]
    lengths = batch["lengths"]

    print(f"Batch keys: {list(batch.keys())}")
    print(
        f"Skeleton shape: {skeleton.shape} (Expected: B, T, {Config.INPUT_DIM_SKELETON})"
    )
    print(f"Audio shape: {audio.shape} (Expected: B, T, {Config.INPUT_DIM_AUDIO})")
    print(f"Labels shape: {labels.shape} (Expected: B, T)")

    # Assertions
    assert skeleton.dim() == 3, "Skeleton should be 3D (B, T, C)"
    assert (
        skeleton.shape[2] == Config.INPUT_DIM_SKELETON
    ), f"Skeleton feature dim mismatch. Got {skeleton.shape[2]}"
    assert audio.dim() == 3, "Audio should be 3D (B, T, C)"
    assert (
        audio.shape[2] == Config.INPUT_DIM_AUDIO
    ), f"Audio feature dim mismatch. Got {audio.shape[2]}"
    assert labels.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert mask.shape == labels.shape, "Mask shape should match labels shape"

    print("Data loading verification passed.")
    return train_loader, val_loader, test_loader


def verify_model_forward(train_loader):
    """
    Verifies the model forward pass.
    """
    print("\n--- Verifying Model Forward Pass ---")
    device = torch.device(Config.DEVICE)
    model = BAMPNet().to(device)

    batch = next(iter(train_loader))
    skeleton = batch["skeleton"].to(device)
    audio = batch["audio"].to(device)
    mask = batch["mask"].to(device)
    lengths = batch[
        "lengths"
    ]  # Keep on CPU usually for pack_padded_sequence logic in model wrapper

    # Forward
    outputs = model(skeleton, audio, lengths, mask)

    logits = outputs["logits"]
    boundary = outputs["boundary"]

    print(f"Logits shape: {logits.shape} (Expected: B, T, {Config.NUM_CLASSES + 1})")
    print(f"Boundary shape: {boundary.shape} (Expected: B, T, 1)")

    # Assertions
    # Output classes = 20 gestures + 1 background = 21
    expected_classes = Config.NUM_CLASSES + 1
    assert (
        logits.shape[2] == expected_classes
    ), f"Logits class dim mismatch. Got {logits.shape[2]}"
    assert boundary.shape[2] == 1, "Boundary output dim should be 1"
    assert logits.shape[1] == skeleton.shape[1], "Temporal dimension mismatch"

    print("Model forward pass verification passed.")
    return model


def verify_training_loop(model, train_loader):
    """
    Verifies that a training step runs and returns valid loss.
    """
    print("\n--- Verifying Training Loop ---")
    device = torch.device(Config.DEVICE)

    # Setup simple optimizer/criterion
    class_weights = torch.ones(Config.NUM_CLASSES + 1).to(device)
    class_weights[Config.BACKGROUND_CLASS_ID] = 0.5
    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    criterion_boundary = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch (which is just a few batches in demo mode)
    metrics = train_epoch(
        model, train_loader, optimizer, criterion_class, criterion_boundary, device
    )

    print(f"Training Metrics: {metrics}")

    assert "loss" in metrics
    assert "class_loss" in metrics
    assert "boundary_loss" in metrics
    assert not np.isnan(metrics["loss"]), "Loss is NaN"
    assert metrics["loss"] > 0, "Loss should be positive"

    print("Training loop verification passed.")


def verify_validation(model, val_loader):
    """
    Verifies the validation logic and metric computation.
    """
    print("\n--- Verifying Validation ---")
    device = torch.device(Config.DEVICE)

    # Build GT Map from the mini validation CSV
    val_df = pd.read_csv(Config.VAL_CSV)
    gt_map = {}
    for _, row in val_df.iterrows():
        lbls = row["labels"]
        if pd.isna(lbls) or lbls == "":
            gt_map[row["sample_id"]] = []
        else:
            gt_map[row["sample_id"]] = [int(float(x)) for x in str(lbls).split(",")]

    ler = validate(model, val_loader, gt_map, device)

    print(f"Validation LER: {ler}")
    assert isinstance(ler, float), "LER should be a float"
    assert ler >= 0, "LER cannot be negative"

    print("Validation verification passed.")


def verify_submission(model, test_loader):
    """
    Verifies submission file generation.
    """
    print("\n--- Verifying Submission Generation ---")
    device = torch.device(Config.DEVICE)

    generate_submission(model, test_loader, device, Config.SUBMISSION_FILE)

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    # Check content format
    with open(Config.SUBMISSION_FILE, "r") as f:
        lines = f.readlines()
        if len(lines) > 0:
            print(f"First line of submission: {lines[0].strip()}")
            parts = lines[0].strip().split(",")
            # First part should be Sample ID (e.g., Sample00001 or Session...)
            assert (
                "Sample" in parts[0] or "Session" in parts[0]
            ), "Invalid ID format in submission"
            # Subsequent parts are integer labels
            if len(parts) > 1:
                assert parts[1].isdigit(), "Labels should be integers"

    print(f"Submission generated at {Config.SUBMISSION_FILE}")
    print("Submission verification passed.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(Config.SEED)
    setup_demo_environment()

    # 2. Data Loading
    train_loader, val_loader, test_loader = verify_data_loading()

    # 3. Model Initialization
    model = verify_model_forward(train_loader)

    # 4. Training
    verify_training_loop(model, train_loader)

    # 5. Validation
    verify_validation(model, val_loader)

    # 6. Submission
    verify_submission(model, test_loader)

    print("\nAll verification steps completed successfully.")
