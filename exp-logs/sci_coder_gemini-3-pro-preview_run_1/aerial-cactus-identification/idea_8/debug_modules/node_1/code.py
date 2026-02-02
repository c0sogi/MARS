import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library components
from library.dataset import (
    load_and_cache_data,
    CactusDataset,
    get_transforms,
    set_seed,
    INPUT_DIR,
    METADATA_DIR,
)
from library.model import CactusRepVGG
from library.optimizer import SAM
from library.engine import train_one_epoch, validate_one_epoch
from library.inference import generate_submission

# Constants for the demo
WORKING_DIR = "./working"
DEMO_MODEL_PATH = os.path.join(WORKING_DIR, "demo_model.pth")
SUBMISSION_PATH = os.path.join(WORKING_DIR, "demo_submission.csv")
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_demo():
    print(f"Running demo on device: {DEVICE}")

    # 1. Setup
    set_seed(42)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Data Loading Demo
    print("\n--- Data Loading Demo ---")
    train_meta_path = os.path.join(METADATA_DIR, "train_metadata.csv")

    # Load data (using cache if available, or creating it)
    # We load everything then slice for speed
    imgs, labels, ids = load_and_cache_data(train_meta_path, "train_demo")

    # Create a small subset for speed (128 images)
    subset_size = 128
    imgs_subset = imgs[:subset_size]
    labels_subset = labels[:subset_size]

    print(f"Loaded subset shape: {imgs_subset.shape}")

    # Instantiate Dataset
    train_dataset = CactusDataset(
        imgs_subset, labels_subset, transform=get_transforms("train")
    )

    # Instantiate DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,  # Ensure consistent batch sizes for this demo
    )

    # Verify Data Shapes
    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"Batch Image Shape: {batch_imgs.shape}")  # Expected: (32, 3, 32, 32)
    print(f"Batch Label Shape: {batch_labels.shape}")  # Expected: (32,)

    assert batch_imgs.shape == (BATCH_SIZE, 3, 32, 32), "Incorrect image batch shape"
    assert batch_labels.shape == (BATCH_SIZE,), "Incorrect label batch shape"
    assert batch_imgs.dtype == torch.float32, "Images should be float tensors"

    # 3. Model Architecture Verification
    print("\n--- Model Architecture Verification ---")
    model = CactusRepVGG(num_classes=1, deploy=False).to(DEVICE)
    model.train()  # Training mode enables Aux head

    dummy_input = torch.randn(2, 3, 32, 32).to(DEVICE)

    # Forward pass in training mode
    main_out, aux_out = model(dummy_input)
    print(f"Training Output Shapes - Main: {main_out.shape}, Aux: {aux_out.shape}")

    assert main_out.shape == (2, 1), "Main output shape mismatch"
    assert aux_out.shape == (2, 1), "Aux output shape mismatch"

    # Verify RepVGG Block Fusion (Switch to Deploy)
    print("Testing switch_to_deploy()...")
    # Count parameters before fusion
    params_before = sum(p.numel() for p in model.parameters())

    model.eval()
    model.switch_to_deploy()

    # Count parameters after fusion (should be fewer due to merging branches)
    params_after = sum(p.numel() for p in model.parameters())
    print(f"Params before: {params_before}, Params after: {params_after}")

    assert params_after < params_before, "Parameter count did not decrease after fusion"

    # Forward pass in deploy mode (returns only main output)
    deploy_out = model(dummy_input)
    assert deploy_out.shape == (2, 1), "Deploy output shape mismatch"

    # 4. Training Loop Demo
    print("\n--- Training Loop Demo ---")
    # Re-instantiate model for training (since we fused the previous one)
    model = CactusRepVGG(num_classes=1, deploy=False).to(DEVICE)

    # Setup SAM Optimizer
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.01, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()

    # Train one epoch
    print("Training for 1 epoch...")
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE, epoch=1, mixup_alpha=0.2
    )

    assert not np.isnan(train_loss), "Training loss is NaN"
    assert train_loss > 0, "Training loss should be positive"

    # Validate
    print("Validating...")
    # Create a small validation loader
    val_dataset = CactusDataset(
        imgs[subset_size : subset_size + 64],  # Next 64 images
        labels[subset_size : subset_size + 64],
        transform=get_transforms("val"),
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    val_metrics = validate_one_epoch(model, val_loader, criterion, DEVICE)
    print(f"Validation Metrics: {val_metrics}")

    assert "val_loss" in val_metrics and "val_auc" in val_metrics

    # Save Model
    print(f"Saving model to {DEMO_MODEL_PATH}...")
    torch.save(model.state_dict(), DEMO_MODEL_PATH)
    assert os.path.exists(DEMO_MODEL_PATH)

    # 5. Inference Demo
    print("\n--- Inference Demo ---")
    test_meta_path = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Run generation with a limit on samples to be fast
    generate_submission(
        model_paths=[DEMO_MODEL_PATH],
        output_file=SUBMISSION_PATH,
        metadata_path=test_meta_path,
        device=DEVICE.type,
        batch_size=BATCH_SIZE,
        load_cached_data=True,
        num_samples=50,  # Only predict first 50 samples for demo
    )

    # Verify Submission
    if os.path.exists(SUBMISSION_PATH):
        df = pd.read_csv(SUBMISSION_PATH)
        print(f"Submission generated with {len(df)} rows.")
        print(df.head())

        assert (
            len(df) == 50
        ), "Submission should have 50 rows (as requested by num_samples)"
        assert "id" in df.columns and "has_cactus" in df.columns
        assert df["has_cactus"].dtype == float
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
