import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from library import config
from library import utils
from library import data
from library import model as lib_model
from library import engine
from library import inference


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup and Reproducibility
    print("\n[1] Setting random seeds...")
    utils.set_seed(42)

    # 2. Data Loading
    print("\n[2] Testing Data Loading...")
    # Use a small subset to ensure speed
    subset_size = 50
    batch_size = 8

    train_loader, val_loader, test_loader = data.get_dataloaders(
        batch_size=batch_size, debug_subset_size=subset_size
    )

    # Verify Train Loader
    images, labels = next(iter(train_loader))
    print(f"    Train batch shape: Images {images.shape}, Labels {labels.shape}")

    # Assertions for data shapes
    assert images.shape == (
        batch_size,
        3,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), "Incorrect image batch shape"
    assert labels.shape == (batch_size,), "Incorrect label batch shape"
    assert len(train_loader) > 0, "Train loader is empty"

    # Verify Test Loader (returns images and IDs)
    test_images, test_ids = next(iter(test_loader))
    print(f"    Test batch shape: Images {test_images.shape}, IDs {test_ids.shape}")
    assert test_ids.shape == (batch_size,), "Incorrect test ID batch shape"

    # 3. Model Initialization
    print("\n[3] Testing Model Initialization...")
    # Using pretrained=False for speed and to avoid network dependency in this demo
    model = lib_model.create_model(pretrained=False)
    model = model.to(config.DEVICE)

    # Verify forward pass with dummy data
    dummy_input = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model output shape: {output.shape}")
    # Expecting output shape (Batch, 1) or (Batch,) depending on timm version/config
    # The engine code expects .view(-1) to work, so we verify it can be flattened
    assert output.view(-1).shape[0] == 2, "Model output batch dimension mismatch"

    # 4. Training Loop
    print("\n[4] Testing Training Loop...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Run for 1 epoch with a small subset
    best_loss = engine.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,  # No scheduler for this short demo
        device=config.DEVICE,
        epochs=1,
        mixup_alpha=0.2,  # Test mixup logic
        patience=1,
    )

    print(f"    Training finished. Best Val Loss: {best_loss}")

    # Verify Checkpoint creation
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "model_best.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print(f"    Checkpoint verified at {checkpoint_path}")

    # 5. Inference and Submission
    print("\n[5] Testing Inference and Submission...")

    # Define a temporary output path for the demo
    demo_submission_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")

    inference.run_inference(
        checkpoint_name="model_best.pth",
        output_path=demo_submission_path,
        batch_size=batch_size,
        device=config.DEVICE,
        debug_subset_size=subset_size,
    )

    # Verify Submission File
    assert os.path.exists(demo_submission_path), "Submission file was not created"

    df = pd.read_csv(demo_submission_path)
    print(f"    Submission file loaded. Rows: {len(df)}")
    print(f"    Columns: {list(df.columns)}")

    # Assertions for submission format
    assert list(df.columns) == ["id", "label"], "Submission columns mismatch"
    assert len(df) == subset_size, f"Expected {subset_size} predictions, got {len(df)}"
    assert (
        df["id"].dtype == np.int64 or df["id"].dtype == np.int32
    ), "ID column should be integer"
    assert (
        df["label"].min() >= 0.0 and df["label"].max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
