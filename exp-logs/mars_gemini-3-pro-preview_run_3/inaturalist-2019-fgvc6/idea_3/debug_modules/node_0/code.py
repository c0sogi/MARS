import os
import torch
import pandas as pd
import numpy as np

# Import components from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data import get_loaders, get_id_map
from library.model import get_model
from library.engine import train_one_epoch, validate, predict


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # ---------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # ---------------------------------------------------------
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples for speed
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Define a temporary working directory for demo outputs
    demo_output_dir = os.path.join(Config.WORKING_DIR, "demo_output")
    os.makedirs(demo_output_dir, exist_ok=True)

    # Create a lightweight phase configuration for testing
    # We use a smaller image size and batch size to run quickly on any hardware
    demo_phase_config = Config.PHASE_1.copy()
    demo_phase_config.update(
        {
            "batch_size": 8,
            "epochs": 1,
            "img_size": 128,
            "mixup_active": True,  # Enable mixup to test that path in training
            "name": "demo_phase",
        }
    )

    print(
        f"Configuration: Device={Config.DEVICE}, Debug={Config.DEBUG}, Subset={Config.DEBUG_SUBSET_SIZE}"
    )

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[Step 1] Initializing DataLoaders...")
    train_loader, val_loader, test_loader, mixup_fn = get_loaders(demo_phase_config)

    # Verify Loaders are not empty
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Validation loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."

    # Verify Batch Structure
    images, targets = next(iter(train_loader))
    print(f" - Train Batch Shape: {images.shape}")
    print(f" - Targets Shape: {targets.shape}")

    assert images.shape[0] == demo_phase_config["batch_size"], "Batch size mismatch."
    assert images.shape[1] == 3, "Image channel count mismatch (expected 3)."
    assert images.shape[2] == demo_phase_config["img_size"], "Image height mismatch."

    # Verify ID Mapping
    id2idx, idx2id = get_id_map()
    assert (
        len(id2idx) == Config.NUM_CLASSES
    ), f"ID Map size mismatch. Expected {Config.NUM_CLASSES}, got {len(id2idx)}"
    print(" - DataLoaders and ID Mapping verified.")

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("\n[Step 2] Initializing Model...")
    # We use 'resnet18' and pretrained=False for the demo to ensure speed and
    # avoid downloading large weights during the timed run.
    model = get_model(
        num_classes=Config.NUM_CLASSES, pretrained=False, model_name="resnet18"
    )

    # Verify model device
    param_device = next(model.parameters()).device
    assert str(param_device).startswith(
        Config.DEVICE.split(":")[0]
    ), f"Model device mismatch. Expected {Config.DEVICE}"
    print(f" - Model initialized: {model.__class__.__name__} on {param_device}")

    # ---------------------------------------------------------
    # 4. Training Loop (Single Epoch)
    # ---------------------------------------------------------
    print("\n[Step 3] Running Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_metrics = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=Config.DEVICE,
        epoch=1,
        mixup_fn=mixup_fn,
    )

    # Verify Metrics
    print(f" - Train Metrics: {train_metrics}")
    assert "Loss" in train_metrics, "Loss metric missing from training output."
    assert (
        "Top1_Error" in train_metrics
    ), "Top1_Error metric missing from training output."
    assert train_metrics["Loss"]["avg"] > 0, "Training loss should be positive."

    # ---------------------------------------------------------
    # 5. Validation Loop
    # ---------------------------------------------------------
    print("\n[Step 4] Running Validation Loop...")
    val_metrics = validate(model, val_loader, Config.DEVICE)

    # Verify Metrics
    print(f" - Val Metrics: {val_metrics}")
    assert "Loss" in val_metrics, "Loss metric missing from validation output."
    assert (
        "Top1_Error" in val_metrics
    ), "Top1_Error metric missing from validation output."

    # ---------------------------------------------------------
    # 6. Inference and Submission
    # ---------------------------------------------------------
    print("\n[Step 5] Running Inference...")

    # Disable TTA for speed during demo
    Config.INFERENCE["tta"] = False
    submission_file = os.path.join(demo_output_dir, "demo_submission.csv")

    predict(model, test_loader, Config.DEVICE, output_file=submission_file)

    # Verify Submission File
    assert os.path.exists(submission_file), "Submission file was not created."

    df_sub = pd.read_csv(submission_file)
    print(f" - Submission file loaded. Shape: {df_sub.shape}")
    print(df_sub.head(3))

    assert "id" in df_sub.columns, "Submission missing 'id' column."
    assert "predicted" in df_sub.columns, "Submission missing 'predicted' column."

    # Check if number of rows matches the debug subset size
    # Note: test_loader uses the test dataframe which was sliced by DEBUG_SUBSET_SIZE
    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission row count {len(df_sub)} != Debug Subset Size {Config.DEBUG_SUBSET_SIZE}"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
