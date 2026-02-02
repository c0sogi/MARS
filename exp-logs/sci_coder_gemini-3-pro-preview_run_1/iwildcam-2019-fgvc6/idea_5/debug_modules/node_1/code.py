import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
import importlib
import library.dataset
import library.engine

importlib.reload(library.dataset)
importlib.reload(library.engine)

from library.config import Config
from library.utils import seed_everything
from library.dataset import AnimalDataset, get_transforms
from library.model import MultiTaskConvNeXt
from library.loss import CompositeLoss
from library.engine import train_model, generate_submission


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_run"
    Config.IMG_SIZE = 128  # Reduce image size for faster processing
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 2
    Config.EPOCHS = 1  # Only 1 epoch for demonstration

    # Update paths based on new working dir
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.EMA_MODEL_PATH = os.path.join(Config.WORKING_DIR, "ema_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")
    Config.LOG_FILE = os.path.join(Config.WORKING_DIR, "train.log")

    # Set seed
    seed_everything(42)
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Dataset Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Dataset...")

    # Load a tiny subset of training data
    sample_size = 10
    train_ds = AnimalDataset(
        mode="train",
        transform=get_transforms("train"),
        sample_size=sample_size,
        load_cached_data=False,  # Force reload to test logic
    )

    # Check length
    assert (
        len(train_ds) == sample_size
    ), f"Dataset size mismatch. Expected {sample_size}, got {len(train_ds)}"

    # Check item structure
    item = train_ds[0]
    required_keys = ["image", "species_label", "detection_label", "id"]
    for key in required_keys:
        assert key in item, f"Missing key in dataset item: {key}"

    # Check shapes
    img = item["image"]
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {img.shape}. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE})"

    print("Dataset verification passed.")

    # ---------------------------------------------------------
    # 3. Model & Loss Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model and Loss...")

    device = torch.device(
        "cpu"
    )  # Use CPU for simple logic check to avoid GPU overhead initialization if not needed
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Instantiate Model (using pretrained=False for speed in this unit test)
    model = MultiTaskConvNeXt(pretrained=False).to(device)
    model.eval()

    # Create a dummy batch
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(
        device
    )
    dummy_species_labels = torch.randint(0, Config.NUM_CLASSES, (batch_size,)).to(
        device
    )
    dummy_detection_labels = torch.randint(0, 2, (batch_size, 1)).float().to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_images)

    # Check outputs
    assert "species_logits" in outputs
    assert "detection_logits" in outputs
    assert outputs["species_logits"].shape == (batch_size, Config.NUM_CLASSES)
    assert outputs["detection_logits"].shape == (
        batch_size,
        Config.NUM_DETECTION_CLASSES,
    )

    # Check Loss
    # Create dummy class weights
    dummy_weights = torch.ones(Config.NUM_CLASSES).to(device)
    criterion = CompositeLoss(class_weights=dummy_weights)

    targets = {
        "species_label": dummy_species_labels,
        "detection_label": dummy_detection_labels,
    }

    loss = criterion(outputs, targets)
    assert not torch.isnan(loss), "Loss returned NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("Model and Loss verification passed.")

    # ---------------------------------------------------------
    # 4. Training Loop Integration (Engine)
    # ---------------------------------------------------------
    print("\n[4] Running Training Loop (Integration Test)...")

    # We use a slightly larger sample for training loop to ensure batching works
    train_sample_size = 50

    # Run training
    # Note: engine.py imports Config. We modified Config attributes above.
    # Since engine.py uses Config.BATCH_SIZE inside functions, our changes apply.
    train_model(epochs=1, sample_size=train_sample_size)

    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not created."
    assert os.path.exists(Config.EMA_MODEL_PATH), "EMA model file was not created."

    print("Training loop completed successfully.")

    # ---------------------------------------------------------
    # 5. Submission Generation (Engine)
    # ---------------------------------------------------------
    print("\n[5] Generating Submission...")

    # Generate submission using the model we just trained
    generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Validate submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert "Id" in df_sub.columns, "Submission missing 'Id' column"
    assert "Predicted" in df_sub.columns, "Submission missing 'Predicted' column"
    assert len(df_sub) > 0, "Submission file is empty"

    print(f"Submission generated with {len(df_sub)} rows.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\nERROR: {e}")
        exit(1)
