import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config, seed_everything
from library.dataset import ArtworkDataset, get_transforms
from library.model import ArtworkModel
from library.training import train_model
from library.inference import run_inference
from library.utils import calculate_micro_f1, ModelEMA


def run_demo():
    # =========================================================================
    # 1. Configuration Setup for Demo
    # =========================================================================
    print(">>> Setting up Configuration for Demo...")

    # Override Project Name and Directories to isolate demo outputs
    Config.PROJECT_NAME = "demo_execution"
    Config.WORKING_DIR = os.path.join("./working", Config.PROJECT_NAME)

    # Manually update paths dependent on WORKING_DIR
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model_best.pth")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Optimization for speed and resource usage in demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.IMG_SIZE = 224  # Reduce image size for faster processing
    Config.PRETRAINED = False  # Disable downloading weights to ensure offline execution

    # Ensure directories exist
    Config.setup()

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")

    # =========================================================================
    # 2. Unit Tests
    # =========================================================================

    # --- Test Utils ---
    print("\n>>> Testing Utils...")
    # Test F1 Score with known inputs
    # True: [0, 1], [1, 0]
    # Pred: [0, 1], [1, 0]
    y_true_test = np.array([[0, 1], [1, 0]])
    y_pred_test = np.array([[0.1, 0.9], [0.8, 0.2]])  # Threshold 0.5 -> matches y_true
    f1 = calculate_micro_f1(y_pred_test, y_true_test, threshold=0.5)
    assert np.isclose(f1, 1.0), f"F1 calculation failed. Expected 1.0, got {f1}"
    print("Utils test passed.")

    # --- Test Dataset ---
    print("\n>>> Testing Dataset...")
    # Load a tiny subset of the training data
    ds = ArtworkDataset(
        mode="train",
        load_cached_data=False,
        transform=get_transforms("train"),
        data_limit=16,
    )

    assert len(ds) == 16, f"Dataset length mismatch. Expected 16, got {len(ds)}"

    # Fetch one sample
    img, target = ds[0]

    # Verify Image Shape: (Channels, Height, Width)
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected {(3, Config.IMG_SIZE, Config.IMG_SIZE)}, got {img.shape}"

    # Verify Target Shape: (Num Classes,)
    assert target.shape == (
        Config.NUM_CLASSES,
    ), f"Target shape mismatch. Expected {(Config.NUM_CLASSES,)}, got {target.shape}"

    # Verify Target Type
    assert isinstance(target, torch.Tensor), "Target is not a torch.Tensor"
    print("Dataset test passed.")

    # --- Test Model ---
    print("\n>>> Testing Model...")
    # Initialize model (without pretrained weights for speed/offline)
    model = ArtworkModel(pretrained=False)
    model.eval()

    # Create a dummy input batch: (Batch Size, Channels, Height, Width)
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Verify Output Shape: (Batch Size, Num Classes)
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(2, Config.NUM_CLASSES)}, got {output.shape}"
    print("Model test passed.")

    # --- Test Model EMA ---
    print("\n>>> Testing Model EMA...")
    ema = ModelEMA(model, decay=0.5)  # Set low decay to observe changes easily

    # Capture initial state
    param_name = list(model.state_dict().keys())[0]
    initial_model_weight = model.state_dict()[param_name].clone()
    initial_ema_weight = ema.ema.state_dict()[param_name].clone()

    # Modify the base model weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    # Update EMA
    ema.update(model)

    # Check states
    new_model_weight = model.state_dict()[param_name]
    new_ema_weight = ema.ema.state_dict()[param_name]

    # EMA should have changed from initial
    assert not torch.allclose(
        new_ema_weight, initial_ema_weight
    ), "EMA weights did not update."
    # EMA should not be exactly equal to new model (due to decay)
    assert not torch.allclose(
        new_ema_weight, new_model_weight
    ), "EMA weights should lag behind model."
    print("Model EMA test passed.")

    # =========================================================================
    # 3. Integration Tests (Training & Inference)
    # =========================================================================

    # --- Run Training Pipeline ---
    print("\n>>> Running Training Pipeline (Demo)...")
    # Execute training for 1 epoch on a small subset (32 samples)
    # This verifies the training loop, loss computation, backprop, and saving logic.
    train_model(data_limit=32, num_epochs=1)

    # Verify model artifact creation
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model file not found at {Config.MODEL_PATH}"
    print("Training pipeline finished successfully.")

    # --- Run Inference Pipeline ---
    print("\n>>> Running Inference Pipeline (Demo)...")
    # Execute inference using the model we just trained
    # We use a fixed threshold to skip the validation optimization step for speed
    run_inference(
        model_path=Config.MODEL_PATH,
        threshold=0.5,
        val_data_limit=20,
        test_data_limit=20,
        batch_size=Config.BATCH_SIZE,
    )

    # Verify submission file creation
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify submission content format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert (
        "attribute_ids" in df_sub.columns
    ), "Submission missing 'attribute_ids' column"
    assert len(df_sub) > 0, "Submission file is empty"

    print(f"Inference pipeline finished successfully. Submission shape: {df_sub.shape}")

    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
