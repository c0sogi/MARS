import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# Import classes and functions from the provided library files
from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.data import get_dataloaders, get_test_loader
from library.model import DRModel
from library.engine import train_one_epoch, validate, inference


def run_demonstration():
    print("=== Starting Diabetic Retinopathy Pipeline Demonstration ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration Setup
    # ------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config defaults for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples for quick execution
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_DIR = os.path.join(Config.WORKING_DIR, "models")

    # Use smaller image size for the demo to reduce memory and compute time
    demo_img_size = 224
    demo_batch_size = 4

    # Ensure working directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(42)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Image Size: {demo_img_size}")

    # ------------------------------------------------------------------------
    # 2. Data Loading Verification
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading Pipeline...")

    # Generate dataloaders
    train_loader, val_loader = get_dataloaders(
        img_size=demo_img_size, batch_size=demo_batch_size, debug=True
    )

    print(f"    Train Loader Length: {len(train_loader)}")
    print(f"    Val Loader Length: {len(val_loader)}")

    # Fetch a single batch to verify shapes and types
    images, labels = next(iter(train_loader))

    print(f"    Batch Images Shape: {images.shape}")
    print(f"    Batch Labels Shape: {labels.shape}")

    # Assertions to ensure correctness
    assert images.shape == (
        demo_batch_size,
        3,
        demo_img_size,
        demo_img_size,
    ), "Incorrect image batch shape"
    assert labels.shape == (demo_batch_size,), "Incorrect label batch shape"
    assert labels.dtype == torch.float, "Labels should be float for regression"

    # ------------------------------------------------------------------------
    # 3. Model Instantiation and Forward Pass
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    # Instantiate model
    # We use pretrained=False to avoid downloading weights during this quick demo
    model = DRModel(pretrained=False)
    model.to(device)

    # Run a dummy forward pass
    dummy_input = images.to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        demo_batch_size,
        1,
    ), "Model output should be (Batch_Size, 1)"

    # ------------------------------------------------------------------------
    # 4. Training Loop Verification
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Training Engine...")

    # Setup simple optimizer and loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Run one epoch of training
    # Note: train_one_epoch prints the loss internally
    avg_train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        accumulation_steps=1,
        use_amp=True,
    )

    print(f"    Average Train Loss: {avg_train_loss:.4f}")
    assert not np.isnan(avg_train_loss), "Training loss returned NaN"

    # ------------------------------------------------------------------------
    # 5. Validation Loop Verification
    # ------------------------------------------------------------------------
    print("\n[5] Verifying Validation Engine...")

    val_loss, val_qwk = validate(
        model=model, loader=val_loader, criterion=criterion, device=device, use_amp=True
    )

    print(f"    Validation Loss: {val_loss:.4f}")
    print(f"    Validation QWK: {val_qwk:.4f}")

    # QWK can be negative if agreement is worse than chance, but should be a valid float
    assert isinstance(val_qwk, float), "QWK score should be a float"

    # ------------------------------------------------------------------------
    # 6. Inference Verification
    # ------------------------------------------------------------------------
    print("\n[6] Verifying Inference Pipeline...")

    # Get test loader
    test_loader, test_df = get_test_loader(
        img_size=demo_img_size, batch_size=demo_batch_size, debug=True
    )

    # Run inference
    raw_predictions = inference(
        model=model, loader=test_loader, device=device, use_amp=True
    )

    print(f"    Raw Predictions Shape: {raw_predictions.shape}")
    print(f"    Test DataFrame Length: {len(test_df)}")

    assert len(raw_predictions) == len(
        test_df
    ), "Number of predictions must match number of test samples"

    # Simulate post-processing (clipping and rounding)
    final_preds = np.round(raw_predictions.clip(0, 4)).astype(int)
    print(f"    Sample Final Predictions: {final_preds[:5]}")

    # ------------------------------------------------------------------------
    # 7. Metric Logic Verification
    # ------------------------------------------------------------------------
    print("\n[7] Verifying Metric Calculation Logic...")

    # Test perfect agreement
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred_perfect = np.array([0, 1, 2, 3, 4])
    score_perfect = quadratic_weighted_kappa(y_true, y_pred_perfect)
    print(f"    Perfect Agreement Score: {score_perfect}")
    assert np.isclose(score_perfect, 1.0), "Perfect agreement should yield QWK=1.0"

    # Test random/bad agreement
    y_pred_bad = np.array([4, 3, 2, 1, 0])
    score_bad = quadratic_weighted_kappa(y_true, y_pred_bad)
    print(f"    Bad Agreement Score: {score_bad}")
    assert score_bad < 1.0, "Bad agreement should yield QWK < 1.0"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
