import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_class_weights, compute_metric
from library.data import get_loaders
from library.models import create_model
from library.engine import fit
from library.inference import predict_with_tta

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== Starting Library Demonstration ====")

    # 1. Setup & Configuration Override
    # We modify the Config class directly to optimize for a quick demonstration run.
    print("\n[1] Configuring environment for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 images
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.USE_SWA = False  # Disable SWA to save time
    Config.WORK_DIR = "./working/demo_test"  # Separate working dir for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory exists
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=1, BATCH_SIZE=4")

    # 2. Utility Verification
    print("\n[2] Verifying Utilities...")

    # Test calculate_class_weights
    weights = calculate_class_weights()
    assert isinstance(weights, torch.Tensor), "Weights should be a Tensor"
    assert (
        len(weights) == Config.NUM_CLASSES
    ), f"Weights length mismatch. Expected {Config.NUM_CLASSES}, got {len(weights)}"
    print(" - calculate_class_weights: OK")

    # Test compute_metric
    # Create dummy ground truth (one-hot) and predictions (probabilities)
    # 2 samples, 4 classes
    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    y_pred = np.array([[0.9, 0.05, 0.02, 0.03], [0.1, 0.8, 0.05, 0.05]])
    metric = compute_metric(y_true, y_pred)
    assert isinstance(metric, float), "Metric should be a float"
    assert 0 <= metric <= 1, "Metric should be between 0 and 1"
    print(f" - compute_metric (Dummy Test): {metric:.4f} OK")

    # 3. Data Loading Demonstration
    print("\n[3] Verifying Data Loading...")
    img_size = 224
    train_loader, val_loader, test_loader = get_loaders(
        img_size=img_size, batch_size=Config.BATCH_SIZE, debug=True
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    images = batch["image"]
    targets = batch["target"]
    ids = batch["image_id"]

    # Verify shapes
    print(f" - Batch Shapes -> Images: {images.shape}, Targets: {targets.shape}")
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        img_size,
        img_size,
    ), "Image tensor shape mismatch"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Target tensor shape mismatch"
    assert len(ids) == Config.BATCH_SIZE, "Image IDs length mismatch"
    print(" - Data Loading: OK")

    # 4. Model Instantiation
    print("\n[4] Verifying Model Creation...")
    # Using MODEL_2 (ConvNeXt-Tiny) as it is usually lighter/faster to init than some larger Efn
    model_name = Config.MODEL_2_NAME
    model = create_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(Config.DEVICE)

    # Dummy forward pass
    with torch.no_grad():
        output = model(images.to(Config.DEVICE))

    print(f" - Output Logits Shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    print(" - Model Creation & Forward Pass: OK")

    # 5. Training Loop Demonstration
    print("\n[5] Verifying Training Engine (fit)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    # Run fit (trains for 1 epoch as per config override)
    # We use a custom model name for the checkpoint file
    demo_model_name = "demo_model"
    trained_model = fit(
        model=model,
        optimizer=optimizer,
        device=Config.DEVICE,
        model_name=demo_model_name,
        debug=True,
    )

    # Check if checkpoint was saved
    expected_ckpt = os.path.join(Config.WORK_DIR, f"best_model_{demo_model_name}.pth")
    assert os.path.exists(
        expected_ckpt
    ), f"Checkpoint file not found at {expected_ckpt}"
    print(f" - Checkpoint saved successfully at: {expected_ckpt}")
    print(" - Training Loop: OK")

    # 6. Inference Demonstration
    print("\n[6] Verifying Inference (predict_with_tta)...")
    # We use the trained model and the test loader obtained earlier
    preds, pred_ids = predict_with_tta(trained_model, test_loader, Config.DEVICE)

    print(f" - Predictions Shape: {preds.shape}")
    print(f" - Number of IDs: {len(pred_ids)}")

    assert preds.shape[1] == Config.NUM_CLASSES, "Prediction columns mismatch"
    assert len(preds) == len(pred_ids), "Mismatch between predictions and IDs count"

    # Verify values are probabilities (sum to ~1 not guaranteed by logits, but softmax is applied in predict_with_tta)
    # predict_with_tta applies softmax. Sum of rows should be approx 1.
    row_sums = preds.sum(axis=1)
    assert np.allclose(
        row_sums, 1.0, atol=1e-5
    ), "Predictions do not appear to be probabilities (sum != 1)"

    print(" - Inference Logic: OK")

    print("\n==== Demonstration Complete: All checks passed ====")


if __name__ == "__main__":
    run_demo()
