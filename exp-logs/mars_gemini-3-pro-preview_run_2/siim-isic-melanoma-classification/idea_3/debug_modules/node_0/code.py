import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Append current directory to path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_score
from library.data import get_loaders
from library.model import ContextGatedEfficientNet
from library.engine import train_one_epoch, evaluate, predict


def main():
    print("=== Starting ISIC Task Demonstration ===")

    # 1. Configuration Overrides for Speed and Safety
    # We modify the Config class attributes directly to optimize for a quick demo run.
    print("\n[1] Configuring environment...")
    seed_everything(42)

    Config.DEBUG = True  # Use a small subset of data (head of DataFrames)
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size for speed
    Config.NUM_WORKERS = 0  # Disable multiprocessing to avoid overhead in this script
    Config.PRETRAINED = False  # Disable downloading weights to ensure offline execution
    Config.IMG_SIZE = 128  # Reduce image size for faster processing

    # Ensure working directory exists (though Config usually handles this)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print("    Config updated: DEBUG=True, EPOCHS=1, BATCH_SIZE=8, PRETRAINED=False")

    # 2. Data Loading Demonstration
    print("\n[2] Loading Data...")
    # load_cached_data=False ensures we process the debug subset fresh, rather than loading full cached .npy files
    train_loader, val_loader, test_loader = get_loaders(
        load_cached_data=False, debug=True
    )

    print(f"    Train Loader Batches: {len(train_loader)}")
    print(f"    Val Loader Batches:   {len(val_loader)}")
    print(f"    Test Loader Batches:  {len(test_loader)}")

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    (images, meta), targets = batch

    print(
        f"    Batch Shapes -> Images: {images.shape}, Meta: {meta.shape}, Targets: {targets.shape}"
    )

    # Validation assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Image tensor shape incorrect"
    assert (
        meta.ndim == 2 and meta.shape[0] == Config.BATCH_SIZE
    ), "Metadata tensor shape incorrect"
    assert targets.shape[0] == Config.BATCH_SIZE, "Target tensor shape incorrect"

    meta_dim = meta.shape[1]
    print(f"    Detected Metadata Dimension: {meta_dim}")

    # 3. Model Instantiation & Forward Pass
    print("\n[3] Initializing Model...")
    device = torch.device(Config.DEVICE)
    model = ContextGatedEfficientNet(meta_dim=meta_dim, pretrained=Config.PRETRAINED)
    model = model.to(device)
    print(f"    Model '{Config.MODEL_NAME}' created and moved to {device}")

    # Test Forward Pass
    images = images.to(device)
    meta = meta.to(device)

    with torch.no_grad():
        logits = model(images, meta)

    print(f"    Forward pass output shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, 1), "Model output logits shape mismatch"

    # 4. Training Loop Demonstration
    print("\n[4] Running Training Step...")
    # Define minimal optimizer and loss for the demo
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Run one epoch of training
    train_loss, train_auc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    print(f"    Train Epoch Result -> Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")

    # Validate metrics
    assert isinstance(train_loss, float), "Training loss must be a float"
    assert 0.0 <= train_auc <= 1.0, "Training AUC must be between 0 and 1"

    # 5. Evaluation Demonstration
    print("\n[5] Running Evaluation Step...")
    val_loss, val_auc = evaluate(model, val_loader, criterion, device)
    print(f"    Val Epoch Result   -> Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
    assert isinstance(val_loss, float), "Validation loss must be a float"

    # 6. Prediction Demonstration
    print("\n[6] Generating Predictions...")
    image_names, preds = predict(model, test_loader, device)

    print(f"    Generated {len(preds)} predictions.")
    print(f"    Sample Predictions: {preds[:5]}")

    # Validate predictions
    assert len(image_names) == len(preds), "Image names and predictions count mismatch"
    assert len(preds) == len(
        test_loader.dataset
    ), "Prediction count matches dataset size"
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions must be probabilities [0, 1]"

    # 7. Utility Function Check
    print("\n[7] Verifying Utility Functions...")
    y_true_dummy = np.array([0, 1, 0, 1])
    y_pred_dummy = np.array([0.1, 0.9, 0.2, 0.8])
    score = get_score(y_true_dummy, y_pred_dummy)
    print(f"    Dummy AUC Score: {score:.4f}")
    assert (
        score == 1.0
    ), "Utility get_score calculation incorrect for perfect predictions"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
