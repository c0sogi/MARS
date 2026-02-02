import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Import from the provided library
from library.config import (
    set_seed,
    DEVICE,
    BATCH_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INPUT_CHANNELS,
    WORKING_DIR,
    SUBMISSION_PATH,
)
from library.model import load_and_process_data, IcebergDataset, HRDSNet
from library.utils import EarlyStopping
from library.train import train_one_epoch, validate, predict_test


def run_demo():
    print("==========================================")
    print("   Iceberg Classification Demo Script")
    print("==========================================")

    # 1. Setup
    print(f"\n[1] Setting up environment...")
    set_seed(42)
    print(f"    Device: {DEVICE}")
    print(f"    Working Directory: {WORKING_DIR}")

    # 2. Data Processing
    print(f"\n[2] Loading and Processing Data...")
    # We force load_cached_data=True to use existing cache if available, or process if not.
    # The provided environment likely has raw data, so this will process it.
    X_train, y_train, inc_train, X_test, inc_test, test_ids = load_and_process_data(
        load_cached_data=True
    )

    print(f"    X_train shape: {X_train.shape}")
    print(f"    y_train shape: {y_train.shape}")
    print(f"    X_test shape:  {X_test.shape}")

    # Validation: Check shapes
    assert (
        len(X_train) == len(y_train) == len(inc_train)
    ), "Train data dimension mismatch"
    assert X_train.shape[1:] == (
        INPUT_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    ), f"Expected image shape (3, 75, 75), got {X_train.shape[1:]}"
    assert not np.isnan(X_train).any(), "X_train contains NaNs"
    print("    -> Data integrity checks passed.")

    # 3. Dataset & DataLoader
    print(f"\n[3] Testing IcebergDataset...")
    # Create a small subset for demonstration
    subset_indices = list(range(10))
    demo_ds = IcebergDataset(
        X_train[subset_indices],
        y_train[subset_indices],
        inc_train[subset_indices],
        transform=True,
    )

    # Test __getitem__
    img, inc, lbl = demo_ds[0]
    print(f"    Sample Item - Img Shape: {img.shape}, Inc: {inc}, Label: {lbl}")

    assert img.shape == (
        INPUT_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    ), "Dataset returned wrong image shape"
    assert isinstance(inc, torch.Tensor), "Incidence angle should be a tensor"
    assert isinstance(lbl, torch.Tensor), "Label should be a tensor"
    print("    -> Dataset __getitem__ works correctly.")

    # 4. Model Architecture
    print(f"\n[4] Initializing HRDSNet Model...")
    model = HRDSNet().to(DEVICE)

    # Create a dummy batch
    dummy_loader = DataLoader(demo_ds, batch_size=4, shuffle=False)
    imgs, incs, lbls = next(iter(dummy_loader))
    imgs, incs = imgs.to(DEVICE), incs.to(DEVICE)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(imgs, incs)

    print(f"    Output shape: {outputs.shape}")
    assert outputs.shape == (4, 1), f"Expected output shape (4, 1), got {outputs.shape}"
    print("    -> Model forward pass successful.")

    # 5. Early Stopping Utility
    print(f"\n[5] Testing EarlyStopping Utility...")
    # Mock a dummy model and path
    dummy_model = nn.Linear(10, 1)
    chk_path = os.path.join(WORKING_DIR, "demo_checkpoint.pth")

    # Patience = 2 for quick test
    es = EarlyStopping(patience=2, verbose=False, path=chk_path)

    # Simulate loss: 0.5 -> 0.4 (improve) -> 0.45 (worse) -> 0.46 (worse) -> Stop
    losses = [0.5, 0.4, 0.45, 0.46]
    stop_triggered = False

    for i, loss in enumerate(losses):
        es(loss, dummy_model)
        status = "Improved" if loss == es.val_loss_min else "No Improvement"
        print(f"    Step {i}: Loss {loss} -> {status}, Counter: {es.counter}")
        if es.early_stop:
            stop_triggered = True
            print("    -> Early stopping triggered!")
            break

    assert os.path.exists(chk_path), "Checkpoint file was not created"
    assert stop_triggered, "Early stopping should have triggered"
    print("    -> EarlyStopping logic verified.")

    # 6. Training Loop Integration (Speed Optimized)
    print(f"\n[6] Running Mini-Training Loop...")

    # Create tiny subsets for speed
    train_sub = Subset(
        IcebergDataset(X_train, y_train, inc_train, transform=True), range(32)
    )
    val_sub = Subset(
        IcebergDataset(X_train, y_train, inc_train, transform=False), range(32, 64)
    )

    train_loader = DataLoader(train_sub, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=8, shuffle=False)

    # Re-init model and optimizer
    model = HRDSNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Run for 2 epochs
    for epoch in range(2):
        print(f"    Epoch {epoch+1}/2...")
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_acc = validate(model, val_loader, criterion, DEVICE)
        print(
            f"        Train Loss: {t_loss:.4f}, Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}"
        )

        assert not np.isnan(t_loss), "Training loss is NaN"
        assert not np.isnan(v_loss), "Validation loss is NaN"

    print("    -> Training loop executed successfully.")

    # 7. Inference
    print(f"\n[7] Testing Inference on Test Subset...")
    # Use first 10 test images
    X_test_sub = X_test[:10]
    inc_test_sub = inc_test[:10]
    ids_test_sub = test_ids[:10]

    preds = predict_test(model, X_test_sub, inc_test_sub, DEVICE)

    print(f"    Predictions shape: {preds.shape}")
    print(f"    Sample predictions: {preds[:3]}")

    assert len(preds) == 10, "Prediction count mismatch"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    # Generate dummy submission
    df_sub = pd.DataFrame({"id": ids_test_sub, "is_iceberg": preds})
    demo_sub_path = os.path.join(WORKING_DIR, "demo_submission.csv")
    df_sub.to_csv(demo_sub_path, index=False)
    print(f"    -> Demo submission saved to {demo_sub_path}")

    print("\n==========================================")
    print("   Demo Completed Successfully")
    print("==========================================")


if __name__ == "__main__":
    run_demo()
