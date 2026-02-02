import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import library modules
import library.config as config
import library.data_loader as data_loader
import library.engine as engine
import library.architectures as architectures
import library.utils as utils


def main():
    print("=== Cactus Classification Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo Speed
    # -------------------------------------------------------------------------
    print("[1] Configuring parameters for rapid demonstration...")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Device: {device}")

    # Override config values to run a fast demo
    # We must update both the config module and the modules that imported these variables
    demo_batch_size = 32
    demo_debug_size = 500  # Use 500 images for training/val/test
    demo_epochs = 2

    # Update config.py
    config.DEVICE = device
    config.BATCH_SIZE = demo_batch_size
    config.DEBUG_SAMPLE_SIZE = demo_debug_size
    config.NUM_EPOCHS = demo_epochs

    # Update data_loader.py (since it uses 'from config import ...')
    data_loader.BATCH_SIZE = demo_batch_size
    data_loader.DEBUG_SAMPLE_SIZE = demo_debug_size
    data_loader.NUM_WORKERS = 2  # Reduce workers for small demo

    # Update engine.py
    engine.DEVICE = device

    # Set seeds
    utils.set_seed(config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Loading Datasets...")

    # We force load_cached_data=False to demonstrate raw loading logic,
    # though caching is supported.
    loaders = data_loader.get_dataloaders(load_cached_data=False)

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    test_ids = loaders["test_ids"]

    # Verify Data Loading
    print("    Verifying data shapes...")
    sample_imgs, sample_lbls = next(iter(train_loader))

    # Check Batch Shape: (B, 3, 32, 32)
    expected_shape = (demo_batch_size, 3, 32, 32)
    if sample_imgs.shape != expected_shape:
        # It might be smaller if drop_last=False and it's the last batch,
        # but with drop_last=True in train_loader and 500 samples / 32 batch,
        # we should get full batches mostly.
        # However, let's assert dimensions generally.
        assert sample_imgs.ndim == 4 and sample_imgs.shape[1:] == (
            3,
            32,
            32,
        ), f"Image shape mismatch. Expected (B, 3, 32, 32), got {sample_imgs.shape}"

    assert (
        sample_lbls.ndim == 1
    ), f"Label shape mismatch. Expected (B,), got {sample_lbls.shape}"

    print(f"    Train Batch: {sample_imgs.shape}")
    print(f"    Train Labels: {sample_lbls.shape}")
    print(f"    Test IDs count: {len(test_ids)}")

    # Verify Mixup Logic
    print("    Verifying Mixup augmentation...")
    mixed_x, y_a, y_b, lam = utils.mixup_data(
        sample_imgs, sample_lbls, alpha=1.0, device="cpu"
    )
    assert mixed_x.shape == sample_imgs.shape
    assert y_a.shape == sample_lbls.shape
    assert 0 <= lam <= 1.0

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Model (WideSEResNet)...")

    # Using a smaller depth/width than default for speed
    model = architectures.WideSEResNet(
        depth=16,  # (16-4)/6 = 2 blocks per stage
        widen_factor=2,  # Smaller width
        drop_rate=0.0,
        num_classes=1,
        input_channels=3,
    ).to(device)

    # Verify Forward Pass
    print("    Verifying forward pass...")
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print("    Forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print(f"\n[4] Starting Training for {demo_epochs} epochs...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=demo_epochs)

    save_path = os.path.join(config.WORKING_DIR, "demo_best_model.pth")

    trainer = engine.Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=5,
        save_path=save_path,
    )

    best_auc = trainer.fit(num_epochs=demo_epochs)

    print(f"    Training completed. Best Validation AUC: {best_auc:.4f}")

    # Verify model file was saved
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Model checkpoint was not saved at {save_path}")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[5] Generating Submission...")

    submission_file = os.path.join(config.WORKING_DIR, "demo_submission.csv")

    # Use the trained model for inference
    # Note: generate_submission expects a list of models for ensemble
    engine.generate_submission([model], test_loader, test_ids, submission_file)

    # Verify Submission
    print("    Verifying submission file...")
    if not os.path.exists(submission_file):
        raise FileNotFoundError("Submission file not found.")

    df_sub = pd.read_csv(submission_file)

    # Check shape
    assert len(df_sub) == len(
        test_ids
    ), f"Submission rows ({len(df_sub)}) do not match test IDs ({len(test_ids)})"

    # Check columns
    expected_cols = ["id", "has_cactus"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check value range
    probs = df_sub["has_cactus"].values
    assert np.all(
        (probs >= 0) & (probs <= 1)
    ), "Found probabilities outside [0, 1] range."

    print(f"    Submission verified. Shape: {df_sub.shape}")
    print(f"    First 3 rows:\n{df_sub.head(3)}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
