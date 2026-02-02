import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import provided library modules
from library import config
from library import utils
from library import data
from library import model
from library import engine


def main():
    # 1. Setup and Configuration
    print("Initializing demonstration...")
    config.seed_everything(config.SEED)

    # Define paths for demo subsets
    demo_train_path = os.path.join(config.WORKING_DIR, "demo_train.csv")
    demo_val_path = os.path.join(config.WORKING_DIR, "demo_val.csv")
    demo_test_path = os.path.join(config.WORKING_DIR, "demo_test.csv")

    # 2. Prepare Demo Data (Subsetting)
    # We sample a small number of rows from the actual metadata to create a fast-running demo.
    print("Preparing demo datasets...")

    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Original train metadata not found at {config.TRAIN_METADATA_PATH}"
        )

    df_train_full = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_test_full = pd.read_csv(config.TEST_METADATA_PATH)

    # Sample 32 rows for training and 16 for validation/testing
    df_train_demo = df_train_full.sample(n=32, random_state=config.SEED).reset_index(
        drop=True
    )
    df_val_demo = df_train_full.sample(n=16, random_state=config.SEED + 1).reset_index(
        drop=True
    )
    df_test_demo = df_test_full.sample(n=16, random_state=config.SEED).reset_index(
        drop=True
    )

    # Save demo metadata to working directory
    df_train_demo.to_csv(demo_train_path, index=False)
    df_val_demo.to_csv(demo_val_path, index=False)
    df_test_demo.to_csv(demo_test_path, index=False)

    print(f"Demo Train size: {len(df_train_demo)}")
    print(f"Demo Val size: {len(df_val_demo)}")
    print(f"Demo Test size: {len(df_test_demo)}")

    # 3. Test Utility Functions
    print("\nTesting utility functions...")

    # Test pF1 calculation with dummy data
    dummy_targets = np.array([1, 0, 1, 0])
    dummy_preds = np.array([0.9, 0.1, 0.8, 0.2])
    pf1_score = utils.probabilistic_f1(dummy_targets, dummy_preds)
    print(f"Dummy pF1 Score: {pf1_score:.4f}")
    assert 0.0 <= pf1_score <= 1.0, "pF1 score out of range"

    # Test Age Scaler (force recompute to verify logic)
    scaler = utils.get_age_scaler(load_cached_data=False)
    assert scaler.mean_ is not None, "Age scaler mean is None"
    print(f"Age Scaler Mean: {scaler.mean_[0]:.2f}")

    # 4. Test Data Loading
    print("\nTesting DataLoaders...")

    # Use a small batch size for the demo
    demo_batch_size = 4

    train_loader, val_loader, test_loader = data.get_dataloaders(
        train_path=demo_train_path,
        val_path=demo_val_path,
        test_path=demo_test_path,
        batch_size=demo_batch_size,
        num_workers=2,
    )

    # Verify batch structure by fetching one batch
    batch_images, batch_contra, batch_labels = next(iter(train_loader))
    print(
        f"Batch Shapes -> Target: {batch_images.shape}, Contra: {batch_contra.shape}, Labels: {batch_labels.shape}"
    )

    # Assertions for shapes: (B, 3, H, W)
    # Channels = 3 (Image + Age + Implant)
    assert batch_images.shape == (
        demo_batch_size,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
    )
    assert batch_contra.shape == (
        demo_batch_size,
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
    )
    assert batch_labels.shape == (demo_batch_size,)

    # 5. Test Model Initialization
    print("\nInitializing Model...")
    net = model.SiameseEfficientNet()
    net.to(config.DEVICE)

    # Test Forward Pass with the fetched batch
    with torch.no_grad():
        t_img = batch_images.to(config.DEVICE)
        c_img = batch_contra.to(config.DEVICE)
        logits = net(t_img, c_img)

    print(f"Forward pass output shape: {logits.shape}")
    assert logits.shape == (demo_batch_size, 1), "Output shape mismatch"

    # 6. Test Training Loop
    print("\nStarting Training Loop Demo (1 Epoch)...")

    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)

    trainer = engine.Trainer(
        model=net,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE,
        patience=1,
    )

    # Train one epoch
    train_loss, train_pf1 = trainer.train_one_epoch(train_loader, epoch_idx=1)

    # Validate (this should trigger model saving if pF1 improves from -1.0)
    val_loss, val_pf1, stop_training = trainer.validate(val_loader, epoch_idx=1)

    print(f"Training completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Check if model checkpoint was created
    assert os.path.exists(config.MODEL_SAVE_PATH), "Model checkpoint not saved"

    # 7. Test Inference and Submission
    print("\nRunning Inference...")

    # Predict uses the saved best model
    engine.predict(net, test_loader, device=config.DEVICE)

    # Check submission file
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print("Submission file loaded successfully.")
    print(df_sub.head())

    # Validate submission content
    assert config.ID_COL in df_sub.columns
    assert config.TARGET_COL in df_sub.columns
    assert len(df_sub) > 0

    print("\nAll demonstration steps passed successfully!")


if __name__ == "__main__":
    main()
