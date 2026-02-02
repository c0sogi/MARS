import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import provided library modules
from library import config, utils, data, model, train


def run_demo():
    # ==========================================
    # 1. SETUP & CONFIGURATION
    # ==========================================
    print("[Demo] Setting up configuration...")

    # Set reproducible seeds
    utils.seed_everything(seed=42)

    # Define working directory for this demo
    DEMO_WORKING_DIR = "./working/demo_execution"
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Override config parameters for speed
    config.WORKING_DIR = DEMO_WORKING_DIR
    config.NUM_EPOCHS = 2  # Run only 2 epochs
    config.BATCH_SIZE = 8  # Small batch size
    config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Demo] Running on device: {config.DEVICE}")

    # ==========================================
    # 2. DATA LOADING
    # ==========================================
    print("[Demo] Loading data...")

    # Load all image data (Train + Test) into memory
    # This uses the caching mechanism in library.data
    images_dict = data.load_data(load_cached_data=True)

    # Load Metadata
    # We use the pre-generated metadata files
    df_train = pd.read_csv(config.TRAIN_META_PATH)
    df_val = pd.read_csv(config.VAL_META_PATH)
    df_test = pd.read_csv(config.TEST_META_PATH)

    # Subset data for rapid demonstration
    # Keep only 32 training samples and 16 validation samples
    df_train_sub = df_train.head(32).copy()
    df_val_sub = df_val.head(16).copy()
    df_test_sub = df_test.head(16).copy()

    print(
        f"[Demo] Data loaded. Subset sizes - Train: {len(df_train_sub)}, Val: {len(df_val_sub)}"
    )

    # ==========================================
    # 3. PREPROCESSING & DATASETS
    # ==========================================
    print("[Demo] Preparing datasets...")

    # Initialize and fit the FoldScaler
    # It calculates per-channel min/max on the training set
    scaler = utils.FoldScaler()

    # Extract training images to fit the scaler
    train_ids = df_train_sub["id"].values
    train_images_list = [images_dict[i] for i in train_ids]
    train_images_arr = np.stack(train_images_list)  # Shape: (N, 3, 75, 75)

    scaler.fit(train_images_arr)

    # Verify Scaler Logic
    assert scaler.is_fitted, "Scaler should be fitted."
    assert scaler.min_vals.shape == (
        1,
        3,
        1,
        1,
    ), f"Scaler min_vals shape mismatch: {scaler.min_vals.shape}"

    # Create Datasets
    train_dataset = data.IcebergDataset(
        metadata=df_train_sub,
        images_dict=images_dict,
        scaler=scaler,
        transform=data.get_transforms("train"),
    )

    val_dataset = data.IcebergDataset(
        metadata=df_val_sub,
        images_dict=images_dict,
        scaler=scaler,
        transform=data.get_transforms("val"),  # No augmentation for val
    )

    test_dataset = data.IcebergDataset(
        metadata=df_test_sub,
        images_dict=images_dict,
        scaler=scaler,
        transform=data.get_transforms("test"),
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 for simple debugging
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ==========================================
    # 4. MODEL INITIALIZATION & VERIFICATION
    # ==========================================
    print("[Demo] Initializing IDSWNet model...")

    net = model.IDSWNet().to(config.DEVICE)

    # Verify Forward Pass
    print("[Demo] Verifying forward pass...")
    dummy_imgs, dummy_angles, dummy_targets = next(iter(train_loader))
    dummy_imgs = dummy_imgs.to(config.DEVICE)
    dummy_angles = dummy_angles.to(config.DEVICE)

    with torch.no_grad():
        output = net(dummy_imgs, dummy_angles)

    # Assertions
    assert output.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected ({config.BATCH_SIZE}, 1), got {output.shape}"
    print("[Demo] Forward pass successful.")

    # ==========================================
    # 5. TRAINING LOOP
    # ==========================================
    print("[Demo] Starting training loop...")

    # Define Loss, Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Define Early Stopping
    checkpoint_path = os.path.join(DEMO_WORKING_DIR, "model_fold_0.pth")
    early_stopping = train.EarlyStopping(patience=2, verbose=True, path=checkpoint_path)

    # Run Training
    trained_model, history = train.run_training(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,  # Skipping scheduler for short demo
        device=config.DEVICE,
        num_epochs=config.NUM_EPOCHS,
        early_stopping=early_stopping,
    )

    # Verify Training Artifacts
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."
    assert len(history["train_loss"]) > 0, "Training history is empty."
    print(
        f"[Demo] Training complete. Final Train Loss: {history['train_loss'][-1]:.4f}"
    )

    # ==========================================
    # 6. INFERENCE & SUBMISSION
    # ==========================================
    print("[Demo] Generating predictions on test set...")

    trained_model.eval()
    predictions = []
    test_ids = []

    with torch.no_grad():
        for inputs, inc_angles in test_loader:
            inputs = inputs.to(config.DEVICE)
            inc_angles = inc_angles.to(config.DEVICE)

            # Forward pass
            logits = trained_model(inputs, inc_angles)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)

            # We need IDs to map predictions. In a real loop, we'd align with the loader.
            # Here we just iterate to match the batch count, but we can get IDs from dataset.
            pass

    # Align IDs (dataset order is preserved if shuffle=False)
    test_ids = df_test_sub["id"].values

    # Create Submission DataFrame
    submission = pd.DataFrame({"id": test_ids, "is_iceberg": predictions})

    submission_path = os.path.join(DEMO_WORKING_DIR, "demo_submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"[Demo] Submission saved to {submission_path}")
    print(submission.head())

    print("\n[Demo] All steps completed successfully.")


if __name__ == "__main__":
    run_demo()
