import os
import shutil
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
import library.sam as sam
import library.engine as engine


def run_demo():
    # =========================================================================
    # 1. SETUP & CONFIGURATION
    # =========================================================================
    print("--- 1. Setup & Configuration ---")

    # Override config for demo speed and isolation
    config.WORKING_DIR = "./working/demo_execution"
    config.CHECKPOINT_DIR = os.path.join(config.WORKING_DIR, "checkpoints")
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "cache")
    config.SUBMISSION_DIR = os.path.join(config.WORKING_DIR, "submission")

    # Create directories
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    config.setup_directories()

    # Set seed for reproducibility
    utils.set_seed(42)

    # Setup Logger
    logger = utils.setup_logger(os.path.join(config.WORKING_DIR, "train.log"))
    logger.info("Logger initialized.")

    device = config.DEVICE
    print(f"Device: {device}")

    # =========================================================================
    # 2. DATA LOADING & PROCESSING
    # =========================================================================
    print("\n--- 2. Data Loading & Processing ---")

    # Load data (this will process json -> npz if not cached, or load cache)
    # The provided environment likely has the raw json, so this tests process_json_data
    data_container = data.load_data(load_cached_data=False)

    stats = data_container["stats"]
    print(f"Global Stats: {stats}")

    # Verify data structure
    train_data = data_container["train"]
    assert (
        len(train_data) == 5
    ), "Train data should contain 5 elements (b1, b2, ang, lbl, ids)"
    assert train_data[0].shape[1:] == (75, 75), "Band 1 shape mismatch"

    # Get DataLoaders
    # We use a small batch size and just 1 fold for demonstration
    train_loader, val_loader = data.get_dataloaders(
        fold=0, n_folds=5, batch_size=8, mode="train_cv", load_cached_data=True
    )

    # Verify Loader Output
    images, angles, labels = next(iter(train_loader))
    print(
        f"Batch Shapes - Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions for Data
    # Images should be (B, 3, 224, 224) due to resizing in Dataset
    assert images.shape == (8, 3, 224, 224), f"Unexpected image shape: {images.shape}"
    assert angles.shape == (8,), f"Unexpected angle shape: {angles.shape}"
    assert not torch.isnan(images).any(), "Images contain NaNs"

    # =========================================================================
    # 3. MODEL INSTANTIATION
    # =========================================================================
    print("\n--- 3. Model Instantiation ---")

    net = model.IcebergResNet(
        backbone_name="resnet18",
        pretrained=False,  # Speed up init, no download needed for demo logic check
        gem_trainable=True,
    ).to(device)

    # Test Forward Pass
    dummy_images = images.to(device)
    dummy_angles = angles.to(device)

    with torch.no_grad():
        logits = net(dummy_images, dummy_angles)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (8, 1), "Model output shape mismatch"

    # =========================================================================
    # 4. OPTIMIZER (SAM) SETUP
    # =========================================================================
    print("\n--- 4. Optimizer (SAM) Setup ---")

    base_optimizer = AdamW
    optimizer = sam.SAM(net.parameters(), base_optimizer, lr=1e-3, rho=0.05)

    # Verify SAM step logic
    # We need a closure for SAM
    criterion = engine.BCEWithLogitsLossLabelSmoothing()
    dummy_labels = labels.to(device).unsqueeze(1)

    def closure():
        out = net(dummy_images, dummy_angles)
        loss = criterion(out, dummy_labels)
        loss.backward()
        return loss

    initial_loss = closure().item()
    optimizer.zero_grad()  # Clear gradients from the closure check

    # Perform actual step
    loss_after_step = optimizer.step(closure)
    print(
        f"Initial Loss: {initial_loss:.4f}, Loss returned by SAM step: {loss_after_step.item():.4f}"
    )

    # =========================================================================
    # 5. TRAINING ENGINE (FIT MODEL)
    # =========================================================================
    print("\n--- 5. Training Engine Execution ---")

    # Re-init model and optimizer for a clean training run
    net = model.IcebergResNet(pretrained=False).to(device)
    optimizer = sam.SAM(net.parameters(), AdamW, lr=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer.base_optimizer, mode="min", factor=0.5, patience=1
    )

    # Run fit_model
    # We configure it to run 2 epochs:
    # Epoch 1: Normal Phase
    # Epoch 2: SWA Phase (swa_start_epoch=2)
    # This exercises the full logic including SWA BN update.

    trained_model = engine.fit_model(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=2,
        patience=5,
        use_swa=True,
        swa_start_epoch=2,
        save_dir=config.CHECKPOINT_DIR,
        fold_idx=0,
    )

    assert os.path.exists(
        os.path.join(config.CHECKPOINT_DIR, "swa_model_0.pth")
    ), "SWA Checkpoint not found"
    print("Training loop completed successfully.")

    # =========================================================================
    # 6. INFERENCE & SUBMISSION
    # =========================================================================
    print("\n--- 6. Inference & Submission ---")

    # Load Test Data
    test_loader = data.get_dataloaders(mode="test", batch_size=8, load_cached_data=True)

    # Predict using TTA
    preds, ids = engine.predict_tta(test_loader, trained_model, device)

    print(f"Predictions Shape: {preds.shape}")
    print(f"Sample Prediction: {ids[0]} -> {preds[0]:.4f}")

    assert len(preds) == len(ids), "Mismatch between predictions and IDs"
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions out of probability range"

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": ids, "is_iceberg": preds})
    sub_path = config.SUBMISSION_PATH
    df_sub.to_csv(sub_path, index=False)

    print(f"Submission saved to {sub_path}")

    # Verify Submission File
    df_verify = pd.read_csv(sub_path)
    assert df_verify.shape[1] == 2, "Submission file should have 2 columns"
    assert (
        "id" in df_verify.columns and "is_iceberg" in df_verify.columns
    ), "Incorrect columns in submission"

    print("\n=== DEMO COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_demo()
