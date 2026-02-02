import os
import shutil
import torch
import numpy as np
import library.config as config
import library.utils as utils
import library.model as model
import library.data_loader as data_loader
import library.train as train


def run_demo():
    print("Initializing Demo...")

    # =========================================================================
    # 1. Configure for Speed and Demo Isolation
    # =========================================================================
    print("1. Configuring environment...")
    # Override config to run a fast debug session
    config.DEBUG = True
    config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    config.NUM_EPOCHS = 1  # Run only 1 epoch
    config.BATCH_SIZE = 8
    config.NUM_FOLDS = 2  # Reduce folds (though we only run fold 0)

    # Set up a temporary working directory for this demo
    config.WORKING_DIR = "./working/demo_usage"
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "cache")
    config.CHECKPOINT_DIR = os.path.join(config.WORKING_DIR, "checkpoints")
    config.SUBMISSION_DIR = config.WORKING_DIR
    config.SUBMISSION_FILE_PATH = os.path.join(config.WORKING_DIR, "submission.csv")

    # Clean up demo directory if it exists to ensure fresh start
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print(f"   Working Directory set to: {config.WORKING_DIR}")

    # =========================================================================
    # 2. Test Utilities
    # =========================================================================
    print("\n2. Testing Utilities...")

    # Test Seeding
    utils.seed_everything(42)
    r1 = torch.rand(5)
    utils.seed_everything(42)
    r2 = torch.rand(5)
    if not torch.equal(r1, r2):
        raise AssertionError("Seed everything failed to produce deterministic results.")
    print("   Seeding verification passed.")

    # Test Checkpointing
    dummy_model = torch.nn.Linear(10, 1)
    dummy_optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)
    state = {
        "epoch": 1,
        "state_dict": dummy_model.state_dict(),
        "optimizer": dummy_optimizer.state_dict(),
        "best_val_loss": 0.5,
    }

    # Save Checkpoint
    utils.save_checkpoint(
        state, is_best=True, fold=0, checkpoint_dir=config.CHECKPOINT_DIR
    )
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint_fold_0.pth")
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "model_best_fold_0.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint file not created.")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Best model file not created.")

    # Load Checkpoint
    loaded_checkpoint = utils.load_checkpoint(
        checkpoint_path, dummy_model, dummy_optimizer
    )
    if loaded_checkpoint["epoch"] != 1:
        raise AssertionError("Loaded checkpoint epoch mismatch.")
    if loaded_checkpoint["best_val_loss"] != 0.5:
        raise AssertionError("Loaded checkpoint loss mismatch.")
    print("   Checkpoint save/load verification passed.")

    # =========================================================================
    # 3. Test Model Architecture
    # =========================================================================
    print("\n3. Testing Model Architecture...")
    net = model.EAP_CNN()

    # Create dummy input: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(
        config.BATCH_SIZE, config.IN_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH
    )
    # Create dummy angle: (Batch,)
    dummy_angle = torch.randn(config.BATCH_SIZE)

    # Forward pass
    output = net(dummy_input, dummy_angle)

    # Check output shape: (Batch, 1)
    expected_shape = (config.BATCH_SIZE, 1)
    if output.shape != expected_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
        )
    print("   Model forward pass shape verification passed.")

    # Check backward pass
    target = torch.randn(config.BATCH_SIZE, 1)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(output, target)
    loss.backward()
    print("   Model backward pass verification passed.")

    # =========================================================================
    # 4. Test Data Loading
    # =========================================================================
    print("\n4. Testing Data Loading...")
    # This will process data from json (or cache if exists) and return loaders.
    # We force load_cached_data=False to verify the processing logic on the subset.
    train_loader, val_loader, test_loader, ids_test = data_loader.get_loaders(
        fold_index=0, load_cached_data=False
    )

    # Verify Train Loader Batch
    try:
        images, angles, labels = next(iter(train_loader))
    except StopIteration:
        raise AssertionError("Train loader is empty.")

    if images.shape != (config.BATCH_SIZE, 3, 75, 75):
        raise AssertionError(f"Train batch image shape wrong: {images.shape}")
    if angles.shape != (config.BATCH_SIZE,):
        raise AssertionError(f"Train batch angle shape wrong: {angles.shape}")
    if labels.shape != (config.BATCH_SIZE,):
        raise AssertionError(f"Train batch label shape wrong: {labels.shape}")
    print("   Train loader batch structure verified.")

    # Verify Test Loader Batch
    try:
        test_images, test_angles = next(iter(test_loader))
    except StopIteration:
        raise AssertionError("Test loader is empty.")

    if test_images.shape[1:] != (3, 75, 75):
        raise AssertionError("Test batch image dimensions wrong")
    print("   Test loader batch structure verified.")

    # =========================================================================
    # 5. Test Training Loop
    # =========================================================================
    print("\n5. Testing Training Loop (Fold 0)...")
    # This runs the full training logic for 1 epoch on the debug subset
    best_loss = train.train_fold(fold_index=0)

    if not isinstance(best_loss, float):
        raise TypeError("train_fold did not return a float loss value.")
    print(f"   Training loop completed. Best Val Loss: {best_loss:.4f}")

    # Verify checkpoint creation from training
    real_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "model_best_fold_0.pth")
    if not os.path.exists(real_checkpoint_path):
        raise FileNotFoundError("Training loop did not save the best model.")
    print("   Training checkpoint verification passed.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
