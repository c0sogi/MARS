import os
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.model import BHAResNet
from library.data_loader import get_dataloaders
from library.train import train_model

if __name__ == "__main__":
    print("=== Starting Demonstration of Iceberg Classification Library ===")

    # --------------------------------------------------------------------------
    # 1. Configuration and Setup
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo run...")

    # Patch Config to use a demo directory and run fast
    Config.IDEA_NAME = "demo_run"
    Config.WORKING_DIR = "./working"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, Config.IDEA_NAME)
    Config.CHECKPOINT_DIR = os.path.join(Config.CACHE_DIR, "checkpoints")

    # Ensure directories exist
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Update cache paths to point to the new demo directory
    Config.CACHE_X_TRAIN = os.path.join(Config.CACHE_DIR, "X_train.npy")
    Config.CACHE_Y_TRAIN = os.path.join(Config.CACHE_DIR, "y_train.npy")
    Config.CACHE_ANGLE_TRAIN = os.path.join(Config.CACHE_DIR, "angles_train.npy")
    Config.CACHE_IDS_TRAIN = os.path.join(Config.CACHE_DIR, "ids_train.npy")

    Config.CACHE_X_TEST = os.path.join(Config.CACHE_DIR, "X_test.npy")
    Config.CACHE_ANGLE_TEST = os.path.join(Config.CACHE_DIR, "angles_test.npy")
    Config.CACHE_IDS_TEST = os.path.join(Config.CACHE_DIR, "ids_test.npy")

    # Set Hyperparameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(Config.SEED)
    print(f"    Device: {Config.DEVICE}")
    print(f"    Cache Directory: {Config.CACHE_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # --------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Loading...")

    # Generate DataLoaders (this will process raw JSONs and cache them)
    # Note: load_cached_data=True will try to load, fail (since we cleared dir), and then process.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches: {len(val_loader)}")
    print(f"    Test Batches: {len(test_loader)}")

    # Verify Train Batch Structure
    images, angles, labels = next(iter(train_loader))

    print(f"    Sample Batch Shapes:")
    print(f"      Images: {images.shape} (Expected: [{Config.BATCH_SIZE}, 3, 75, 75])")
    print(f"      Angles: {angles.shape} (Expected: [{Config.BATCH_SIZE}])")
    print(f"      Labels: {labels.shape} (Expected: [{Config.BATCH_SIZE}])")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), "Incorrect image tensor shape"
    assert angles.shape == (Config.BATCH_SIZE,), "Incorrect angle tensor shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label tensor shape"
    assert images.dtype == torch.float32, "Images should be float32"

    # Verify Test Batch Structure (returns IDs instead of labels)
    test_images, test_angles, test_ids = next(iter(test_loader))
    assert len(test_ids) == Config.BATCH_SIZE, "Incorrect number of test IDs"
    print("    Data Loading verification successful.")

    # --------------------------------------------------------------------------
    # 3. Model Instantiation and Logic Check
    # --------------------------------------------------------------------------
    print("\n[3] Demonstrating Model Logic...")

    model = BHAResNet().to(Config.DEVICE)

    # Move sample batch to device
    images_dev = images.to(Config.DEVICE)
    angles_dev = angles.to(Config.DEVICE)

    # Forward Pass
    model.eval()
    with torch.no_grad():
        outputs = model(images_dev, angles_dev)

    print(
        f"    Model Output Shape: {outputs.shape} (Expected: [{Config.BATCH_SIZE}, 1])"
    )

    # Assertions
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert not torch.isnan(outputs).any(), "Model produced NaN values"
    print("    Model forward pass verification successful.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Simulation
    # --------------------------------------------------------------------------
    print("\n[4] Demonstrating Training Loop (1 Epoch)...")

    # Run the training function provided in library.train
    # This saves checkpoints to Config.CHECKPOINT_DIR
    train_model(fold=0, epochs=Config.NUM_EPOCHS, patience=1)

    # Verify Checkpoint Creation
    expected_checkpoint = os.path.join(Config.CHECKPOINT_DIR, "checkpoint_fold_0.pth")
    assert os.path.exists(
        expected_checkpoint
    ), f"Checkpoint not found at {expected_checkpoint}"
    print(f"    Training complete. Checkpoint verified at: {expected_checkpoint}")

    # --------------------------------------------------------------------------
    # 5. Inference and Submission Generation
    # --------------------------------------------------------------------------
    print("\n[5] Demonstrating Inference and Submission...")

    # Load the best model (or latest if best wasn't separate, but train_model saves both usually)
    # In 1 epoch, latest is best.
    checkpoint_path = expected_checkpoint

    # Re-initialize model to ensure we are loading weights into a fresh instance
    inference_model = BHAResNet().to(Config.DEVICE)
    checkpoint = load_checkpoint(checkpoint_path, inference_model)
    print(f"    Loaded checkpoint from epoch {checkpoint['epoch']}")

    inference_model.eval()
    sigmoid = torch.nn.Sigmoid()

    predictions = []
    ids_list = []

    # Run inference on a subset of test loader (just 2 batches for speed)
    print("    Running inference on test subset...")
    with torch.no_grad():
        for i, (x_test, ang_test, id_test) in enumerate(test_loader):
            if i >= 2:
                break

            x_test = x_test.to(Config.DEVICE)
            ang_test = ang_test.to(Config.DEVICE)

            logits = inference_model(x_test, ang_test)
            probs = sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)
            ids_list.extend(id_test)

    # Verify Predictions
    predictions = np.array(predictions)
    print(f"    Generated {len(predictions)} predictions.")
    print(
        f"    Prediction Range: Min={predictions.min():.4f}, Max={predictions.max():.4f}"
    )

    assert np.all(predictions >= 0.0) and np.all(
        predictions <= 1.0
    ), "Predictions out of probability range [0, 1]"

    # Create Demo Submission
    submission_df = pd.DataFrame({"id": ids_list, "is_iceberg": predictions})

    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(demo_submission_path, index=False)
    print(f"    Demo submission saved to: {demo_submission_path}")

    print("\n=== Demonstration Complete Successfully ===")
