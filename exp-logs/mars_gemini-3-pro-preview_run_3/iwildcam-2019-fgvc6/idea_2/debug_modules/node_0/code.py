import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import (
    calculate_macro_f1,
    save_checkpoint,
    load_checkpoint,
    seed_everything,
)
from library.dataset import get_dataloaders
from library.model import AnimalClassifier
from library.train import run_training
from library.predict import generate_submission


def main():
    print("=== Starting Demonstration of Animal Classification Library ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Fast Demonstration
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid testing...")

    # Create a separate working directory for this demo to avoid conflicts
    DEMO_WORKING_DIR = "./working/demo_run"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Override Config values
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.MODEL_CHECKPOINT_PATH = os.path.join(DEMO_WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_WORKING_DIR, "demo_submission.csv")

    Config.DEBUG = True  # Use subset of data
    Config.DEBUG_SAMPLE_SIZE = 100  # Small sample size for speed
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_EPOCHS = 1  # Only 1 epoch
    Config.PRETRAINED = False  # Skip downloading weights for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # ------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions (library.utils)...")

    # Test F1 Score
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    f1 = calculate_macro_f1(y_true, y_pred)
    assert f1 == 1.0, f"Expected F1 1.0, got {f1}"

    y_pred_wrong = np.array([1, 2, 0, 1, 2, 0])
    f1_wrong = calculate_macro_f1(y_true, y_pred_wrong)
    assert f1_wrong < 1.0, "F1 should be less than 1.0 for incorrect predictions"
    print("    calculate_macro_f1: Passed")

    # Test Checkpointing
    dummy_state = {"state_dict": {}}
    save_checkpoint(dummy_state, is_best=True, filename="test_ckpt.pth")
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "test_ckpt.pth")
    ), "Checkpoint file not created"
    assert os.path.exists(Config.MODEL_CHECKPOINT_PATH), "Best model copy not created"
    print("    save_checkpoint: Passed")

    # ------------------------------------------------------------------------
    # 3. Verify Dataset and DataLoaders
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Dataset and DataLoaders (library.dataset)...")

    train_loader, val_loader, test_loader = get_dataloaders()

    # Verify Train Loader
    assert len(train_loader) > 0, "Train loader is empty"
    images, targets, ids = next(iter(train_loader))

    # Check shapes
    expected_shape = (Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    assert (
        images.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {images.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Target shape mismatch. Expected ({Config.BATCH_SIZE},), got {targets.shape}"
    assert len(ids) == Config.BATCH_SIZE, "ID list length mismatch"

    # Check types
    assert isinstance(images, torch.Tensor), "Images should be a Tensor"
    assert isinstance(targets, torch.Tensor), "Targets should be a Tensor"

    print(f"    Train Batch Shape: {images.shape}")
    print(f"    Target Batch Shape: {targets.shape}")
    print("    get_dataloaders: Passed")

    # ------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture (library.model)...")

    model = AnimalClassifier(pretrained=False)
    model.to(Config.DEVICE)
    model.eval()

    # Run forward pass with the batch fetched earlier
    with torch.no_grad():
        images = images.to(Config.DEVICE)
        outputs = model(images)

    expected_output_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)
    assert (
        outputs.shape == expected_output_shape
    ), f"Model output shape mismatch. Expected {expected_output_shape}, got {outputs.shape}"

    print(f"    Model Output Shape: {outputs.shape}")
    print("    AnimalClassifier Forward Pass: Passed")

    # ------------------------------------------------------------------------
    # 5. Verify Training Loop
    # ------------------------------------------------------------------------
    print("\n[5] Running Training Loop Integration (library.train)...")

    # run_training handles the loop, validation, and saving
    # We set epochs=1 and debug=True, so this should be fast
    run_training(num_epochs=Config.NUM_EPOCHS, patience=1)

    # Verify artifacts exist
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not saved after training"
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), "Submission file was not generated after training"

    print("    run_training execution: Passed")

    # ------------------------------------------------------------------------
    # 6. Verify Prediction/Inference
    # ------------------------------------------------------------------------
    print("\n[6] Verifying Prediction Output (library.predict)...")

    # Load the generated submission
    df_submission = pd.read_csv(Config.SUBMISSION_PATH)

    # Verify columns
    assert "Id" in df_submission.columns, "Submission missing 'Id' column"
    assert "Predicted" in df_submission.columns, "Submission missing 'Predicted' column"

    # Verify rows (Should match the debug sample size for test set)
    # Note: DEBUG_SAMPLE_SIZE is applied to test set in get_dataloaders
    # We requested 100 samples, but actual size is min(len(df), 100).
    # Since test.csv has ~16k rows, it should be exactly 100.
    assert (
        len(df_submission) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission length mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_submission)}"

    # Verify values
    assert (
        df_submission["Predicted"].dtype == np.int64
        or df_submission["Predicted"].dtype == np.int32
    ), "Predicted column should be integers"

    print(f"    Submission Shape: {df_submission.shape}")
    print("    generate_submission: Passed")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
