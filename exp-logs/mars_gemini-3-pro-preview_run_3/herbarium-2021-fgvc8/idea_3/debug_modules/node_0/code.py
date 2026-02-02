import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the path for imports
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, calculate_f1_macro
from library.dataset import HerbariumDataset, get_dataloaders
from library.model import HerbariumConvNeXt
from library.train import run_training
from library.predict import run_prediction

if __name__ == "__main__":
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Configuring environment for fast demonstration...")
    seed_everything(42)

    # Override Config for a quick demo run
    Config.WORKING_DIR = "./working/demo_run"
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.EPOCHS = 2
    Config.SWA_START_EPOCH = 1  # Trigger SWA logic quickly
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model_best.pth")
    Config.SWA_MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model_swa.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Re-run setup to create the new working directory
    Config.setup()

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Functions...")
    y_true = [0, 1, 2, 0, 1]
    y_pred_perfect = [0, 1, 2, 0, 1]
    y_pred_bad = [0, 0, 0, 0, 0]

    f1_perfect = calculate_f1_macro(y_true, y_pred_perfect)
    f1_bad = calculate_f1_macro(y_true, y_pred_bad)

    assert f1_perfect == 1.0, f"Expected F1 1.0, got {f1_perfect}"
    assert f1_bad < 1.0, f"Expected F1 < 1.0, got {f1_bad}"
    print("Utility logic verified.")

    # 3. Verify Dataset and DataLoaders
    print("\n[3] Verifying Dataset and DataLoaders...")
    # Load metadata manually for explicit dataset testing
    train_df = pd.read_csv(Config.TRAIN_CSV).iloc[: Config.DEBUG_SAMPLE_SIZE]
    val_df = pd.read_csv(Config.VAL_CSV).iloc[: Config.DEBUG_SAMPLE_SIZE]
    test_df = pd.read_csv(Config.TEST_CSV).iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Instantiate Dataset
    dataset = HerbariumDataset(train_df, transforms=None, mode="train")

    # Test __getitem__
    img, label = dataset[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Label: {label}")

    # Assertions
    assert isinstance(
        img, np.ndarray
    ), "Image should be a numpy array (before transform)"
    assert (
        img.shape == (1000, 666, 3) or img.shape[2] == 3
    ), f"Unexpected image shape: {img.shape}"
    assert isinstance(label, torch.Tensor), "Label should be a torch tensor"

    # Test DataLoaders (this uses the library function which applies transforms)
    train_loader, val_loader, test_loader = get_dataloaders(
        pd.read_csv(Config.TRAIN_CSV),
        pd.read_csv(Config.VAL_CSV),
        pd.read_csv(Config.TEST_CSV),
        load_cached_data=False,  # Force recompute for demo safety
    )

    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"Batch Image Shape: {batch_imgs.shape}")
    print(f"Batch Label Shape: {batch_labels.shape}")

    assert batch_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect batch image shape"
    assert batch_labels.shape == (Config.BATCH_SIZE,), "Incorrect batch label shape"
    print("Dataset and DataLoaders verified.")

    # 4. Verify Model Architecture
    print("\n[4] Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = HerbariumConvNeXt(
        pretrained=False
    )  # False for speed, we just check architecture
    model.to(device)
    model.eval()

    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected output (2, {Config.NUM_CLASSES}), got {output.shape}"
    print("Model architecture verified.")

    # 5. Run Full Training Pipeline
    print("\n[5] Running Training Pipeline (Debug Mode)...")
    # This will train for 2 epochs on 50 samples
    # It tests: train loop, validation, SWA, saving
    run_training(debug=True, epochs=Config.EPOCHS, load_cached_data=False)

    # Verify outputs
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Best model file not found"
    assert os.path.exists(Config.SWA_MODEL_SAVE_PATH), "SWA model file not found"
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    submission = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Generated submission with {len(submission)} rows.")
    assert (
        len(submission) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} predictions, got {len(submission)}"
    print("Training pipeline verified.")

    # 6. Run Prediction/Inference Pipeline
    print("\n[6] Running Prediction Pipeline...")
    # Delete previous submission to ensure this run generates a new one
    os.remove(Config.SUBMISSION_PATH)

    # Run prediction (loads SWA model by default if present)
    run_prediction(debug=True, load_cached_data=False)

    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), "Prediction pipeline failed to generate submission"
    submission_new = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(submission_new) == Config.DEBUG_SAMPLE_SIZE
    print("Prediction pipeline verified.")

    print("\n=== Demonstration Completed Successfully ===")
