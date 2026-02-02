import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Import classes and functions from the provided library files
from library.config import Config
from library.utils import set_seed, probabilistic_f1
from library.data import get_dataloaders
from library.modules import SiameseFPNModel
from library.train import run_training
from library.predict import inference_fn


def demo_pipeline():
    print("===========================================================")
    print("   Breast Cancer Detection: Library Usage Demonstration    ")
    print("===========================================================")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    # We modify the Config class attributes to create a fast, lightweight
    # execution environment for demonstration purposes.
    print("\n[1] Configuring Environment for Demo...")

    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples per split
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.IMG_SIZE = (256, 256)  # Reduced resolution for speed

    # Redirect outputs to the working directory to ensure write permissions
    Config.WORKING_DIR = "./working"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure clean state for demo directories
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Image Size: {Config.IMG_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test Reproducibility
    set_seed(42)

    # Test Probabilistic F1 Score
    # Case 1: Perfect prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    pf1_perfect = probabilistic_f1(y_true, y_pred_perfect)
    assert np.isclose(pf1_perfect, 1.0), f"Expected pF1=1.0, got {pf1_perfect}"

    # Case 2: Mixed prediction
    y_pred_mixed = np.array([0.8, 0.2, 0.6, 0.4])
    pf1_mixed = probabilistic_f1(y_true, y_pred_mixed)
    print(f"    pF1 Score Test (Mixed): {pf1_mixed:.4f}")
    assert 0.0 <= pf1_mixed <= 1.0, "pF1 score outside valid range [0, 1]"

    print("    Utils verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Verify Data Pipeline
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Loading & Processing...")

    # Generate dataloaders. load_cached_data=False forces the processing logic to run.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Validate Train Loader Batch
    try:
        batch = next(iter(train_loader))
        target_img, contra_img, labels = batch

        print(
            f"    Batch Shapes -> Target: {target_img.shape}, Contra: {contra_img.shape}, Labels: {labels.shape}"
        )

        # Assertions for shapes
        expected_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
        assert (
            target_img.shape == expected_shape
        ), f"Target image shape mismatch. Expected {expected_shape}, got {target_img.shape}"
        assert (
            contra_img.shape == expected_shape
        ), f"Contra image shape mismatch. Expected {expected_shape}, got {contra_img.shape}"
        assert labels.shape == (
            Config.BATCH_SIZE,
        ), f"Labels shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"

        # Assertions for content (Channel 0 is Image, should be normalized)
        # Note: Channels 1 (Age) and 2 (Implant) might be outside [0,1] depending on normalization stats
        assert (
            target_img[:, 0].max() <= 1.0 + 1e-5
        ), "Image channel contains values > 1.0"
        assert (
            target_img[:, 0].min() >= 0.0 - 1e-5
        ), "Image channel contains values < 0.0"

        print("    Data pipeline verified successfully.")

    except Exception as e:
        print(f"    [ERROR] Data verification failed: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Siamese FPN Model...")

    device = Config.DEVICE
    model = SiameseFPNModel().to(device)

    # Move batch to device
    t_img = target_img.to(device)
    c_img = contra_img.to(device)

    # Forward Pass
    with torch.no_grad():
        logits = model(t_img, c_img)

    print(f"    Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {logits.shape}"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    print("    Model architecture verified successfully.")

    # -------------------------------------------------------------------------
    # 5. Integration Test: Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (Integration Test)...")
    print("    Executing 'run_training()' with reduced epochs and subset data.")

    # This function handles the full lifecycle: Setup -> Train -> Validate -> Save -> Predict
    run_training()

    # Verify artifacts
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"    Success: Best model saved at {best_model_path}")
    else:
        print(
            "    Note: No best model saved (Validation metric might not have improved in 1 epoch)."
        )

    print("    Training loop executed successfully.")

    # -------------------------------------------------------------------------
    # 6. Integration Test: Inference
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference Standalone (Integration Test)...")

    # Demonstrating the standalone inference function
    # We use the model weights from the training run (or random if not saved)
    submission_df = inference_fn(save_submission=True, load_cached_data=True)

    print(f"    Submission DataFrame Shape: {submission_df.shape}")
    print(f"    Sample Prediction:\n{submission_df.head(2)}")

    # Assertions
    assert "prediction_id" in submission_df.columns, "Missing 'prediction_id' column"
    assert "cancer" in submission_df.columns, "Missing 'cancer' column"
    assert len(submission_df) > 0, "Submission DataFrame is empty"

    # Verify output file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"    Success: Submission file found at {Config.SUBMISSION_PATH}")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n===========================================================")
    print("   Demonstration Completed Successfully")
    print("===========================================================")


if __name__ == "__main__":
    demo_pipeline()
