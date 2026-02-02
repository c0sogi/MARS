import os
import torch
import pandas as pd
import numpy as np
import shutil
import sys

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import CatDogDataset
from library.models import build_model
from library.train import train_one_model
from library.inference import run_ensemble


def run_demo():
    print("=== Starting Demonstration of Dog vs Cat Classification Library ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config settings for speed and debugging
    # We modify the class attributes directly since Python classes are mutable
    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Use only 10 images for quick verification
    Config.BATCH_SIZE = 2

    # Limit to a single model to save time (ResNet50 is standard and robust)
    Config.MODELS = ["resnet50.a1_in1k"]

    # Redirect output directories to a demo folder
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up and recreate demo directories
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("    Configuration overrides applied.")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Dataset Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Dataset Loading...")

    # Test Training Dataset
    try:
        train_ds = CatDogDataset(Config.TRAIN_CSV, phase="train", debug=True)
        print(f"    Successfully loaded Train Dataset. Size: {len(train_ds)}")

        # Assertions
        assert (
            len(train_ds) == Config.DEBUG_SAMPLE_SIZE
        ), f"Expected {Config.DEBUG_SAMPLE_SIZE} samples, got {len(train_ds)}"

        img, label = train_ds[0]
        assert isinstance(img, torch.Tensor), "Output image must be a torch.Tensor"
        assert img.dim() == 3, f"Image must be 3D (C, H, W), got {img.dim()}"
        assert isinstance(label, torch.Tensor), "Label must be a torch.Tensor"
        # Label should be a scalar float for BCEWithLogitsLoss
        assert label.dtype == torch.float32, "Label must be float32"

        print("    Train dataset item check passed.")

    except Exception as e:
        print(f"    FAILED to load Train Dataset: {e}")
        raise e

    # Test Test Dataset
    try:
        test_ds = CatDogDataset(Config.TEST_CSV, phase="test", debug=True)
        print(f"    Successfully loaded Test Dataset. Size: {len(test_ds)}")

        img, img_id = test_ds[0]
        # ID should be an integer or numpy integer
        assert isinstance(
            img_id, (int, np.integer)
        ), f"ID must be integer, got {type(img_id)}"

        print("    Test dataset item check passed.")

    except Exception as e:
        print(f"    FAILED to load Test Dataset: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 3. Model Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Construction...")

    model_name = Config.MODELS[0]
    try:
        # Build model (pretrained=False for speed/offline safety in demo,
        # though training usually uses True)
        model = build_model(model_name, pretrained=False)
        print(f"    Successfully built model: {model_name}")

        assert isinstance(model, torch.nn.Module), "Model must be a PyTorch Module"

        # Verify Forward Pass
        dummy_input = torch.randn(
            Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE
        )
        with torch.no_grad():
            output = model(dummy_input)

        # Expected output shape: [Batch, 1] (Binary Classification Logits)
        expected_shape = (Config.BATCH_SIZE, 1)
        assert (
            output.shape == expected_shape
        ), f"Output shape mismatch. Expected {expected_shape}, got {output.shape}"

        print("    Model forward pass check passed.")

    except Exception as e:
        print(f"    FAILED to build or run model: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 4. Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Training Loop...")

    try:
        # Run training for the single model configured
        # This function handles data loading, training, validation, and saving
        train_one_model(model_name, patience=1)

        # Verify Checkpoint Creation
        expected_checkpoint = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")
        assert os.path.exists(
            expected_checkpoint
        ), f"Checkpoint file not found at {expected_checkpoint}"

        print("    Training loop completed and checkpoint verified.")

    except Exception as e:
        print(f"    FAILED during training execution: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 5. Inference & Ensemble Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Inference and Ensembling...")

    try:
        # Run the ensemble inference
        # This loads the checkpoint we just trained and generates submission.csv
        run_ensemble()

        # Verify Submission File
        assert os.path.exists(
            Config.SUBMISSION_PATH
        ), f"Submission file not found at {Config.SUBMISSION_PATH}"

        # Verify Submission Content
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)

        # Check Columns
        required_cols = {"id", "label"}
        assert required_cols.issubset(
            df_sub.columns
        ), f"Submission missing required columns. Found: {df_sub.columns}"

        # Check Length (Should match DEBUG_SAMPLE_SIZE)
        assert (
            len(df_sub) == Config.DEBUG_SAMPLE_SIZE
        ), f"Submission length mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_sub)}"

        # Check Values (Probabilities should be between 0 and 1)
        assert (
            df_sub["label"].min() >= 0.0 and df_sub["label"].max() <= 1.0
        ), "Probabilities out of range [0, 1]"

        print("    Inference completed and submission file verified.")
        print(f"    Sample predictions:\n{df_sub.head()}")

    except Exception as e:
        print(f"    FAILED during inference execution: {e}")
        raise e

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
