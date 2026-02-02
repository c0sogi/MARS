import os
import sys
import pandas as pd
import numpy as np
import torch
import time

# Import from the provided library
from library.config import Config
from library.utils import set_seed, load_dicom_image, resize_image, normalize_image
from library.data_loader import get_dataloader
from library.model import AsymmetricEfficientNet
from library.train import run_training
from library.evaluate import generate_submission


def run_demo():
    print("=== Starting Library Demonstration ===\n")

    # --------------------------------------------------------------------------
    # 0. Configuration & Setup
    # --------------------------------------------------------------------------
    print("--- Step 0: Configuring Environment for Speed ---")

    # Modify Config for rapid execution
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SIZE = 8  # Use a very small subset
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure cache dir exists for our demo
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    set_seed(Config.SEED)
    print(
        f"Config configured: Epochs={Config.NUM_EPOCHS}, Batch={Config.BATCH_SIZE}, DebugSize={Config.DEBUG_SIZE}"
    )
    print("Random seed set.\n")

    # --------------------------------------------------------------------------
    # 1. Verify Utils (Image Processing)
    # --------------------------------------------------------------------------
    print("--- Step 1: Verifying Utility Functions ---")

    # Load metadata to find a real file
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    row = df_train.iloc[0]
    flair_dir = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])

    # Find a DICOM file
    files = [f for f in os.listdir(flair_dir) if f.endswith(".dcm")]
    assert len(files) > 0, "No DICOM files found in sample directory."
    sample_path = os.path.join(flair_dir, files[0])
    print(f"Testing with file: {sample_path}")

    # Test load_dicom_image
    img = load_dicom_image(sample_path)
    assert img is not None, "Failed to load DICOM image."
    assert isinstance(img, np.ndarray), "Loaded image is not a numpy array."
    print(f"Image loaded successfully. Shape: {img.shape}")

    # Test resize_image
    img_resized = resize_image(img, size=Config.IMG_SIZE)
    assert img_resized.shape == (
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Resize failed. Expected ({Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img_resized.shape}"
    print("Image resized successfully.")

    # Test normalize_image
    img_norm = normalize_image(img_resized)
    assert img_norm.dtype == np.float32, "Normalized image is not float32."
    assert (
        0.0 <= img_norm.min() and img_norm.max() <= 1.0 + 1e-6
    ), f"Normalization range error. Min: {img_norm.min()}, Max: {img_norm.max()}"
    print("Image normalized successfully.\n")

    # --------------------------------------------------------------------------
    # 2. Verify Data Loader
    # --------------------------------------------------------------------------
    print("--- Step 2: Verifying Data Loader ---")

    # Initialize loader in debug mode
    # This triggers process_subject -> caching -> dataset creation
    loader = get_dataloader(split="train", batch_size=Config.BATCH_SIZE, debug=True)

    # Fetch one batch
    inputs, targets = next(iter(loader))

    # Verify shapes
    # Inputs: (B, 12, 224, 224)
    expected_input_shape = (
        Config.BATCH_SIZE,
        Config.NUM_CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    assert (
        inputs.shape == expected_input_shape
    ), f"Input batch shape mismatch. Expected {expected_input_shape}, got {inputs.shape}"

    # Targets: (B,) or (B, 1) depending on loader implementation, but dataset returns scalar
    # The loader stacks them, so it should be (B,)
    assert targets.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch."

    print(f"Batch loaded. Inputs: {inputs.shape}, Targets: {targets.shape}")
    print("Data Loader verification passed.\n")

    # --------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("--- Step 3: Verifying Model Architecture ---")

    device = torch.device(Config.DEVICE)
    model = AsymmetricEfficientNet(pretrained=False)  # False for speed
    model.to(device)

    # Forward pass with the batch from Step 2
    inputs = inputs.to(device)
    outputs = model(inputs)

    # Verify output shape: (B, 1)
    expected_output_shape = (Config.BATCH_SIZE, 1)
    assert (
        outputs.shape == expected_output_shape
    ), f"Model output shape mismatch. Expected {expected_output_shape}, got {outputs.shape}"

    print(f"Forward pass successful. Output shape: {outputs.shape}")
    print("Model verification passed.\n")

    # --------------------------------------------------------------------------
    # 4. Demonstrate Training Loop
    # --------------------------------------------------------------------------
    print("--- Step 4: Running Training Loop (Demo) ---")

    # run_training uses Config settings we modified in Step 0
    # It will use the cached data generated in Step 2 if available/compatible
    run_training(debug=True)

    # Verify model checkpoint was created
    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint was not saved."
    print(f"Training finished. Checkpoint found at {Config.MODEL_PATH}\n")

    # --------------------------------------------------------------------------
    # 5. Demonstrate Inference & Submission
    # --------------------------------------------------------------------------
    print("--- Step 5: Generating Submission (Demo) ---")

    # generate_submission loads the model from Config.MODEL_PATH and predicts on test set
    generate_submission(debug=True)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission file loaded.")
    print(df_sub.head())

    # Verify format
    assert (
        "BraTS21ID" in df_sub.columns and "MGMT_value" in df_sub.columns
    ), "Submission file missing required columns."
    assert len(df_sub) > 0, "Submission file is empty."

    # Verify probability range
    probs = df_sub["MGMT_value"].values
    assert np.all(
        (probs >= 0) & (probs <= 1)
    ), "Predictions contain values outside [0, 1]."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
