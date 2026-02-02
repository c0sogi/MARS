import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import FractureDataset
from library.model import FractureModel
from library.train import run_training
from library.inference import generate_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    print("[1/5] Setting up environment and configuration...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Initialize directories and seeds
    Config.setup()
    seed_everything(Config.SEED)

    # Override Config for speed optimization in this demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Use a tiny subset for demonstration
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure working directory for cache exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(
        f"    Configuration set: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Debug={Config.DEBUG}"
    )

    # -------------------------------------------------------------------------
    # 2. Dataset Logic Verification
    # -------------------------------------------------------------------------
    print("\n[2/5] Verifying Dataset and Data Loading...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH).head(Config.DEBUG_SAMPLE_SIZE)

    # Instantiate Dataset
    dataset = FractureDataset(
        train_df,
        mode="train",
        load_cached_data=False,  # Disable cache reading to force processing logic verification
    )

    # Fetch a single sample
    images, labels = dataset[0]

    # Verification Assertions
    # Expected Image Shape: (NUM_SLICES, 3, IMG_SIZE, IMG_SIZE)
    expected_img_shape = (Config.NUM_SLICES, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"

    # Expected Label Shape: (7,) for C1-C7
    expected_label_shape = (7,)
    assert (
        labels.shape == expected_label_shape
    ), f"Label shape mismatch. Expected {expected_label_shape}, got {labels.shape}"

    print(
        f"    Dataset verified. Sample shape: {images.shape}, Labels: {labels.numpy()}"
    )

    # -------------------------------------------------------------------------
    # 3. Model Logic Verification
    # -------------------------------------------------------------------------
    print("\n[3/5] Verifying Model Architecture and Forward Pass...")

    device = Config.DEVICE
    model = FractureModel(
        backbone_name=Config.BACKBONE,
        pretrained=False,  # Skip download for speed
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    # Create a dummy batch: (Batch_Size, NUM_SLICES, 3, IMG_SIZE, IMG_SIZE)
    dummy_input = torch.randn(
        2, Config.NUM_SLICES, 3, Config.IMG_SIZE, Config.IMG_SIZE
    ).to(device)

    # Perform forward pass
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)

    # Verification Assertions
    # Expected Output Shape: (Batch_Size, Num_Classes)
    expected_out_shape = (2, Config.NUM_CLASSES)
    assert (
        logits.shape == expected_out_shape
    ), f"Model output shape mismatch. Expected {expected_out_shape}, got {logits.shape}"

    print(f"    Model forward pass successful. Output shape: {logits.shape}")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[4/5] Executing Training Loop (Debug Mode)...")

    # Run training using the library function
    # This will train for 1 epoch on the debug subset and save the model
    run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG)

    # Verify Checkpoint Creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Training failed to save checkpoint at {checkpoint_path}"
        )

    print(f"    Training complete. Checkpoint saved at: {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission Demonstration
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating Submission (Debug Mode)...")

    # Generate submission using the library function
    submission_path = Config.SUBMISSION_PATH
    generate_submission(
        checkpoint_path=checkpoint_path,
        output_path=submission_path,
        batch_size=Config.BATCH_SIZE,
        device=Config.DEVICE,
        debug=Config.DEBUG,
    )

    # Verify Submission File
    if not os.path.exists(submission_path):
        raise FileNotFoundError(
            f"Inference failed to save submission at {submission_path}"
        )

    # Validate Submission Content
    sub_df = pd.read_csv(submission_path)
    required_cols = ["row_id", "fractured"]

    assert all(
        col in sub_df.columns for col in required_cols
    ), f"Submission missing required columns. Found: {sub_df.columns}"

    assert len(sub_df) > 0, "Submission file is empty."

    # Check if probabilities are valid
    if sub_df["fractured"].min() < 0 or sub_df["fractured"].max() > 1:
        raise ValueError("Submission contains probabilities outside [0, 1].")

    print(f"    Submission generated successfully at: {submission_path}")
    print(f"    First 5 rows:\n{sub_df.head().to_string(index=False)}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
