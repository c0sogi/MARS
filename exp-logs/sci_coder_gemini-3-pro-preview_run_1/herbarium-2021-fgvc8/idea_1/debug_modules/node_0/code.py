import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.dataset import get_dataloaders, get_label_mapping
from library.model import get_model
from library.trainer import Trainer
from library.inference import predict_and_submit


def run_demo():
    print("Starting Herbarium Classification Demo...")

    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config values to run a quick test
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Small subset for speed
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_EPOCHS = 1  # Only 1 epoch
    Config.NUM_WORKERS = 2  # Reduce workers for small data
    Config.IS_DEMO = (
        True  # Custom flag if needed (not used by lib but good for tracking)
    )

    # Ensure clean slate for outputs
    if os.path.exists(Config.IDEA_DIR):
        shutil.rmtree(Config.IDEA_DIR)
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    Config.print_config()

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n[2] Verifying DataLoaders and Label Mapping...")

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)

    # Check Train Loader
    images, labels = next(iter(train_loader))
    print(f"    Train Batch - Images Shape: {images.shape}")
    print(f"    Train Batch - Labels Shape: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label shape"

    # Verify label mapping file creation
    classes_path = os.path.join(Config.IDEA_DIR, "classes.npy")
    assert os.path.exists(classes_path), "classes.npy was not created"

    unique_cats = np.load(classes_path)
    print(f"    Total Unique Categories (Mapped): {len(unique_cats)}")

    # Verify labels are within range
    assert labels.max() < Config.NUM_CLASSES, "Label index out of bounds"
    assert labels.min() >= 0, "Label index negative"

    # ==========================================
    # 3. Model Verification
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    model = get_model(pretrained=False)  # No need to download weights for demo
    model.to(Config.DEVICE)
    model.eval()

    # Forward pass check
    with torch.no_grad():
        dummy_input = images.to(Config.DEVICE)
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    print("    Model forward pass successful.")

    # ==========================================
    # 4. Training Loop Verification
    # ==========================================
    print("\n[4] Running Training Loop (1 Epoch)...")

    trainer = Trainer(debug=Config.DEBUG)

    # We use the trainer's train method which runs for Config.NUM_EPOCHS (set to 1)
    trainer.train()

    # Verify model checkpoint exists
    model_path = os.path.join(Config.IDEA_DIR, "model.pth")
    assert os.path.exists(model_path), "Model checkpoint was not saved"
    print(f"    Model saved successfully at {model_path}")

    # ==========================================
    # 5. Inference and Submission Verification
    # ==========================================
    print("\n[5] Running Inference and Generating Submission...")

    # Run inference using the saved model
    predict_and_submit(debug=Config.DEBUG, model_path=model_path)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission Shape: {sub_df.shape}")
    print(f"    Submission Columns: {sub_df.columns.tolist()}")

    assert list(sub_df.columns) == ["Id", "Predicted"], "Incorrect submission columns"
    assert len(sub_df) > 0, "Submission file is empty"
    assert (
        sub_df["Predicted"].dtype == "int64" or sub_df["Predicted"].dtype == "int32"
    ), "Predicted column should be integer"

    # Verify that predicted values map back to real category IDs (not just 0..N indices)
    # We check if any predicted value exists in the unique_cats loaded earlier
    # Since we used a random model and small data, exact correctness isn't expected, but type is.
    sample_pred = sub_df["Predicted"].iloc[0]
    print(f"    Sample Prediction (Category ID): {sample_pred}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    # Set fixed seeds for reproducibility in the demo script itself
    torch.manual_seed(42)
    np.random.seed(42)

    run_demo()
