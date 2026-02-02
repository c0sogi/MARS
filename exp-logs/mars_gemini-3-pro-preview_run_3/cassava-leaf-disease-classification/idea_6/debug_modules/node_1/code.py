import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Configuration Setup
# We import Config first to override settings for a fast demonstration.
from library.config import Config

# Override Config for Speed and Demo purposes
Config.DEBUG = True
Config.EPOCHS = 1
Config.IMG_SIZE = 128  # Reduce image size for faster processing
Config.BATCH_SIZE = 8
Config.ACCUMULATION_STEPS = 1
Config.WORKING_DIR = "./working/demo_execution"
Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
Config.NUM_WORKERS = 2  # Reduce workers for small debug batch

# Ensure directories exist
Config.setup()

# Now import the rest of the library modules
from library.utils import seed_everything, calculate_accuracy
from library.dataset import get_dataset
from library.model import DualStreamModel
from library.engine import train_model, generate_submission


def run_demo():
    print("=== Starting Demo Execution ===")

    # 2. Verify Utils
    print("\n[1/6] Verifying Utils...")
    seed_everything(Config.SEED)

    # Test calculate_accuracy
    logits = np.array([[2.0, 0.5, 0.1], [0.1, 2.0, 0.1]])
    targets = np.array([0, 1])
    acc = calculate_accuracy(logits, targets)
    assert acc == 1.0, f"Accuracy calculation failed. Expected 1.0, got {acc}"

    logits_tensor = torch.tensor([[0.1, 0.1, 2.0], [2.0, 0.1, 0.1]])
    targets_tensor = torch.tensor([2, 0])
    acc_tensor = calculate_accuracy(logits_tensor, targets_tensor)
    assert (
        acc_tensor == 1.0
    ), f"Tensor accuracy calculation failed. Expected 1.0, got {acc_tensor}"
    print("Utils verified successfully.")

    # 3. Verify Dataset & DataLoaders
    print("\n[2/6] Verifying Dataset Loading...")
    # Load datasets in debug mode (100 samples)
    train_ds = get_dataset("train", debug=True)
    val_ds = get_dataset("val", debug=True)
    test_ds = get_dataset("test", debug=True)

    # Assertions
    assert (
        len(train_ds) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train ds size mismatch: {len(train_ds)}"
    assert (
        len(test_ds) == Config.DEBUG_SAMPLE_SIZE
    ), f"Test ds size mismatch: {len(test_ds)}"

    # Check item structure
    img, label = train_ds[0]
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch: {img.shape}"
    assert isinstance(label, torch.Tensor), "Label is not a tensor"

    # Check test item structure (should return image_id)
    img_t, label_t, img_id_t = test_ds[0]
    assert isinstance(img_id_t, str), "Image ID is not a string"

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    print("Datasets and Loaders initialized successfully.")

    # 4. Verify Model Architecture
    print("\n[3/6] Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)

    # Instantiate model (using pretrained=False for speed in instantiation check)
    model = DualStreamModel(pretrained=False)
    model.to(device)
    model.eval()

    # Dummy forward pass
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch: {output.shape}"
    print("Model architecture verified successfully.")

    # 5. Run Training Loop (Engine)
    print("\n[4/6] Running Training Loop (1 Epoch)...")
    # Note: engine.train_model instantiates its own model internally.

    # Clean up previous run if exists
    if os.path.exists(Config.MODEL_SAVE_PATH):
        os.remove(Config.MODEL_SAVE_PATH)

    best_acc = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=Config.EPOCHS,
        patience=1,
        save_path=Config.MODEL_SAVE_PATH,
    )

    print(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."

    # 6. Run Inference (Engine)
    print("\n[5/6] Running Inference...")
    if os.path.exists(Config.SUBMISSION_PATH):
        os.remove(Config.SUBMISSION_PATH)

    generate_submission(
        test_loader=test_loader,
        device=device,
        model_path=Config.MODEL_SAVE_PATH,
        output_path=Config.SUBMISSION_PATH,
    )

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."
    print("Inference complete.")

    # 7. Final Output Verification
    print("\n[6/6] Verifying Submission File...")
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check shape (should match debug sample size)
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission length mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_sub)}"

    # Check columns
    assert (
        "image_id" in df_sub.columns and "label" in df_sub.columns
    ), "Submission columns mismatch."

    # Check values
    assert (
        df_sub["label"].dtype == np.int64 or df_sub["label"].dtype == np.int32
    ), "Label column is not integer."

    print("Submission file valid.")
    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
