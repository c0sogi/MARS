import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import provided library components
from library.config import Config
from library.dataset import CactusDataset
from library.model import HybridNarrowSEResNet
from library.train import run_training
from library.inference import run_inference
from library.utils import seed_everything


def main():
    print("============================================================")
    print("       Cactus Identification: Demo & Verification Script    ")
    print("============================================================")

    # -------------------------------------------------------------------------
    # 1. Configuration Modification for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo run...")

    # Set a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)  # Clean start
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce computational load for the demo
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.SEEDS = [42]  # Use only one seed
    Config.NUM_WORKERS = 2  # Reduce workers for simple script

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Seeds: {Config.SEEDS}")

    # -------------------------------------------------------------------------
    # 2. Verify Dataset Logic
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Dataset Logic...")

    # Load a tiny subset of the training data
    demo_dataset = CactusDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        phase="train",
        load_cached_data=False,  # Force reload from disk to test logic
        limit=10,
    )

    print(f"Loaded demo dataset with size: {len(demo_dataset)}")

    # Fetch one sample
    img, label, img_id = demo_dataset[0]

    # Assertions
    assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor"
    assert img.shape == (
        3,
        32,
        32,
    ), f"Expected image shape (3, 32, 32), got {img.shape}"
    assert isinstance(
        label, (float, np.float32)
    ), f"Label should be float, got {type(label)}"
    assert isinstance(img_id, str), "ID should be a string"

    print("Dataset verification passed: Shape and types are correct.")

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridNarrowSEResNet().to(device)
    model.eval()

    # Create dummy batch (Batch Size=4, Channels=3, Height=32, Width=32)
    dummy_input = torch.randn(4, 3, 32, 32).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Assertions
    assert output.shape == (4, 1), f"Expected output shape (4, 1), got {output.shape}"
    print(
        "Model verification passed: Forward pass successful with correct output shape."
    )

    # -------------------------------------------------------------------------
    # 4. Run Training (Fast Mode)
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop (1 Epoch)...")

    # This uses the imported run_training function which uses the modified Config
    best_auc = run_training(seed=42)

    # Verify model artifact creation
    expected_model_path = os.path.join(Config.WORKING_DIR, "model_seed_42.pth")
    assert os.path.exists(
        expected_model_path
    ), f"Model file not found at {expected_model_path}"

    print(f"Training complete. Best AUC: {best_auc:.4f}")
    print(f"Model saved to: {expected_model_path}")

    # -------------------------------------------------------------------------
    # 5. Run Inference
    # -------------------------------------------------------------------------
    print("\n[5] Running Inference...")

    # This uses the imported run_inference function
    # It will load the model we just trained (seed 42) because Config.SEEDS=[42]
    run_inference()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Basic validation of submission content
    assert (
        "id" in df_sub.columns and "has_cactus" in df_sub.columns
    ), "Missing columns in submission."
    assert len(df_sub) == 3325, f"Expected 3325 predictions, got {len(df_sub)}"
    assert (
        df_sub["has_cactus"].min() >= 0.0 and df_sub["has_cactus"].max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("Inference verification passed.")

    print("\n============================================================")
    print("       Demo Completed Successfully!                         ")
    print("============================================================")


if __name__ == "__main__":
    seed_everything(42)
    main()
