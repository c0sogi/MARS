import os
import sys
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.dataset import CactusDataset
from library.model import UltraWideSERepNeXt
from library.train import train_model
from library.inference import generate_submission


def main():
    # 1. Setup and Reproducibility
    print("Initializing demonstration...")
    seed = 42
    set_seed(seed)

    # 2. Configuration Override for Fast Demonstration
    # We override Config attributes to run a minimal version of the pipeline.
    print("Overriding configuration for speed...")
    Config.EPOCHS = 1
    Config.SEEDS = [seed]  # Only run one seed
    Config.BATCH_SIZE = 16  # Small batch size for debug
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 3. Data Verification
    print("\n--- Verifying Dataset Logic ---")
    # Initialize dataset (this will process metadata and cache data if needed)
    # We use 'train' mode to check image and label loading
    train_ds = CactusDataset(mode="train", load_cached_data=True)

    # Check dataset length
    print(f"Dataset length: {len(train_ds)}")
    if len(train_ds) == 0:
        raise AssertionError("Dataset is empty.")

    # Check item retrieval
    img, label = train_ds[0]

    # Verify Image shape: (3, 32, 32)
    print(f"Sample image shape: {img.shape}")
    if img.shape != (3, 32, 32):
        raise AssertionError(f"Expected image shape (3, 32, 32), got {img.shape}")

    # Verify Label shape and type
    print(f"Sample label: {label} (Type: {type(label)})")
    if not isinstance(label, torch.Tensor):
        raise AssertionError("Label should be a torch.Tensor")

    print("Dataset verification passed.")

    # 4. Model Verification
    print("\n--- Verifying Model Logic ---")
    device = Config.DEVICE
    model = UltraWideSERepNeXt().to(device)

    # Create dummy input: Batch size 2, 3 channels, 32x32
    dummy_input = torch.randn(2, 3, 32, 32).to(device)

    # Test Training Forward Pass
    model.train()
    output = model(dummy_input)
    print(f"Training output shape: {output.shape}")

    if output.shape != (2, 1):
        raise AssertionError(f"Expected output shape (2, 1), got {output.shape}")

    # Test Structural Re-parameterization (Switch to Deploy)
    print("Testing switch_to_deploy()...")
    model.eval()
    model.switch_to_deploy()

    # Test Inference Forward Pass
    with torch.no_grad():
        deploy_output = model(dummy_input)

    print(f"Deploy output shape: {deploy_output.shape}")
    if deploy_output.shape != (2, 1):
        raise AssertionError(
            f"Expected deploy output shape (2, 1), got {deploy_output.shape}"
        )

    print("Model verification passed.")

    # 5. Training Demonstration
    print("\n--- Running Training (Debug Mode) ---")
    # Run training for 1 epoch on a subset of data
    # train_model returns the best validation AUC
    best_auc = train_model(seed=seed, epochs=Config.EPOCHS, debug=True)

    print(f"Training completed with Best Val AUC: {best_auc}")

    # Verify that the model checkpoint was saved
    expected_ckpt_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")
    if not os.path.exists(expected_ckpt_path):
        raise AssertionError(f"Checkpoint file not found at {expected_ckpt_path}")

    print(f"Checkpoint verified at: {expected_ckpt_path}")

    # 6. Inference Demonstration
    print("\n--- Running Inference (Debug Mode) ---")
    # generate_submission uses Config.SEEDS to find models.
    # We set Config.SEEDS = [42] earlier, so it will look for model_seed_42.pth
    generate_submission(debug=True)

    # Verify submission file
    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise AssertionError(f"Submission file not found at {submission_path}")

    # Verify content format
    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print("Submission head:")
    print(df_sub.head())

    required_cols = {"id", "has_cactus"}
    if not required_cols.issubset(df_sub.columns):
        raise AssertionError(f"Submission missing columns. Found: {df_sub.columns}")

    # Check values are probabilities
    if df_sub["has_cactus"].min() < 0 or df_sub["has_cactus"].max() > 1:
        raise AssertionError("Predictions contain values outside [0, 1] range.")

    print("Inference verification passed.")

    print("\n========================================")
    print("   DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("========================================")


if __name__ == "__main__":
    main()
