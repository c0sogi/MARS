import os
import torch
import pandas as pd
import numpy as np
import shutil
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, mixup_data, mixed_criterion, calculate_roc_auc
from library.model import SEResNet
from library.dataset import WhaleDataset
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Right Whale Detection Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    print("\n[1/5] Configuring Environment...")

    # Define paths
    demo_working_dir = "./working/demo_execution"
    # We point cache to 'idea_2' to use existing preprocessed data (train.npz, etc.)
    # If this directory didn't exist, the Dataset class would process raw audio (slow).
    existing_cache_dir = "./working/idea_2"

    # Clean up previous demo run if exists
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)

    # Override Config attributes for a fast demo run
    Config.WORKING_DIR = demo_working_dir
    Config.CACHE_DIR = existing_cache_dir
    Config.SUBMISSION_DIR = os.path.join(demo_working_dir, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script

    # Initialize environment (creates directories)
    Config.setup()
    set_seed(Config.SEED)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Cache Directory:   {Config.CACHE_DIR}")
    print(f"    Device:            {Config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("\n[2/5] Verifying Utilities...")

    # Test Mixup
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 128, 125)  # (B, C, F, T)
    dummy_target = torch.tensor([0.0, 1.0, 0.0, 1.0])

    mixed_x, y_a, y_b, lam = mixup_data(dummy_input, dummy_target, alpha=0.4)

    assert mixed_x.shape == dummy_input.shape, "Mixup: Input shape mismatch"
    assert y_a.shape == dummy_target.shape, "Mixup: Target shape mismatch"
    assert 0.0 <= lam <= 1.0, "Mixup: Lambda out of range"
    print("    Mixup logic verified.")

    # Test Loss
    dummy_pred = torch.randn(batch_size, 1)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = mixed_criterion(criterion, dummy_pred, y_a.view(-1, 1), y_b.view(-1, 1), lam)

    assert isinstance(loss.item(), float), "Criterion: Loss is not a float"
    print("    Mixed Criterion verified.")

    # Test AUC
    test_y_true = [0, 1, 0, 1]
    test_y_pred = [0.1, 0.9, 0.3, 0.8]
    auc_score = calculate_roc_auc(test_y_true, test_y_pred)

    assert 0.0 <= auc_score <= 1.0, "AUC: Score out of range"
    print(f"    ROC AUC Calculation verified (Score: {auc_score:.4f}).")

    # ---------------------------------------------------------
    # 3. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[3/5] Verifying SEResNet Model...")

    model = SEResNet()
    device = torch.device(Config.DEVICE)
    model.to(device)

    # Forward pass with dummy data
    dummy_input_device = dummy_input.to(device)
    output = model(dummy_input_device)

    # Expected output: (Batch_Size, Num_Classes) -> (4, 1)
    assert output.shape == (
        batch_size,
        1,
    ), f"Model: Output shape mismatch. Expected {(batch_size, 1)}, got {output.shape}"
    print("    Model forward pass successful.")

    # ---------------------------------------------------------
    # 4. Run Trainer (Debug Mode)
    # ---------------------------------------------------------
    print("\n[4/5] Running Trainer (Debug Mode)...")

    # Trainer(debug=True) subsets the dataset to ~100 samples and limits inference batches
    trainer = Trainer(debug=True)

    # Verify dataset subsetting
    train_size = len(trainer.train_dataset)
    val_size = len(trainer.val_dataset)
    print(f"    Debug Train Size: {train_size}")
    print(f"    Debug Val Size:   {val_size}")

    assert (
        train_size <= 100
    ), "Trainer: Debug mode did not subset training data correctly."

    # Train
    print("    Starting training loop...")
    trainer.fit(epochs=Config.EPOCHS)

    # Check if model saved (Note: might not save if validation AUC is poor/random, but usually does)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(model_path):
        print(f"    Model saved to {model_path}")
    else:
        print(
            "    Notice: No best model saved (likely due to no AUC improvement in 1 epoch)."
        )

    # Predict
    print("    Starting inference loop...")
    trainer.predict()

    # ---------------------------------------------------------
    # 5. Validate Output
    # ---------------------------------------------------------
    print("\n[5/5] Validating Submission...")

    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_FILE}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission shape: {df_sub.shape}")
    print("    First 3 rows:")
    print(df_sub.head(3))

    # Validation assertions
    assert "clip" in df_sub.columns, "Submission: Missing 'clip' column"
    assert "probability" in df_sub.columns, "Submission: Missing 'probability' column"
    assert len(df_sub) > 0, "Submission: File is empty"

    # In debug mode, Trainer.predict limits to 5 batches.
    # Batch size 8 * 5 = 40 samples max.
    assert (
        len(df_sub) <= 40
    ), "Submission: Debug mode inference produced too many predictions."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
