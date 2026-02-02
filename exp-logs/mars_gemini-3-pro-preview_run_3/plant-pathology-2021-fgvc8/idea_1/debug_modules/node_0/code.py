import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_f1_score
from library.dataset import get_loaders
from library.model import AppleResNet18
from library.engine import run_training, run_inference


def main():
    print("Starting Apple Disease Detection Task Demonstration...")

    # ==========================================
    # 1. Setup and Configuration Overrides
    # ==========================================
    # Set seed for reproducibility
    set_seed(42)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device selected: {device}")

    # Override Config for speed and demonstration purposes
    # Disable downloading pretrained weights to ensure offline execution and speed
    Config.PRETRAINED = False
    # Reduce workers to minimize spawning overhead for small data
    Config.NUM_WORKERS = 2

    # Define small sample sizes for quick execution
    DEBUG_TRAIN_SIZE = 64
    DEBUG_TEST_SIZE = 20

    # Clean up previous runs if any
    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    if os.path.exists(Config.MODEL_CHECKPOINT_DIR):
        shutil.rmtree(Config.MODEL_CHECKPOINT_DIR)
    os.makedirs(Config.MODEL_CHECKPOINT_DIR, exist_ok=True)

    # ==========================================
    # 2. Verify Metric Logic
    # ==========================================
    print("\n--- Verifying Metric Logic ---")
    # Case 1: Perfect prediction
    # Logits > 0 -> Sigmoid > 0.5 -> Pred 1
    # Logits < 0 -> Sigmoid < 0.5 -> Pred 0
    logits_perfect = torch.tensor([[10.0, -10.0, 10.0], [-10.0, 10.0, -10.0]])
    targets_perfect = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    score_perfect = calculate_f1_score(logits_perfect, targets_perfect)
    print(f"Perfect Score (Expected 1.0): {score_perfect}")
    assert np.isclose(
        score_perfect, 1.0
    ), "Metric calculation failed for perfect predictions."

    # Case 2: Complete mismatch
    logits_wrong = torch.tensor([[-10.0, 10.0, -10.0], [10.0, -10.0, 10.0]])
    score_wrong = calculate_f1_score(logits_wrong, targets_perfect)
    print(f"Wrong Score (Expected 0.0): {score_wrong}")
    assert np.isclose(
        score_wrong, 0.0
    ), "Metric calculation failed for wrong predictions."
    print("Metric verification passed.")

    # ==========================================
    # 3. Verify Dataset and DataLoader
    # ==========================================
    print("\n--- Verifying Dataset and DataLoader ---")
    # Load a small subset of data
    train_loader, val_loader, test_loader = get_loaders(
        debug_sample_size=DEBUG_TRAIN_SIZE
    )

    # Fetch one batch
    images, targets, _ = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Assertions
    assert images.dim() == 4, "Images should be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert (
        images.shape[2] == Config.IMG_SIZE and images.shape[3] == Config.IMG_SIZE
    ), f"Images should be resized to {Config.IMG_SIZE}x{Config.IMG_SIZE}"
    assert (
        targets.shape[1] == Config.NUM_CLASSES
    ), f"Targets should have {Config.NUM_CLASSES} classes"
    assert targets.dtype == torch.float32, "Targets should be float32"
    print("Dataset verification passed.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n--- Verifying Model Architecture ---")
    model = AppleResNet18(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch"
    print("Model architecture verification passed.")

    # ==========================================
    # 5. Execute Training Pipeline
    # ==========================================
    print("\n--- Executing Training Pipeline ---")
    # Run training for 1 epoch with a small subset
    # This tests the Engine, Model, Loss, Optimizer, and Checkpointing
    trained_model = run_training(
        debug_sample_size=DEBUG_TRAIN_SIZE, epochs=1, device=device
    )

    # Verify Checkpoint
    best_model_path = os.path.join(Config.MODEL_CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."
    print(f"Training complete. Checkpoint found at {best_model_path}")

    # ==========================================
    # 6. Execute Inference Pipeline
    # ==========================================
    print("\n--- Executing Inference Pipeline ---")
    # Run inference using the trained model
    run_inference(trained_model, debug_sample_size=DEBUG_TEST_SIZE, device=device)

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not generated."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission Shape: {df_sub.shape}")
    print("Submission Head:")
    print(df_sub.head())

    # Assertions
    assert (
        "image" in df_sub.columns and "labels" in df_sub.columns
    ), "Submission missing required columns."
    # Note: test_loader drop_last=False, so we expect exactly DEBUG_TEST_SIZE rows
    assert (
        len(df_sub) == DEBUG_TEST_SIZE
    ), f"Expected {DEBUG_TEST_SIZE} predictions, got {len(df_sub)}"

    # Check label format (should be string, possibly empty or space-delimited)
    # Since we trained for 1 epoch on random weights/small data, predictions might be noise,
    # but the format must be correct.
    assert (
        df_sub["labels"].dtype == object
    ), "Labels column should be object/string type."

    print("Inference verification passed.")

    print("\nAll demonstrations and assertions passed successfully.")


if __name__ == "__main__":
    main()
