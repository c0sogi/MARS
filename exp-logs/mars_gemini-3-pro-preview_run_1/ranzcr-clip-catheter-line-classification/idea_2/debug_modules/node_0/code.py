import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterModel
from library.trainer import Trainer, get_pos_weights
from library.inference import predict
from library.utils import seed_everything, calculate_metric

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=" * 50)
    print("Starting Catheter Detection Pipeline Demo")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Configuration Overrides for Speed and Demonstration
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Set paths to a specific demo directory to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Override hyperparameters for speed
    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 16  # Small sample for quick execution
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.IMAGE_SIZE = 224  # Smaller image size for CPU/fast GPU processing
    Config.PRETRAINED = False  # Skip downloading weights for demo
    Config.NUM_WORKERS = 2  # Reduce workers for small demo

    seed_everything(Config.SEED)
    print("Configuration updated successfully.")

    # ---------------------------------------------------------
    # 2. Dataset Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Dataset Logic...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)

    # Initialize dataset in debug mode
    dataset = CatheterDataset(
        df=train_df,
        transforms=get_transforms("train"),
        mode="train",
        debug=True,
        debug_size=Config.DEBUG_SAMPLE_SIZE,
    )

    print(f"Dataset initialized. Size: {len(dataset)}")

    # Fetch one sample
    image, label = dataset[0]

    # Assertions
    assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor"
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"
    assert image.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Expected image shape (3, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE}), got {image.shape}"
    assert label.shape == (
        Config.NUM_CLASSES,
    ), f"Expected label shape ({Config.NUM_CLASSES},), got {label.shape}"

    print("Dataset verification passed: Output shapes are correct.")

    # ---------------------------------------------------------
    # 3. Model Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = CatheterModel(model_name=Config.MODEL_NAME, pretrained=False)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_input)

    # Assertions
    assert logits.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected logits shape (2, {Config.NUM_CLASSES}), got {logits.shape}"

    print("Model verification passed: Forward pass successful.")

    # ---------------------------------------------------------
    # 4. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[4] Running Training Loop (1 Epoch)...")

    # Initialize Trainer
    trainer = Trainer(debug=True)

    # Run training
    # This covers: DataLoader, Loss (with pos_weights), Optimizer, Backprop, Validation, Saving
    trainer.fit(epochs=Config.EPOCHS)

    # Check if model was saved
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "best_model.pth was not saved!"

    # Check if pos_weights were cached
    weights_path = os.path.join(Config.WORKING_DIR, "pos_weights.npy")
    assert os.path.exists(weights_path), "pos_weights.npy was not cached!"

    print("Training loop completed successfully. Model artifact generated.")

    # ---------------------------------------------------------
    # 5. Inference Demonstration
    # ---------------------------------------------------------
    print("\n[5] Running Inference and Generating Submission...")

    # Run prediction
    # This covers: Loading saved model, TTA, Submission generation
    predict(debug=True, debug_size=Config.DEBUG_SAMPLE_SIZE)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found!"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {sub_df.shape}")

    # Verify columns
    expected_cols = ["StudyInstanceUID"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), "Submission columns do not match requirements."

    # Verify row count matches debug size
    assert (
        len(sub_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows, got {len(sub_df)}"

    print("Inference verification passed.")

    # ---------------------------------------------------------
    # 6. Utility Verification (Metric)
    # ---------------------------------------------------------
    print("\n[6] Verifying Metric Calculation...")

    # Create synthetic data
    # Case 1: Perfect prediction
    y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    y_pred = np.array([[0.1, 0.9, 0.1], [0.9, 0.1, 0.1], [0.1, 0.1, 0.9]])

    # Note: calculate_metric expects (N, 11), but iterates by column.
    # We can test with smaller shape if we mock the function or just pad it.
    # Let's use the actual shape (N, 11) to be safe with the function's expectations.

    y_true_full = np.zeros((10, Config.NUM_CLASSES))
    y_pred_full = np.zeros((10, Config.NUM_CLASSES))

    # Set column 0 to have variance (0s and 1s)
    y_true_full[:5, 0] = 1
    y_pred_full[:5, 0] = 0.9  # Good predictions
    y_pred_full[5:, 0] = 0.1

    # Set column 1 to be all zeros (should be skipped or handled gracefully)
    y_true_full[:, 1] = 0
    y_pred_full[:, 1] = 0.2

    score = calculate_metric(y_true_full, y_pred_full)

    # Since column 0 is perfect (AUC=1.0) and column 1 is skipped (undefined AUC),
    # the result should be 1.0 (average of valid columns).
    print(f"Calculated Metric: {score}")
    assert score == 1.0, f"Expected metric 1.0, got {score}"

    print("Metric verification passed.")

    print("\n" + "=" * 50)
    print("ALL DEMONSTRATION STEPS COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
