import os
import shutil
import numpy as np
import pandas as pd
import torch

# Import from the provided library files
from library.config import CFG
from library.utils import set_seed, calculate_per_class_lwlrap, calculate_overall_lwlrap
from library.dataset import AudioDataset, get_dataloader
from library.model import AudioEfficientNet
from library.trainer import Trainer


def run_demo():
    # ==========================================
    # 1. Configure for Speed and Demo
    # ==========================================
    print("--- Configuring for Speed and Demo ---")
    # Override Config for fast execution
    CFG.debug = True  # Use small subset (100 samples)
    CFG.epochs = 1  # Run only 1 epoch
    CFG.batch_size = 4  # Small batch size
    CFG.inference_batch_size = 4
    CFG.pretrained = False  # Disable downloading weights for speed

    # Ensure working directories are clean
    if os.path.exists(CFG.output_dir):
        shutil.rmtree(CFG.output_dir)
    os.makedirs(CFG.output_dir, exist_ok=True)

    # Set seed for reproducibility
    set_seed(CFG.seed)
    print("Configuration updated.")

    # ==========================================
    # 2. Verify Utils (Metric Calculation)
    # ==========================================
    print("\n--- Verifying Utils (LWLRAP Metric) ---")
    # Create synthetic ground truth (binary) and predictions (probabilities)
    # 3 samples, 3 classes
    truth = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
    scores = np.array([[0.8, 0.1, 0.1], [0.2, 0.9, 0.8], [0.4, 0.3, 0.3]])

    # Calculate metrics
    score_per_class, weight_per_class = calculate_per_class_lwlrap(truth, scores)
    overall_score = calculate_overall_lwlrap(truth, scores)

    print(f"Per Class Score: {score_per_class}")
    print(f"Overall LWLRAP: {overall_score:.4f}")

    # Assertions
    assert (
        len(score_per_class) == 3
    ), "Per-class score should have length equal to num_classes"
    assert 0.0 <= overall_score <= 1.0, "Overall score must be between 0 and 1"
    print("Metric verification passed.")

    # ==========================================
    # 3. Verify Dataset and DataLoader
    # ==========================================
    print("\n--- Verifying Dataset and DataLoader ---")
    # Initialize DataLoader in debug mode (loads 100 samples)
    train_loader = get_dataloader("train", debug=True)

    # Fetch one batch
    batch = next(iter(train_loader))
    images = batch["image"]
    targets = batch["target"]
    fnames = batch["fname"]

    print(f"Batch Image Shape: {images.shape}")  # Expected: (4, 1, 128, Time)
    print(f"Batch Target Shape: {targets.shape}")  # Expected: (4, 80)

    # Assertions
    assert images.ndim == 4, "Images should be 4D tensors (B, C, F, T)"
    assert images.shape[0] == CFG.batch_size, f"Batch size should be {CFG.batch_size}"
    assert images.shape[1] == 1, "Input channel should be 1 (spectrogram)"
    assert images.shape[2] == CFG.n_mels, f"Frequency dim should be {CFG.n_mels}"
    assert (
        targets.shape[1] == CFG.num_classes
    ), f"Target dim should be {CFG.num_classes}"
    print("Dataset and DataLoader verification passed.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n--- Verifying Model Architecture ---")
    # Instantiate model
    # Note: We set CFG.pretrained = False earlier to avoid downloads
    model = AudioEfficientNet(pretrained=CFG.pretrained)
    model.to(CFG.device)
    model.eval()

    # Forward pass with the batch fetched earlier
    with torch.no_grad():
        outputs = model(images.to(CFG.device))

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        CFG.batch_size,
        CFG.num_classes,
    ), f"Output shape mismatch. Expected {(CFG.batch_size, CFG.num_classes)}, got {outputs.shape}"
    print("Model architecture verification passed.")

    # ==========================================
    # 5. Verify Trainer (Fit & Predict)
    # ==========================================
    print("\n--- Verifying Trainer (Training & Inference) ---")
    trainer = Trainer()

    # 5a. Run Training Loop
    print("Starting training loop...")
    trainer.fit(epochs=CFG.epochs)

    # Verify model checkpoint exists
    best_model_path = os.path.join(CFG.output_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    print(f"Training finished. Model saved at {best_model_path}")

    # 5b. Run Inference
    print("Starting inference...")
    trainer.predict()

    # Verify submission file
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    # Check submission content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission DataFrame Shape: {sub_df.shape}")

    # Assertions
    # In debug mode, test loader also loads 100 samples
    expected_rows = 100
    expected_cols = CFG.num_classes + 1  # +1 for fname

    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(sub_df)}"
    assert (
        sub_df.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, got {sub_df.shape[1]}"
    assert "fname" in sub_df.columns, "Submission must contain 'fname' column"

    print("Trainer verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
