import os
import torch
import pandas as pd
import numpy as np
from library.utils import set_seed, get_device
from library.dataset import get_dataset, IcebergDataset
from library.model import MicroSEResNet
from library.trainer import train_cross_validation
from library.inference import run_inference


def main():
    # 1. Setup
    print("=== Step 1: Initialization ===")
    set_seed(42)
    device = get_device()
    print(f"Device selected: {device}")

    # Define temporary directories for this demo to avoid overwriting production files
    DEMO_WORK_DIR = "./working/demo_run"
    DEMO_SUBMISSION_PATH = "./working/demo_submission.csv"
    os.makedirs(DEMO_WORK_DIR, exist_ok=True)

    # 2. Dataset Verification
    print("\n=== Step 2: Dataset Loading and Verification ===")
    # Load training dataset (this will use cached data in ./working/idea_6/ if available)
    train_ds = get_dataset("train", load_cached_data=True)

    print(f"Training dataset size: {len(train_ds)}")

    # Verify a single sample
    img, angle, label = train_ds[0]

    # Check shapes
    # Image should be (3, 75, 75) -> [Band1, Band2, Avg]
    assert img.shape == (
        3,
        75,
        75,
    ), f"Expected image shape (3, 75, 75), got {img.shape}"
    # Angle should be a tensor of shape (1,)
    assert angle.shape == (1,), f"Expected angle shape (1,), got {angle.shape}"
    # Label should be a scalar tensor
    assert label.numel() == 1, "Expected scalar label"

    print("Dataset verification passed: Shapes are correct.")

    # 3. Model Architecture Check
    print("\n=== Step 3: Model Instantiation and Forward Pass ===")
    model = MicroSEResNet(dropout_rate=0.5).to(device)

    # Create dummy batch: Batch Size=4, Channels=3, H=75, W=75
    dummy_input = torch.randn(4, 3, 75, 75).to(device)
    dummy_angle = torch.tensor([30.0, 35.0, 40.0, 45.0]).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input, dummy_angle)

    # Verify output
    # Expected shape: (Batch Size, 1)
    assert output.shape == (4, 1), f"Expected output shape (4, 1), got {output.shape}"
    # Expected values: Probabilities between 0 and 1 (Sigmoid)
    assert output.min() >= 0 and output.max() <= 1, "Output values out of range [0, 1]"

    print("Model forward pass verification passed.")

    # 4. Training Loop Demonstration (Minimal CV)
    print("\n=== Step 4: Running Minimal Cross-Validation ===")
    # We run 2 folds, 1 epoch each, with a small batch size for speed
    n_folds = 2
    epochs = 1
    batch_size = 16

    cv_scores = train_cross_validation(
        n_folds=n_folds,
        batch_size=batch_size,
        lr=1e-3,
        epochs=epochs,
        patience=1,
        save_dir=DEMO_WORK_DIR,
    )

    print(f"CV Scores: {cv_scores}")
    assert len(cv_scores) == n_folds, "CV scores length mismatch"

    # Verify model files were created
    for i in range(n_folds):
        model_path = os.path.join(DEMO_WORK_DIR, f"fold_{i}", "model_best.pth")
        assert os.path.exists(model_path), f"Model checkpoint not found: {model_path}"

    print("Training demonstration passed: Models saved successfully.")

    # 5. Inference and Submission
    print("\n=== Step 5: Inference on Test Set ===")
    # We limit samples to 20 for quick execution
    limit_samples = 20

    run_inference(
        batch_size=batch_size,
        folds=n_folds,
        limit_samples=limit_samples,
        model_dir=DEMO_WORK_DIR,
        output_path=DEMO_SUBMISSION_PATH,
    )

    # Verify submission file
    assert os.path.exists(DEMO_SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(DEMO_SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    # Check columns
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Submission columns mismatch"
    # Check length
    assert (
        len(df_sub) == limit_samples
    ), f"Expected {limit_samples} predictions, got {len(df_sub)}"

    print("Inference demonstration passed: Submission file generated correctly.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
