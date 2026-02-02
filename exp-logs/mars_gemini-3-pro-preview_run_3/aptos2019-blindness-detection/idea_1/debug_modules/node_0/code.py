import os
import sys
import warnings
import torch
import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.dataset import RetinopathyDataset, get_transforms
from library.model import ResNet18Regression
from library.engine import get_dataloaders, train, evaluate, generate_submission


def run_demo():
    print("=== Starting Diabetic Retinopathy Task Demo ===")

    # 1. Setup and Reproducibility
    # Set fixed seeds for reproducibility
    seed_everything(Config.SEED)
    print(f"Device selected: {Config.DEVICE}")

    # Define temporary paths for this demo run
    demo_model_path = os.path.join(Config.WORKING_DIR, "demo_resnet18.pth")
    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # ==========================================
    # 2. Metric Verification (Unit Test)
    # ==========================================
    print("\n[1/5] Verifying Quadratic Weighted Kappa (QWK) metric logic...")

    # Case A: Perfect agreement
    y_true_perfect = [0, 1, 2, 3, 4]
    y_pred_perfect = [0.1, 1.1, 2.1, 3.1, 4.1]  # Regression outputs close to integers
    score_perfect = compute_qwk(y_true_perfect, y_pred_perfect)

    # Case B: Complete disagreement (inverse)
    y_true_bad = [0, 1, 2, 3, 4]
    y_pred_bad = [4.0, 3.0, 2.0, 1.0, 0.0]
    score_bad = compute_qwk(y_true_bad, y_pred_bad)

    print(f"  Perfect Agreement Score: {score_perfect:.4f}")
    print(f"  Bad Agreement Score:     {score_bad:.4f}")

    # Assertions
    if not np.isclose(score_perfect, 1.0, atol=0.05):
        raise AssertionError(
            f"QWK calculation failed for perfect agreement. Got {score_perfect}"
        )
    if score_bad >= 0.0:
        raise AssertionError(
            f"QWK calculation failed for bad agreement. Should be negative, got {score_bad}"
        )

    print("  Metric verification passed.")

    # ==========================================
    # 3. Data Loading
    # ==========================================
    print("\n[2/5] Loading DataLoaders (Debug Mode)...")

    # We use debug=True and a small debug_size to ensure the script runs quickly
    batch_size = 8
    debug_size = 32

    train_loader, val_loader, test_loader, test_df = get_dataloaders(
        batch_size=batch_size, debug=True, debug_size=debug_size
    )

    # Verify DataLoaders
    try:
        images, labels = next(iter(train_loader))
    except StopIteration:
        raise AssertionError("Train loader is empty.")

    print(f"  Train batch shape: {images.shape}")
    print(f"  Labels batch shape: {labels.shape}")

    # Assertions for shapes
    expected_shape = (batch_size, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    if images.shape != expected_shape:
        raise AssertionError(
            f"Image batch shape mismatch. Expected {expected_shape}, got {images.shape}"
        )

    if labels.shape != (batch_size,):
        raise AssertionError(
            f"Label batch shape mismatch. Expected {(batch_size,)}, got {labels.shape}"
        )

    print("  Data loading verification passed.")

    # ==========================================
    # 4. Model Initialization & Sanity Check
    # ==========================================
    print("\n[3/5] Initializing Model...")

    model = ResNet18Regression(pretrained=True)
    model.to(Config.DEVICE)

    # Run a dummy forward pass to verify architecture
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(
            Config.DEVICE
        )
        dummy_output = model(dummy_input)

    print(f"  Model output shape: {dummy_output.shape}")

    # Assertion: Output should be (Batch_Size, 1) for regression
    if dummy_output.shape != (2, 1):
        raise AssertionError(
            f"Model output shape incorrect. Expected (2, 1), got {dummy_output.shape}"
        )

    print("  Model initialization verification passed.")

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print("\n[4/5] Starting Training (1 Epoch)...")

    # Train for 1 epoch to demonstrate functionality without timeout
    trained_model = train(
        train_loader,
        val_loader,
        epochs=1,
        lr=1e-4,
        device=Config.DEVICE,
        save_path=demo_model_path,
    )

    # Check if model file was created
    if not os.path.exists(demo_model_path):
        raise FileNotFoundError(f"Model file was not saved at {demo_model_path}")

    print(f"  Training complete. Model saved to {demo_model_path}")

    # Evaluate on validation set
    print("  Evaluating on validation set...")
    val_loss, val_qwk = evaluate(trained_model, val_loader, device=Config.DEVICE)

    # Basic sanity check on metrics
    if not isinstance(val_loss, float) or not isinstance(val_qwk, float):
        raise TypeError("Evaluation metrics should be floats.")

    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation QWK:  {val_qwk:.4f}")

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n[5/5] Generating Submission...")

    generate_submission(
        trained_model,
        test_loader,
        test_df,
        device=Config.DEVICE,
        output_path=demo_submission_path,
    )

    # Verify submission file
    if not os.path.exists(demo_submission_path):
        raise FileNotFoundError(f"Submission file not found at {demo_submission_path}")

    sub_df = pd.read_csv(demo_submission_path)
    print(f"  Submission file loaded. Shape: {sub_df.shape}")
    print(f"  Columns: {sub_df.columns.tolist()}")

    # Assertions for submission format
    required_cols = ["id_code", "diagnosis"]
    if not all(col in sub_df.columns for col in required_cols):
        raise AssertionError(
            f"Submission missing required columns. Found: {sub_df.columns}"
        )

    if len(sub_df) != len(test_df):
        raise AssertionError(
            f"Submission row count mismatch. Expected {len(test_df)}, got {len(sub_df)}"
        )

    # Check values are integers 0-4
    unique_vals = sub_df["diagnosis"].unique()
    if not all(v in [0, 1, 2, 3, 4] for v in unique_vals):
        raise AssertionError(f"Invalid diagnosis values found: {unique_vals}")

    print("  Submission verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
