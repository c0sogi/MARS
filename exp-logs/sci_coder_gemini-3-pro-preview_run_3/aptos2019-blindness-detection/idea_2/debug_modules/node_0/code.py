import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.data import get_dataloaders
from library.model import EfficientNetRegressor
from library.train import run


def main():
    print("=== Starting Diabetic Retinopathy Task Demonstration ===")

    # 1. Setup and Configuration
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Optimize Config for Speed/Demo purposes
    # Note: Some Config attributes are bound as default arguments at import time (like in model __init__),
    # but those accessed inside functions (like in get_dataloaders) can be modified here.
    Config.IMG_SIZE = 128  # Reduce image size for faster processing
    Config.BATCH_SIZE = 8  # Reduce batch size for the demo
    Config.NUM_WORKERS = 2  # Adjust workers

    print(
        f"Configuration updated: IMG_SIZE={Config.IMG_SIZE}, BATCH_SIZE={Config.BATCH_SIZE}"
    )

    # 2. Verify Data Loading
    print("\n[1/5] Verifying Data Loading...")
    # Request debug loaders which return a small subset of data (100 samples)
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, batch_size=Config.BATCH_SIZE
    )

    # Fetch one batch from training loader
    try:
        images, labels = next(iter(train_loader))
        print(f"   Train Batch - Images: {images.shape}, Labels: {labels.shape}")

        # Validate shapes and types
        assert (
            images.shape[0] == Config.BATCH_SIZE
        ), f"Expected batch size {Config.BATCH_SIZE}, got {images.shape[0]}"
        assert images.shape[1] == 3, "Expected 3 channels (RGB)"
        assert images.shape[2] == Config.IMG_SIZE, f"Expected height {Config.IMG_SIZE}"
        assert images.shape[3] == Config.IMG_SIZE, f"Expected width {Config.IMG_SIZE}"
        assert (
            labels.dtype == torch.float
        ), "Labels should be float tensors for regression"

        print("   Data loading assertions passed.")
    except Exception as e:
        print(f"   Data loading failed: {e}")
        raise e

    # 3. Verify Model Architecture
    print("\n[2/5] Verifying Model Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")

    # Instantiate model (pretrained=False for this quick unit test to avoid download overhead)
    model = EfficientNetRegressor(pretrained=False)
    model.to(device)
    model.eval()

    # Create dummy input tensor
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)

    try:
        with torch.no_grad():
            output = model(dummy_input)

        print(f"   Model Output Shape: {output.shape}")

        # Validate output
        assert output.shape == (
            2,
            1,
        ), f"Expected output shape (2, 1), got {output.shape}"
        assert not torch.isnan(output).any(), "Model output contains NaNs"

        print("   Model architecture assertions passed.")
    except Exception as e:
        print(f"   Model verification failed: {e}")
        raise e

    # 4. Verify Metric (Quadratic Weighted Kappa)
    print("\n[3/5] Verifying Metric Calculation...")

    # Case 1: Perfect Agreement
    y_true_perfect = np.array([0, 1, 2, 3, 4])
    y_pred_perfect = np.array([0, 1, 2, 3, 4])
    score_perfect = quadratic_weighted_kappa(y_true_perfect, y_pred_perfect)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Expected 1.0 for perfect agreement, got {score_perfect}"

    # Case 2: Regression Float Handling (should round correctly)
    y_true_float = np.array([0, 4])
    y_pred_float = np.array([0.2, 3.8])  # Should round to 0 and 4
    score_float = quadratic_weighted_kappa(y_true_float, y_pred_float)
    assert np.isclose(
        score_float, 1.0
    ), f"Expected 1.0 for rounded float agreement, got {score_float}"

    print("   Metric assertions passed.")

    # 5. Verify Full Training Pipeline
    print("\n[4/5] Running Training Pipeline (Debug Mode)...")
    # Run the main execution function from library.train
    # debug=True limits data to 100 samples
    # epochs=1 ensures quick execution
    try:
        best_qwk = run(debug=True, epochs=1)
        print(f"   Pipeline completed. Best Validation QWK: {best_qwk:.4f}")
    except Exception as e:
        print(f"   Training pipeline failed: {e}")
        raise e

    # 6. Verify Submission Output
    print("\n[5/5] Verifying Submission File...")
    submission_path = Config.SUBMISSION_PATH

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    sub_df = pd.read_csv(submission_path)
    print(f"   Submission loaded. Shape: {sub_df.shape}")
    print(f"   Columns: {sub_df.columns.tolist()}")

    # Validate submission content
    # In debug mode, the test set is also clipped to Config.DEBUG_SUBSET_SIZE (100)
    expected_rows = Config.DEBUG_SUBSET_SIZE
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(sub_df)}"
    assert "id_code" in sub_df.columns, "Missing 'id_code' column"
    assert "diagnosis" in sub_df.columns, "Missing 'diagnosis' column"
    assert pd.api.types.is_integer_dtype(
        sub_df["diagnosis"]
    ), "Diagnosis column must be integer"

    print("   Submission assertions passed.")

    print("\n=== All demonstrations completed successfully ===")


if __name__ == "__main__":
    main()
