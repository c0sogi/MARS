import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import WORKING_DIR, SUBMISSION_PATH, DEVICE
from library.utils import rle_encode, fbeta_score, optimize_threshold
from library.losses import BCEDiceLoss
from library.models import build_model
from library.data import get_loaders
from library.train import train_model, set_seed
from library.inference import run_inference


def test_utils():
    print("\n=== Testing Utils ===")
    # Test RLE Encoding
    # Mask: 0 1 1 1 0 0 1 0
    # 1-based indices: 2,3,4 and 7
    # Expected: 2 3 7 1
    mask = np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    rle = rle_encode(mask)
    print(f"RLE Output: {rle}")
    assert rle == "2 3 7 1", f"RLE encoding incorrect. Expected '2 3 7 1', got '{rle}'"

    # Test F-beta Score
    preds = torch.tensor([0.1, 0.9, 0.8, 0.2])
    targets = torch.tensor([0, 1, 1, 0])
    score = fbeta_score(preds, targets, threshold=0.5, beta=0.5)
    print(f"F0.5 Score (Perfect): {score}")
    assert np.isclose(score, 1.0), "F0.5 score should be 1.0 for perfect predictions"

    # Test Threshold Optimization
    best_thresh, best_score = optimize_threshold(preds, targets)
    print(f"Optimized Threshold: {best_thresh}, Score: {best_score}")
    assert 0.0 < best_thresh < 1.0, "Threshold should be in (0, 1)"
    print("Utils tests passed.")


def test_loss_and_model():
    print("\n=== Testing Model and Loss ===")
    # 1. Build Model
    model = build_model()
    model.to(DEVICE)
    print(f"Model {model.__class__.__name__} built successfully.")

    # 2. Dummy Input (Batch=2, Channels=3, H=512, W=512)
    dummy_input = torch.randn(2, 3, 512, 512).to(DEVICE)

    # 3. Forward Pass
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}")
    assert output.shape == (
        2,
        1,
        512,
        512,
    ), f"Expected output (2, 1, 512, 512), got {output.shape}"

    # 4. Loss Calculation
    criterion = BCEDiceLoss()
    dummy_target = torch.randint(0, 2, (2, 1, 512, 512)).float().to(DEVICE)

    loss = criterion(output, dummy_target)
    print(f"Loss Value: {loss.item()}")

    # 5. Backward Pass
    loss.backward()
    print("Backward pass successful.")

    # Clean up
    del model, dummy_input, dummy_target, output, loss
    torch.cuda.empty_cache()
    print("Model and Loss tests passed.")


def test_data_loading():
    print("\n=== Testing Data Loading ===")
    # Use a very small sample size for speed
    train_loader, val_loader, test_loader = get_loaders(
        max_train_samples=4,
        batch_size=2,
        num_workers=0,  # Avoid multiprocessing overhead for small test
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Fetch one batch
    images, labels = next(iter(train_loader))
    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    assert images.shape == (2, 3, 512, 512), "Incorrect image batch shape"
    assert labels.shape == (2, 1, 512, 512), "Incorrect label batch shape"
    assert images.dtype == torch.float32, "Images should be float32"

    print("Data loading tests passed.")


def test_training_pipeline():
    print("\n=== Testing Training Pipeline ===")
    # Run a minimal training loop
    # max_train_samples=4 ensures very fast epochs
    # baseline_score=-1.0 ensures submission generation is attempted regardless of score
    best_score = train_model(
        max_train_samples=4, num_epochs=1, patience=1, baseline_score=-1.0
    )

    print(f"Training completed. Best Score: {best_score}")

    # Check if model was saved
    model_path = os.path.join(WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    print("Training pipeline tests passed.")


def test_inference_pipeline():
    print("\n=== Testing Inference Pipeline ===")
    # Ensure we have a model (created in previous step)
    model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run training test first.")

    # Run inference explicitly
    # This tests the TTA and tiling logic in library.inference
    run_inference(
        model_path=model_path, threshold=0.5, submission_output="submission_test.csv"
    )

    assert os.path.exists("submission_test.csv"), "Inference output file not created"

    # Check content format
    df = pd.read_csv("submission_test.csv")
    print("Submission Head:")
    print(df.head())
    assert (
        "Id" in df.columns and "Predicted" in df.columns
    ), "Submission columns missing"

    print("Inference pipeline tests passed.")


if __name__ == "__main__":
    # Ensure reproducible runs
    set_seed(42)

    # Clean working directory to start fresh (optional but good for testing)
    if os.path.exists(WORKING_DIR):
        # Don't delete if it contains cached numpy files that take time to generate,
        # but for this demo we assume we can write freely.
        # We'll just ensure it exists.
        os.makedirs(WORKING_DIR, exist_ok=True)

    try:
        test_utils()
        test_loss_and_model()
        test_data_loading()
        test_training_pipeline()
        test_inference_pipeline()

        print("\nAll integration tests passed successfully.")

    except Exception as e:
        print(f"\nTest Failed with error: {e}")
        raise e
