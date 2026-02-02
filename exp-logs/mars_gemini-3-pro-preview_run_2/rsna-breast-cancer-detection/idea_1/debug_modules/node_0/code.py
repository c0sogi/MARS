import os
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library
from library.config import seed_everything, CACHE_DIR, SUBMISSION_DIR, WORKING_DIR
from library.utils import probabilistic_f1
from library.data import get_dataloaders
from library.model import HybridEfficientNet
from library.train import run_training
from library.inference import predict_and_submit


def test_probabilistic_f1():
    """
    Demonstrates and validates the probabilistic F1 score calculation.
    """
    print("\n=== Testing Probabilistic F1 Metric ===")

    # Case 1: Perfect prediction
    y_true = torch.tensor([1, 0, 1, 0])
    y_pred = torch.tensor([1.0, 0.0, 1.0, 0.0])
    score = probabilistic_f1(y_true, y_pred)
    print(f"Perfect Prediction Score: {score.item()}")
    assert torch.isclose(
        score, torch.tensor(1.0)
    ), "Score should be 1.0 for perfect predictions"

    # Case 2: Complete failure
    y_pred_bad = torch.tensor([0.0, 1.0, 0.0, 1.0])
    score_bad = probabilistic_f1(y_true, y_pred_bad)
    print(f"Bad Prediction Score: {score_bad.item()}")
    assert torch.isclose(
        score_bad, torch.tensor(0.0)
    ), "Score should be 0.0 for completely wrong predictions"

    # Case 3: Probabilistic values
    # TP = 0.8*1 + 0.6*1 = 1.4
    # FP = 0.8*0 + 0.6*0 + 0.2*1 + 0.1*1 = 0.3
    # Total Positives = 2
    # Precision = 1.4 / (1.4 + 0.3) = 0.8235
    # Recall = 1.4 / 2 = 0.7
    # F1 = 2 * (0.8235 * 0.7) / (0.8235 + 0.7) = 1.1529 / 1.5235 = 0.7567
    y_true_mixed = torch.tensor([1, 1, 0, 0])
    y_pred_mixed = torch.tensor([0.8, 0.6, 0.2, 0.1])
    score_mixed = probabilistic_f1(y_true_mixed, y_pred_mixed)
    print(f"Mixed Prediction Score: {score_mixed.item()}")

    # Validation
    p_tp = (y_true_mixed * y_pred_mixed).sum()
    p_fp = ((1 - y_true_mixed) * y_pred_mixed).sum()
    p_prec = p_tp / (p_tp + p_fp + 1e-7)
    p_rec = p_tp / (y_true_mixed.sum() + 1e-7)
    expected_f1 = 2 * (p_prec * p_rec) / (p_prec + p_rec + 1e-7)

    assert torch.isclose(
        score_mixed, expected_f1
    ), "Calculated pF1 does not match expected manual calculation"
    print("Metric validation successful.")


def test_data_pipeline():
    """
    Demonstrates data loading and verifies batch shapes.
    """
    print("\n=== Testing Data Pipeline ===")

    # Use debug=True to load a small subset quickly
    # load_cached_data=False forces the processing logic to run
    train_loader, val_loader, test_loader, num_features = get_dataloaders(
        load_cached_data=False, debug=True
    )

    print(f"Number of tabular features detected: {num_features}")
    assert num_features > 0, "No tabular features were extracted."

    # Fetch one batch from training loader
    (images, tabular), targets = next(iter(train_loader))

    print(f"Image Batch Shape: {images.shape}")
    print(f"Tabular Batch Shape: {tabular.shape}")
    print(f"Targets Batch Shape: {targets.shape}")

    # Assertions
    assert len(images.shape) == 4, "Images should be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels (RGB)"
    assert (
        images.shape[2] == 512 and images.shape[3] == 512
    ), "Images should be resized to 512x512"
    assert (
        tabular.shape[1] == num_features
    ), "Tabular data dim should match num_features"
    assert (
        targets.shape[0] == images.shape[0]
    ), "Batch size mismatch between inputs and targets"

    return num_features


def test_model_architecture(num_features):
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n=== Testing Model Architecture ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridEfficientNet(
        num_tabular_features=num_features,
        backbone_name="efficientnet_b0",
        pretrained=False,  # False for speed in test, though True is default
    ).to(device)

    # Create dummy input
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 512, 512).to(device)
    dummy_tabular = torch.randn(batch_size, num_features).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model((dummy_images, dummy_tabular))

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (batch_size, 1), "Output shape should be (Batch_Size, 1)"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    print("Model forward pass successful.")


def test_training_loop():
    """
    Demonstrates the training function.
    """
    print("\n=== Testing Training Loop ===")

    save_path = os.path.join(CACHE_DIR, "test_model.pth")

    # Run training for 1 epoch in debug mode
    best_score = run_training(
        debug=True,
        load_cached_data=True,  # Use cache generated in previous step
        epochs=1,
        save_path=save_path,
    )

    print(f"Training finished. Best Validation pF1: {best_score}")

    # Assertions
    assert os.path.exists(save_path), f"Model file was not saved to {save_path}"
    assert isinstance(best_score, float), "Score should be a float"

    return save_path


def test_inference_pipeline(model_path):
    """
    Demonstrates the inference and submission generation.
    """
    print("\n=== Testing Inference Pipeline ===")

    output_csv = os.path.join(SUBMISSION_DIR, "test_submission.csv")

    # Run inference
    submission_df = predict_and_submit(
        model_path=model_path, output_path=output_csv, debug=True, load_cached_data=True
    )

    # Assertions
    assert os.path.exists(output_csv), "Submission CSV was not created"
    assert "prediction_id" in submission_df.columns, "Missing prediction_id column"
    assert "cancer" in submission_df.columns, "Missing cancer column"
    assert (
        not submission_df["prediction_id"].duplicated().any()
    ), "Duplicate prediction_ids found"

    # Check probability range
    probs = submission_df["cancer"].values
    assert (probs >= 0).all() and (probs <= 1).all(), "Predictions out of [0, 1] range"

    print("Inference pipeline successful.")
    print(submission_df.head())


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    # 1. Test Metrics
    test_probabilistic_f1()

    # 2. Test Data Loading
    # Returns num_features needed for model init
    num_features = test_data_pipeline()

    # 3. Test Model
    test_model_architecture(num_features)

    # 4. Test Training
    # Returns path to the saved model
    saved_model_path = test_training_loop()

    # 5. Test Inference
    test_inference_pipeline(saved_model_path)

    print("\nAll demonstrations and validations passed successfully.")
