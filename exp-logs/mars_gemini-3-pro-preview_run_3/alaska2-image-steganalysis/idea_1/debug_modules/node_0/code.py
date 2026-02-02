import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library files
from library.config import DEVICE, SUBMISSION_PATH, IDEA_DIR
from library.utils import seed_everything, weighted_auc_score
from library.dataset import get_loaders
from library.model import MonoResidualResNet
from library.engine import run_training


def test_metric_logic():
    """
    Validates the Weighted AUC metric implementation.
    """
    print("\n[1/4] Testing Metric Logic...")

    # Case 1: Perfect prediction
    y_true = np.array([0, 0, 1, 1])
    y_score_perfect = np.array([0.1, 0.4, 0.8, 0.9])
    score = weighted_auc_score(y_true, y_score_perfect)
    print(f"   Perfect Score: {score}")
    assert np.isclose(score, 1.0), "Metric failed on perfect predictions"

    # Case 2: Inverse prediction (should be 0.0)
    y_score_worst = np.array([0.9, 0.8, 0.4, 0.1])
    score = weighted_auc_score(y_true, y_score_worst)
    print(f"   Worst Score: {score}")
    assert np.isclose(score, 0.0), "Metric failed on inverted predictions"

    print("   Metric logic verified.")


def test_data_pipeline():
    """
    Validates the data loading pipeline, including shapes and value ranges.
    """
    print("\n[2/4] Testing Data Pipeline...")

    # Use debug=True and a small sample size for speed
    batch_size = 4
    train_loader, val_loader, test_loader = get_loaders(
        debug=True,
        debug_sample_size=20,  # Small subset
        batch_size=batch_size,
        num_workers=2,
    )

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Verify Shapes
    # Expected: (Batch, 1, 512, 512) for Mono-channel input
    print(f"   Input Shape: {images.shape}")
    print(f"   Label Shape: {labels.shape}")

    assert images.dim() == 4, "Images must be 4D tensors"
    assert images.size(1) == 1, "Model expects 1 channel (Luminance)"
    assert images.size(2) == 512 and images.size(3) == 512, "Image resolution mismatch"
    assert labels.size(0) == batch_size, "Batch size mismatch"

    # Verify Value Range (Normalized to [0, 1])
    # Note: Due to preprocessing, values should be float32
    min_val, max_val = images.min().item(), images.max().item()
    print(f"   Value Range: [{min_val:.4f}, {max_val:.4f}]")

    assert 0.0 <= min_val and max_val <= 1.0, "Images not properly normalized to [0, 1]"

    print("   Data pipeline verified.")
    return train_loader, val_loader, test_loader


def test_model_architecture():
    """
    Validates the model architecture, specifically the Fixed High-Pass Filter.
    """
    print("\n[3/4] Testing Model Architecture...")

    # Initialize model (pretrained=False for speed in this demo)
    model = MonoResidualResNet(pretrained=False)
    model.to(DEVICE)
    model.eval()

    # Check 1: Fixed High-Pass Filter
    # The first layer 'preprocessing' should have gradients disabled
    hpf_layer = model.preprocessing.conv
    print(f"   HPF Layer Trainable: {hpf_layer.weight.requires_grad}")
    assert (
        hpf_layer.weight.requires_grad is False
    ), "High-Pass Filter weights should be frozen"

    # Check 2: Forward Pass
    dummy_input = torch.randn(2, 1, 512, 512).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"   Output Shape: {output.shape}")
    assert output.shape == (2, 1), "Output shape mismatch (should be [Batch, 1])"

    print("   Model architecture verified.")
    return model


def test_training_engine(model, train_loader, val_loader, test_loader):
    """
    Validates the training loop and submission generation.
    """
    print("\n[4/4] Testing Training Engine & Inference...")

    # Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run Training
    # We use 1 epoch and a small patience to ensure it runs quickly
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=None,  # No scheduler for this short test
        device=DEVICE,
        num_epochs=1,
        early_stopping_patience=1,
        label_smoothing=0.05,
    )

    # Verify Submission File
    if os.path.exists(SUBMISSION_PATH):
        df = pd.read_csv(SUBMISSION_PATH)
        print(f"   Submission generated with {len(df)} rows.")

        # Check columns
        assert (
            "Id" in df.columns and "Label" in df.columns
        ), "Submission missing required columns"

        # Check if predictions are within valid range (Sigmoid output implies 0-1)
        # Note: TTA averages probabilities, so they must be in [0, 1]
        preds = df["Label"].values
        assert np.all(
            (preds >= 0) & (preds <= 1)
        ), "Predictions out of probability range [0, 1]"

        print("   Submission file verified.")
    else:
        raise FileNotFoundError(f"Submission file not found at {SUBMISSION_PATH}")


if __name__ == "__main__":
    # 1. Reproducibility
    seed_everything(42)

    # 2. Metric Validation
    test_metric_logic()

    # 3. Data Pipeline Validation
    # We keep the loaders to use in the training test
    train_loader, val_loader, test_loader = test_data_pipeline()

    # 4. Model Validation
    model = test_model_architecture()

    # 5. Engine Validation
    test_training_engine(model, train_loader, val_loader, test_loader)

    print("\nAll demonstrations completed successfully.")
