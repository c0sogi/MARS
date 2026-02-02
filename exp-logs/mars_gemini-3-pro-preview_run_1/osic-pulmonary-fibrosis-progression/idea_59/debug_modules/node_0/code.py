import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import components from the provided library files
from library.utils import seed_everything, compute_metric, get_device
from library.dataset import get_dataloaders, LungDataset
from library.model import BCSLNet
from library.train import run_training
from library.predict import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def demo_metric_logic():
    """
    Demonstrates and validates the custom metric calculation.
    """
    print("\n=== Demo: Metric Calculation ===")

    # Create dummy data for a "perfect" prediction scenario
    # True values
    y_true = torch.tensor([2500.0, 3000.0], dtype=torch.float32)
    # Predicted values (identical to true)
    y_pred = torch.tensor([2500.0, 3000.0], dtype=torch.float32)
    # Confidence (sigma), clipped at 70 internally, so we provide 100
    sigma = torch.tensor([100.0, 100.0], dtype=torch.float32)

    # Calculate metric
    # Formula: - (sqrt(2) * delta / sigma) - ln(sqrt(2) * sigma)
    # Since delta is 0, first term is 0.
    # Metric = - ln(sqrt(2) * 100) = - ln(141.421) ≈ -4.9517
    metric = compute_metric(y_true, y_pred, sigma)

    print(f"Calculated Metric for perfect match: {metric.item():.4f}")

    # Validation
    expected_val = -np.log(np.sqrt(2) * 100.0)
    assert np.isclose(
        metric.item(), expected_val, atol=1e-3
    ), f"Metric calculation mismatch. Expected {expected_val}, got {metric.item()}"

    print("Metric logic verified successfully.")


def demo_data_pipeline():
    """
    Demonstrates data loading and validates tensor shapes.
    """
    print("\n=== Demo: Data Pipeline ===")

    # Initialize dataloaders with a small batch size
    batch_size = 4
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, num_workers=0
    )

    print(f"Training Batches: {len(train_loader)}")
    print(f"Validation Batches: {len(val_loader)}")

    # Fetch a single batch
    batch = next(iter(train_loader))

    # Check for required keys
    required_keys = ["axial", "coronal", "tabular", "delta_week", "base_fvc", "target"]
    for k in required_keys:
        assert k in batch, f"Batch missing key: {k}"

    # Validate Shapes
    # Image inputs should be (B, 3, 224, 224)
    axial = batch["axial"]
    coronal = batch["coronal"]
    # Tabular input should be (B, 7)
    tabular = batch["tabular"]

    print(f"Axial Image Shape: {axial.shape}")
    print(f"Tabular Data Shape: {tabular.shape}")

    assert axial.shape == (batch_size, 3, 224, 224), "Incorrect Axial Image dimensions"
    assert coronal.shape == (
        batch_size,
        3,
        224,
        224,
    ), "Incorrect Coronal Image dimensions"
    assert tabular.shape == (batch_size, 7), "Incorrect Tabular data dimensions"

    print("Data pipeline verified successfully.")
    return batch


def demo_model_forward(batch):
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n=== Demo: Model Architecture ===")

    device = get_device()
    model = BCSLNet().to(device)
    model.eval()

    # Move batch to device
    axial = batch["axial"].to(device)
    coronal = batch["coronal"].to(device)
    tabular = batch["tabular"].to(device)
    delta_week = batch["delta_week"].to(device)
    base_fvc = batch["base_fvc"].to(device)

    # Perform forward pass
    with torch.no_grad():
        fvc_pred, sigma_pred = model(axial, coronal, tabular, delta_week, base_fvc)

    print(f"Prediction FVC Shape: {fvc_pred.shape}")
    print(f"Prediction Sigma Shape: {sigma_pred.shape}")

    # Validate outputs
    batch_size = axial.size(0)
    assert fvc_pred.shape == (batch_size,), "FVC output shape mismatch"
    assert sigma_pred.shape == (batch_size,), "Sigma output shape mismatch"

    # Sigma must be positive (Softplus activation)
    assert (sigma_pred > 0).all(), "Model produced non-positive confidence values"

    print("Model forward pass verified successfully.")


def demo_training_execution():
    """
    Demonstrates the training loop using the library function.
    """
    print("\n=== Demo: Training Execution ===")

    save_path = "./working/demo_model.pth"

    # Run training for 1 epoch with a small batch size to ensure speed
    # The library function handles the loop, loss, and saving
    best_metric = run_training(epochs=1, batch_size=8, patience=1, save_path=save_path)

    print(f"Training completed. Best Validation Metric: {best_metric:.4f}")

    # Verify checkpoint creation
    assert os.path.exists(save_path), f"Model checkpoint not found at {save_path}"
    print("Training execution verified successfully.")

    return save_path


def demo_inference_execution(model_path):
    """
    Demonstrates inference and submission file generation.
    """
    print("\n=== Demo: Inference & Submission ===")

    output_path = "./submission/demo_submission.csv"

    # Generate submission using the trained model
    # limit_batches is used to speed up the demonstration
    generate_submission(
        model_path=model_path, output_path=output_path, batch_size=8, limit_batches=5
    )

    # Validate submission file
    assert os.path.exists(output_path), "Submission file was not created"

    df = pd.read_csv(output_path)
    print(f"Submission file loaded. Rows: {len(df)}")
    print(df.head(3))

    required_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in required_cols:
        assert col in df.columns, f"Submission missing column: {col}"

    print("Inference execution verified successfully.")


def main():
    # Set seeds for reproducibility
    seed_everything(42)

    # 1. Verify Metric
    demo_metric_logic()

    # 2. Verify Data Loading
    batch = demo_data_pipeline()

    # 3. Verify Model
    demo_model_forward(batch)

    # 4. Verify Training
    model_path = demo_training_execution()

    # 5. Verify Inference
    demo_inference_execution(model_path)

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
