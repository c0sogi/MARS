import os
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library files
from library.utils import seed_everything, LaplaceLogLikelihood
from library.data import get_dataloaders
from library.model import TSCPNet, negative_laplace_log_likelihood
from library.train import run_training
from library.predict import generate_submission


def test_data_loading(device):
    print("\n--- Testing Data Loading ---")
    # Use debug=True to load a small subset of data for speed
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=4, num_workers=0, debug=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Fetch one batch from training loader
    batch = next(iter(train_loader))

    # Verify keys
    expected_keys = [
        "img_ax",
        "img_cor",
        "tabular",
        "target",
        "weeks",
        "base_fvc",
        "patient_id",
    ]
    for key in expected_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Verify shapes
    # Images should be (B, 3, 224, 224) - assuming resize in data.py
    # Note: data.py generates tri-slabs which are 3 channels.
    img_ax = batch["img_ax"]
    img_cor = batch["img_cor"]
    tabular = batch["tabular"]
    target = batch["target"]

    print(f"Img Axial Shape: {img_ax.shape}")
    print(f"Img Coronal Shape: {img_cor.shape}")
    print(f"Tabular Shape: {tabular.shape}")

    assert img_ax.dim() == 4 and img_ax.shape[1] == 3, "Axial image incorrect shape"
    assert img_cor.dim() == 4 and img_cor.shape[1] == 3, "Coronal image incorrect shape"
    assert (
        tabular.dim() == 2 and tabular.shape[1] == 7
    ), "Tabular data incorrect shape (expected 7 features)"
    assert target.dim() == 1, "Target should be 1D tensor"

    return batch


def test_model_forward(device, batch):
    print("\n--- Testing Model Forward Pass ---")
    model = TSCPNet().to(device)
    model.eval()

    img_ax = batch["img_ax"].to(device)
    img_cor = batch["img_cor"].to(device)
    tabular = batch["tabular"].to(device)

    with torch.no_grad():
        alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

    print(f"Alpha shape: {alpha.shape}")
    print(f"Sigma Base shape: {sigma_base.shape}")
    print(f"Sigma Growth shape: {sigma_growth.shape}")

    batch_size = img_ax.size(0)
    assert alpha.shape == (batch_size,), "Alpha output shape mismatch"
    assert sigma_base.shape == (batch_size,), "Sigma Base output shape mismatch"
    assert sigma_growth.shape == (batch_size,), "Sigma Growth output shape mismatch"

    # Verify non-negativity constraints (Softplus used in model for sigmas)
    assert torch.all(sigma_base >= 0), "Sigma base should be positive"
    assert torch.all(sigma_growth >= 0), "Sigma growth should be positive"

    return model, alpha, sigma_base, sigma_growth


def test_loss_and_metric(batch, outputs, device):
    print("\n--- Testing Loss and Metric ---")
    alpha, sigma_base, sigma_growth = outputs

    target = batch["target"].to(device)
    weeks = batch["weeks"].to(device)
    base_fvc = batch["base_fvc"].to(device)

    # Reconstruct predictions
    fvc_pred = base_fvc + alpha * weeks
    sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

    # Calculate Loss
    loss = negative_laplace_log_likelihood(target, fvc_pred, sigma_pred)
    print(f"Calculated Loss: {loss.item()}")
    assert torch.isfinite(loss), "Loss is not finite"

    # Calculate Metric
    # Metric function expects numpy or tensor, returns float
    metric = LaplaceLogLikelihood(target, fvc_pred, sigma_pred)
    print(f"Calculated Metric: {metric}")
    assert np.isfinite(metric), "Metric is not finite"


def test_training_loop():
    print("\n--- Testing Training Loop (Integration) ---")
    # Run for 1 epoch with debug=True to ensure speed
    # This uses library.train.run_training
    best_model_path = run_training(epochs=1, batch_size=4, debug=True, patience=1)

    assert os.path.exists(best_model_path), f"Model file not found at {best_model_path}"
    print("Training loop completed successfully.")
    return best_model_path


def test_submission_generation(model_path):
    print("\n--- Testing Submission Generation ---")
    # This uses library.predict.generate_submission
    generate_submission(model_path=model_path, batch_size=4, debug=True)

    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file was not created"

    df = pd.read_csv(submission_path)
    print(f"Submission shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in expected_cols:
        assert col in df.columns, f"Submission missing column: {col}"

    # Check if confidence is clipped
    min_conf = df["Confidence"].min()
    print(f"Minimum Confidence in submission: {min_conf}")
    assert min_conf >= 70, "Confidence values were not clipped at 70"


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running tests on device: {device}")

    # 2. Data Loading Test
    batch = test_data_loading(device)

    # 3. Model Forward Test
    model, alpha, sigma_base, sigma_growth = test_model_forward(device, batch)

    # 4. Loss & Metric Test
    test_loss_and_metric(batch, (alpha, sigma_base, sigma_growth), device)

    # 5. Full Training Integration Test
    # This will train a model on a subset and save it
    model_path = test_training_loop()

    # 6. Prediction Integration Test
    # This will load the trained model and generate a submission
    test_submission_generation(model_path)

    print("\nAll verification steps passed successfully!")
