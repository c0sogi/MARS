import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import get_dataloaders
from library.model import DualAxisFiLMNet
from library.train import Trainer, LaplaceLoss, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("Initializing Demonstration...")

    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    print("\n[1] Overriding Configuration for Demo Speed...")
    # Reduce dataset size and training duration for the demo
    Config.DEBUG = True
    Config.DEBUG_SIZE = 16  # Use a tiny subset of data
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Redirect outputs to a demo specific directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = "./working/demo_submission.csv"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n[2] Verifying Data Pipeline...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))

    axial = batch["axial"]
    coronal = batch["coronal"]
    tabular = batch["tabular"]
    target = batch["target"]

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Axial Shape: {axial.shape} (Expected: [{Config.BATCH_SIZE}, 3, 224, 224])")
    print(
        f"Coronal Shape: {coronal.shape} (Expected: [{Config.BATCH_SIZE}, 3, 224, 224])"
    )
    print(
        f"Tabular Shape: {tabular.shape} (Expected: [{Config.BATCH_SIZE}, {Config.TABULAR_INPUT_DIM}])"
    )

    # Assertions
    assert axial.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Incorrect Axial Image Shape"
    assert coronal.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Incorrect Coronal Image Shape"
    assert tabular.shape == (
        Config.BATCH_SIZE,
        Config.TABULAR_INPUT_DIM,
    ), "Incorrect Tabular Shape"
    assert target.shape == (Config.BATCH_SIZE,), "Incorrect Target Shape"

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n[3] Verifying Model Architecture...")
    model = DualAxisFiLMNet().to(device)

    # Move batch to device
    axial = axial.to(device)
    coronal = coronal.to(device)
    tabular = tabular.to(device)

    # Forward pass
    alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

    print(f"Alpha Shape: {alpha.shape}")
    print(f"Sigma Base Shape: {sigma_base.shape}")

    # Assertions
    assert alpha.shape == (Config.BATCH_SIZE,), "Output Alpha shape mismatch"
    assert sigma_base.shape == (Config.BATCH_SIZE,), "Output Sigma Base shape mismatch"

    # Check positivity of sigmas (Softplus constraint)
    assert torch.all(sigma_base > 0), "Sigma Base must be positive"
    assert torch.all(sigma_growth > 0), "Sigma Growth must be positive"

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print("\n[4] Verifying Loss Calculation...")
    criterion = LaplaceLoss()

    time_delta = batch["time_delta"].to(device)
    baseline_fvc = batch["baseline_fvc"].to(device)
    target = target.to(device)

    loss = criterion(alpha, sigma_base, sigma_growth, time_delta, baseline_fvc, target)

    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Infinite"

    # ==========================================
    # 5. Training Loop Execution
    # ==========================================
    print("\n[5] Executing Training Loop (Short Run)...")
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit()

    # Verify model checkpoint was saved
    assert os.path.exists(trainer.best_model_path), "Best model checkpoint not found"

    # ==========================================
    # 6. Inference & Submission
    # ==========================================
    print("\n[6] Generating Submission...")
    # Load best model state
    model.load_state_dict(torch.load(trainer.best_model_path, map_location=device))

    generate_submission(model, test_loader, device)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {sub_df.shape}")
    print("Submission Head:")
    print(sub_df.head())

    assert "Patient_Week" in sub_df.columns
    assert "FVC" in sub_df.columns
    assert "Confidence" in sub_df.columns

    # ==========================================
    # 7. Metric Logic Check
    # ==========================================
    print("\n[7] Verifying Metric Logic...")
    # Test Case 1: Perfect prediction
    # FVC_true = 2000, FVC_pred = 2000, Sigma = 70 (clipped min)
    # Delta = 0
    # Metric = - (sqrt(2)*0)/70 - ln(sqrt(2)*70) = -ln(98.99) approx -4.595
    y_true = np.array([2000])
    y_pred = np.array([2000])
    sigma = np.array([50])  # Should be clipped to 70

    score = laplace_log_likelihood_metric(y_true, y_pred, sigma)
    expected_score = -np.log(np.sqrt(2) * 70)

    print(f"Perfect Score (clipped sigma): {score:.4f}")
    assert np.isclose(
        score, expected_score, atol=1e-4
    ), "Metric calculation mismatch on perfect case"

    # Test Case 2: Large Error
    # Error > 1000, should be clipped to 1000
    y_true_bad = np.array([2000])
    y_pred_bad = np.array([4000])  # Error 2000 -> clipped to 1000
    sigma_bad = np.array([100])

    score_bad = laplace_log_likelihood_metric(y_true_bad, y_pred_bad, sigma_bad)

    # Manual calc
    delta = 1000
    sigma_val = 100
    expected_bad = -(np.sqrt(2) * delta) / sigma_val - np.log(np.sqrt(2) * sigma_val)

    print(f"Bad Score (clipped error): {score_bad:.4f}")
    assert np.isclose(
        score_bad, expected_bad, atol=1e-4
    ), "Metric calculation mismatch on large error case"

    print("\n" + "=" * 40)
    print("DEMONSTRATION COMPLETE: All checks passed.")
    print("=" * 40)


if __name__ == "__main__":
    main()
