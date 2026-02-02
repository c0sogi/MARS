import os
import pandas as pd
import torch
import numpy as np
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.data import get_dataloaders
from library.model import ResidualCrossAttentionNet
from library.train import LaplaceLoss, train_one_epoch, validate
from library.predict import generate_predictions


def create_subset_metadata():
    """
    Creates small subset CSVs in the working directory to ensure the demo runs quickly.
    It takes the top N rows from the pre-generated metadata files.
    """
    print("Creating subset metadata for demonstration...")

    # Load original metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Take small subsets (enough to form at least one or two batches)
    train_subset = train_df.head(8).copy()
    val_subset = val_df.head(4).copy()
    test_subset = test_df.head(4).copy()

    # Save to working directory
    subset_train_path = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    subset_val_path = os.path.join(Config.WORKING_DIR, "val_subset.csv")
    subset_test_path = os.path.join(Config.WORKING_DIR, "test_subset.csv")

    train_subset.to_csv(subset_train_path, index=False)
    val_subset.to_csv(subset_val_path, index=False)
    test_subset.to_csv(subset_test_path, index=False)

    return subset_train_path, subset_val_path, subset_test_path


def demo_metric_logic():
    """
    Verifies the Laplace Log Likelihood metric logic against manual calculations.
    """
    print("\n--- Verifying Metric Logic ---")

    # Case 1: Perfect prediction
    # Delta = 0, Sigma = 100 (clipped to 100, which is > 70)
    # Metric = - (sqrt(2)*0)/100 - ln(sqrt(2)*100) = -ln(141.42) approx -4.95
    y_true = np.array([2000])
    y_pred = np.array([2000])
    sigma = np.array([100])

    score = laplace_log_likelihood(y_true, y_pred, sigma)
    expected_score = -np.log(np.sqrt(2) * 100)

    assert np.isclose(
        score, expected_score, atol=1e-4
    ), f"Metric mismatch for perfect prediction. Got {score}, expected {expected_score}"
    print("Metric verification (Perfect Prediction): Passed")

    # Case 2: Clipping Sigma
    # Sigma = 10 (should be clipped to 70)
    y_true = np.array([2000])
    y_pred = np.array([2000])
    sigma = np.array([10])

    score = laplace_log_likelihood(y_true, y_pred, sigma)
    expected_score = -np.log(np.sqrt(2) * 70)

    assert np.isclose(
        score, expected_score, atol=1e-4
    ), f"Metric mismatch for sigma clipping. Got {score}, expected {expected_score}"
    print("Metric verification (Sigma Clipping): Passed")


def demo_pipeline():
    # 1. Configuration Override for Demo
    print("--- Configuring Demo ---")
    Config.setup_directories()
    seed_everything(Config.SEED)

    # Override paths to use subsets
    train_csv, val_csv, test_csv = create_subset_metadata()
    Config.TRAIN_CSV = train_csv
    Config.VAL_CSV = val_csv
    Config.TEST_CSV = test_csv

    # Override Hyperparameters for speed
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.EXPERIMENT_NAME = "demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, Config.EXPERIMENT_NAME, "cache")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("\n--- Loading Data ---")
    # get_dataloaders reads from the Config paths we just modified
    train_loader, val_loader, test_loader = get_dataloaders(Config)

    # Verify Loader
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty. Check subset creation.")

    print(f"Batch keys: {list(batch.keys())}")

    # Check shapes
    # img_ax: (B, 3, 224, 224)
    assert batch["img_ax"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect Axial Image Shape"
    assert batch["tabular"].shape == (
        Config.BATCH_SIZE,
        Config.TABULAR_INPUT_DIM,
    ), "Incorrect Tabular Shape"
    assert "fvc_target" in batch, "Target missing in training batch"

    print("Data Loading Verification: Passed")

    # 3. Model Initialization & Forward Pass
    print("\n--- Initializing Model ---")
    model = ResidualCrossAttentionNet().to(device)

    # Move batch to device
    img_ax = batch["img_ax"].to(device)
    img_cor = batch["img_cor"].to(device)
    tabular = batch["tabular"].to(device)
    relative_week = batch["relative_week"].to(device)
    baseline_fvc = batch["baseline_fvc"].to(device)

    print("Running Forward Pass...")
    fvc_pred, sigma_pred = model(tabular, img_ax, img_cor, relative_week, baseline_fvc)

    # Verify Output Shapes
    assert fvc_pred.shape == (
        Config.BATCH_SIZE,
    ), f"Pred shape mismatch: {fvc_pred.shape}"
    assert sigma_pred.shape == (
        Config.BATCH_SIZE,
    ), f"Sigma shape mismatch: {sigma_pred.shape}"
    # Verify Sigma Positivity (Softplus is used in the model)
    assert (sigma_pred > 0).all(), "Sigma predictions must be positive"

    print("Model Forward Pass: Passed")

    # 4. Loss Calculation
    print("\n--- Calculating Loss ---")
    criterion = LaplaceLoss()
    fvc_target = batch["fvc_target"].to(device)

    loss = criterion(fvc_pred, sigma_pred, fvc_target)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.dim() == 0, "Loss should be a scalar"

    print("Loss Calculation: Passed")

    # 5. Training Step
    print("\n--- Running Training Step ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch (on the subset)
    epoch_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch Loss: {epoch_loss:.4f}")

    # Loss = -Metric. Metric is typically negative (-6 to -8). So Loss should be positive (6 to 8).
    # We just check it's a valid number.
    assert (
        epoch_loss > 0
    ), "Loss expected to be positive (since it is negative log likelihood)"

    print("Training Step: Passed")

    # 6. Validation Step
    print("\n--- Running Validation Step ---")
    val_score = validate(model, val_loader, device)
    print(f"Validation Score: {val_score:.4f}")
    # Score should be negative (Metric values)
    assert val_score < 10, "Score should be reasonable (likely negative)"

    print("Validation Step: Passed")

    # 7. Saving Model
    print("\n--- Saving Model ---")
    weights_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    torch.save(model.state_dict(), weights_path)
    assert os.path.exists(weights_path), "Model weights file not created"

    print("Model Saving: Passed")

    # 8. Inference
    print("\n--- Running Inference ---")
    # We use the generate_predictions function from library.predict
    # It expects weights at a path.
    # It will use Config.TEST_CSV which we overrode.

    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df = generate_predictions(weights_path=weights_path, batch_size=2)

    print("Submission Head:")
    print(submission_df.head())

    # Verify Submission
    assert "Patient_Week" in submission_df.columns
    assert "FVC" in submission_df.columns
    assert "Confidence" in submission_df.columns
    assert len(submission_df) == len(pd.read_csv(Config.TEST_CSV))

    print("Inference: Passed")

    print("\nAll Demonstrations Completed Successfully.")


if __name__ == "__main__":
    # Suppress specific warnings for clean output
    warnings.filterwarnings("ignore", category=UserWarning)

    # Run Metric Logic Check
    demo_metric_logic()

    # Run Main Pipeline Demo
    demo_pipeline()
