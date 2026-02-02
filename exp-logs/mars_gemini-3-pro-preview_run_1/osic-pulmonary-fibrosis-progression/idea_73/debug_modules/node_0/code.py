import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_metric, LaplaceLogLikelihoodLoss
from library.data import get_dataloaders, LungDataset
from library.model import PCCGNet
from library.train import run_training
from library.inference import run_inference


def setup_demo_environment():
    """
    Sets up a temporary environment and subsets data to ensure the demo
    runs quickly within the time limits.
    """
    print("--- Setting up Demo Environment ---")

    # 1. Define paths for demo outputs
    demo_id = "demo_run"
    base_dir = os.path.join("./working", demo_id)
    meta_dir = os.path.join(base_dir, "metadata")

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # 2. Subset Metadata (Take top 4 rows for Train/Val/Test)
    # This minimizes DICOM I/O and training time
    subset_size = 4

    try:
        # Train
        df_train = pd.read_csv(Config.TRAIN_CSV)
        df_train.head(subset_size).to_csv(
            os.path.join(meta_dir, "train.csv"), index=False
        )

        # Val
        df_val = pd.read_csv(Config.VAL_CSV)
        df_val.head(subset_size).to_csv(os.path.join(meta_dir, "val.csv"), index=False)

        # Test
        df_test = pd.read_csv(Config.TEST_CSV)
        df_test.head(subset_size).to_csv(
            os.path.join(meta_dir, "test.csv"), index=False
        )

        print(f"Metadata subsetted to {subset_size} samples each.")

    except FileNotFoundError as e:
        print(f"Error reading original metadata: {e}")
        sys.exit(1)

    # 3. Override Config globally
    Config.IDEA_ID = demo_id
    Config.WORKING_DIR = base_dir
    Config.CACHE_DIR = os.path.join(base_dir, "cache")
    Config.CHECKPOINT_DIR = os.path.join(base_dir, "checkpoints")
    Config.SUBMISSION_DIR = base_dir
    Config.SUBMISSION_FILE = os.path.join(base_dir, "submission.csv")

    # Point to new metadata
    Config.TRAIN_CSV = os.path.join(meta_dir, "train.csv")
    Config.VAL_CSV = os.path.join(meta_dir, "val.csv")
    Config.TEST_CSV = os.path.join(meta_dir, "test.csv")

    # Create necessary directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Optimize Hyperparameters for Speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script
    Config.IMG_SIZE = 224  # Keep standard resolution

    print("Configuration updated for rapid execution.")


def verify_logic():
    """
    Verifies the correctness of the metric, loss function, and model architecture.
    """
    print("\n--- Verifying Logic & Components ---")

    # 1. Verify Metric Calculation
    # Scenario: Pred=2000, True=2000, Conf=100. Delta=0. Sigma_clipped=100.
    # Metric = - (sqrt(2)*0/100) - ln(sqrt(2)*100) = -ln(141.42) ~= -4.95
    preds = np.array([[2000, 100]])
    targets = np.array([2000])
    score = calculate_metric(preds, targets)
    print(f"Metric Check (Perfect Match): {score:.4f}")
    assert score < 0, "Metric should be negative."
    assert np.isfinite(score), "Metric should be finite."

    # 2. Verify Loss Function
    criterion = LaplaceLogLikelihoodLoss()
    t_preds = torch.tensor(
        [[2000.0, 50.0]], requires_grad=True
    )  # Sigma 50 will be clipped to 70
    t_targets = torch.tensor([2100.0])  # Delta 100
    loss = criterion(t_preds, t_targets)
    loss.backward()

    print(f"Loss Check: {loss.item():.4f}")
    assert t_preds.grad is not None, "Gradients should flow back to predictions."

    # 3. Verify Data Loading & Shapes
    print("Initializing DataLoaders...")
    train_loader, _, _ = get_dataloaders()
    batch = next(iter(train_loader))
    inputs, targets = batch

    # Check keys
    required_keys = ["axial", "coronal", "tabular", "delta_week", "base_fvc"]
    for k in required_keys:
        assert k in inputs, f"Missing key {k} in batch."

    # Check dimensions
    # Axial: [B, 3, H, W]
    assert inputs["axial"].shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    # Tabular: [B, 7]
    assert inputs["tabular"].shape == (Config.BATCH_SIZE, 7)

    print("Data loading verified.")

    # 4. Verify Model Forward Pass
    print("Initializing Model...")
    model = PCCGNet()
    model.eval()

    with torch.no_grad():
        outputs = model(
            inputs["axial"],
            inputs["coronal"],
            inputs["tabular"],
            inputs["delta_week"],
            inputs["base_fvc"],
        )

    # Output should be [B, 2] -> FVC, Confidence
    assert outputs.shape == (
        Config.BATCH_SIZE,
        2,
    ), f"Incorrect output shape: {outputs.shape}"
    assert torch.all(torch.isfinite(outputs)), "Model output contains NaNs or Infs."

    print("Model forward pass verified.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    # 1. Prepare Environment
    setup_demo_environment()

    # 2. Verify Components
    verify_logic()

    # 3. Run Training Pipeline
    # This uses the subset data and runs for 1 epoch
    print("\n--- Starting Training Pipeline ---")
    run_training()

    # 4. Run Inference Pipeline
    # This generates predictions on the subset test data
    print("\n--- Starting Inference Pipeline ---")
    run_inference()

    # 5. Validate Submission
    if os.path.exists(Config.SUBMISSION_FILE):
        sub_df = pd.read_csv(Config.SUBMISSION_FILE)
        print("\n--- Submission Generated ---")
        print(sub_df)

        # Basic checks
        assert len(sub_df) == 4, "Submission should have 4 rows (matching subset)."
        assert "FVC" in sub_df.columns and "Confidence" in sub_df.columns
        assert not sub_df.isnull().values.any(), "Submission contains null values."
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemo completed successfully.")
