import sys
import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_log_mae
from library.data import get_dataloaders
from library.model import DistanceWeightedGCN
from library.engine import Engine


def verify_metric_logic():
    """
    Verifies the implementation of the Log Mean Absolute Error metric.
    """
    print("1. Verifying Metric Logic...")

    # Case 1: Perfect prediction (Error should be extremely low due to log(epsilon))
    preds_perfect = np.array([10.0, 20.0])
    targets_perfect = np.array([10.0, 20.0])
    types_perfect = np.array([0, 1])  # Two different types

    score_perfect = calculate_log_mae(preds_perfect, targets_perfect, types_perfect)
    # log(1e-9) is approx -20.7
    assert score_perfect < -10.0, f"Perfect prediction score too high: {score_perfect}"

    # Case 2: Known error
    # Type 0: Error = 1.0 -> log(1.0) = 0.0
    # Type 1: Error = e   -> log(e)   = 1.0
    # Average LogMAE = (0.0 + 1.0) / 2 = 0.5
    preds_known = np.array([11.0, 20.0 + np.e])
    targets_known = np.array([10.0, 20.0])
    types_known = np.array([0, 1])

    score_known = calculate_log_mae(preds_known, targets_known, types_known)
    expected_score = 0.5

    # Allow small tolerance for floating point arithmetic and epsilon
    assert np.isclose(
        score_known, expected_score, atol=1e-4
    ), f"Metric mismatch. Expected ~{expected_score}, got {score_known}"

    print("   -> Metric logic verified successfully.")


def configure_demo_environment():
    """
    Overrides Config attributes to ensure the script runs quickly for demonstration.
    """
    print("\n2. Configuring Environment for Demo...")

    # Use a separate directory for demo artifacts
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config
    Config.IDEA_WORK_DIR = demo_dir
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 molecules
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size
    Config.EARLY_STOPPING_PATIENCE = 2

    # Update paths to point to the demo directory
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "demo_model.pt")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")

    print(f"   -> Working Directory: {Config.IDEA_WORK_DIR}")
    print(f"   -> Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"   -> Epochs: {Config.NUM_EPOCHS}")


def verify_data_pipeline():
    """
    Verifies data loading, graph construction, and batching.
    """
    print("\n3. Verifying Data Pipeline...")

    # Force processing (load_cached=False) to ensure we use the small sample size
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch a single batch
    batch = next(iter(train_loader))

    print(f"   -> Batch retrieved. Num Graphs: {batch.num_graphs}")

    # Assertions on Data Structure
    # x: [Num_Nodes] (Atom indices)
    assert batch.x.dim() == 1, "Node features 'x' should be 1D tensor of indices."

    # edge_index: [2, Num_Edges]
    assert (
        batch.edge_index.dim() == 2 and batch.edge_index.shape[0] == 2
    ), "Edge index should be shape [2, Num_Edges]."

    # y: [Num_Couples] (Targets)
    assert batch.y.dim() == 1, "Target 'y' should be 1D tensor."

    # couple_index: [2, Num_Couples]
    assert (
        batch.couple_index.shape[0] == 2
    ), "Couple index should be shape [2, Num_Couples]."

    print("   -> Data shapes and structures verified.")
    return batch


def verify_model_forward(batch):
    """
    Verifies the model initialization and forward pass.
    """
    print("\n4. Verifying Model Forward Pass...")

    device = Config.DEVICE
    model = DistanceWeightedGCN().to(device)
    batch = batch.to(device)

    model.eval()
    with torch.no_grad():
        output = model(batch)

    print(f"   -> Input Targets Shape: {batch.y.shape}")
    print(f"   -> Model Output Shape: {output.shape}")

    # Output should be [Num_Couples, 1]
    expected_shape = (batch.y.shape[0], 1)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print("   -> Model forward pass successful.")


def run_full_engine():
    """
    Runs the Engine to demonstrate training, validation, and submission generation.
    """
    print("\n5. Running Full Engine (Train -> Val -> Predict)...")

    engine = Engine()

    # Execute the training loop
    # This will use the cached data generated in verify_data_pipeline
    engine.run()

    # Verify artifacts
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError("Model file was not saved.")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    # Check submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   -> Submission generated. Shape: {df_sub.shape}")

    assert "id" in df_sub.columns, "Submission missing 'id' column."
    assert (
        "scalar_coupling_constant" in df_sub.columns
    ), "Submission missing target column."
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."

    print("   -> Engine execution completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(Config.SEED)

    # 1. Verify Metric
    verify_metric_logic()

    # 2. Configure for Demo
    configure_demo_environment()

    # 3. Verify Data Pipeline
    sample_batch = verify_data_pipeline()

    # 4. Verify Model
    verify_model_forward(sample_batch)

    # 5. Run Engine
    run_full_engine()

    print("\nAll demonstrations passed successfully.")
