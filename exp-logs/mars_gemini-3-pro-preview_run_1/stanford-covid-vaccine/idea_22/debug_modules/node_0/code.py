import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config, set_seed
from library.utils import mcrmse_metric
from library.dataset import (
    RNADataset,
    process_dataframe,
    parse_structure_distances,
    get_dataloaders,
)
from library.model import RNAModel
from library.loss import MaskedMSELoss
from library.train import train
from library.predict import generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def create_demo_data():
    """
    Creates a small subset of the metadata to use for rapid demonstration.
    """
    print("Creating demo datasets...")

    # Define paths
    demo_dir = "./working/demo_data"
    os.makedirs(demo_dir, exist_ok=True)

    # Load a small chunk of the real metadata
    # We use the parquet files generated in the metadata directory
    full_train_path = "./metadata/train.parquet"
    full_test_path = "./metadata/test.parquet"

    if not os.path.exists(full_train_path) or not os.path.exists(full_test_path):
        raise FileNotFoundError("Metadata parquet files not found in ./metadata")

    # Read top 20 rows for training/validation
    df_train_full = pd.read_parquet(full_train_path).head(20)

    # Split into mini train and val
    df_train_mini = df_train_full.iloc[:16].copy()
    df_val_mini = df_train_full.iloc[16:].copy()

    # Read top 5 rows for testing
    df_test_mini = pd.read_parquet(full_test_path).head(5)

    # Save to demo directory
    train_path = os.path.join(demo_dir, "mini_train.parquet")
    val_path = os.path.join(demo_dir, "mini_val.parquet")
    test_path = os.path.join(demo_dir, "mini_test.parquet")

    df_train_mini.to_parquet(train_path, index=False)
    df_val_mini.to_parquet(val_path, index=False)
    df_test_mini.to_parquet(test_path, index=False)

    return train_path, val_path, test_path


def configure_demo(train_path, val_path, test_path):
    """
    Overrides the global Config class to use demo paths and lightweight hyperparameters.
    """
    print("Configuring experiment settings for demo...")

    # Update Experiment Identity
    Config.EXPERIMENT_NAME = "demo_run"

    # Update Paths
    Config.WORKING_DIR = os.path.join("./working", Config.EXPERIMENT_NAME)
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    Config.TRAIN_DATA_PATH = train_path
    Config.VAL_DATA_PATH = val_path
    Config.TEST_DATA_PATH = test_path

    # Update Hyperparameters for Speed
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size
    Config.HIDDEN_DIM = 64  # Smaller model width
    Config.NUM_LAYERS = 2  # Fewer layers
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Clean up any previous cache in this demo dir to ensure fresh processing
    for f in os.listdir(Config.WORKING_DIR):
        if f.endswith(".npz"):
            os.remove(os.path.join(Config.WORKING_DIR, f))


def verify_structure_parsing():
    """
    Verifies the logic of converting dot-bracket structure to distance matrices.
    """
    print("Verifying structure parsing logic...")

    # Test case: (..)
    # Indices:   0123
    # Pairs:     (0, 3)
    # Dist at 0: 3 - 0 = 3
    # Dist at 3: 0 - 3 = -3
    # Dist at 1, 2: 0 (unpaired)
    structure = "(..)"
    seq_len = 4
    expected = np.array([3.0, 0.0, 0.0, -3.0], dtype=np.float32)

    # Pad expected to Config.SEQ_LEN (107) because the function pads
    full_expected = np.zeros(Config.SEQ_LEN, dtype=np.float32)
    full_expected[:4] = expected

    # The function expects the full sequence length
    result = parse_structure_distances(structure, Config.SEQ_LEN)

    # Check the relevant part
    np.testing.assert_allclose(
        result[:4], expected, err_msg="Structure distance parsing failed."
    )
    print("Structure parsing verified.")


def verify_dataset_processing():
    """
    Verifies that the dataset class correctly processes raw data into tensors.
    """
    print("Verifying dataset processing...")

    # Load the mini train data
    df = pd.read_parquet(Config.TRAIN_DATA_PATH)

    # Process dataframe
    sequences, loops, distances, targets, ids = process_dataframe(df, mode="train")

    # Check shapes
    assert sequences.shape == (16, Config.SEQ_LEN), "Sequence shape mismatch"
    assert loops.shape == (16, Config.SEQ_LEN), "Loop shape mismatch"
    assert distances.shape == (16, Config.SEQ_LEN), "Distance shape mismatch"
    assert targets.shape == (16, Config.SEQ_LEN, 3), "Target shape mismatch"
    assert len(ids) == 16, "ID list length mismatch"

    # Instantiate Dataset
    dataset = RNADataset(sequences, loops, distances, targets, ids)
    item = dataset[0]

    # Check Tensor types and shapes
    assert isinstance(item["seq"], torch.Tensor)
    assert item["seq"].shape == (Config.SEQ_LEN,)
    assert item["target"].shape == (Config.SEQ_LEN, 3)

    print("Dataset processing verified.")
    return dataset


def verify_model_forward_pass(dataset):
    """
    Verifies that the model can perform a forward pass and output the correct shape.
    """
    print("Verifying model architecture...")

    device = torch.device("cpu")  # Use CPU for simple verification
    model = RNAModel(config=Config).to(device)

    # Create a small batch
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))

    seq = batch["seq"].to(device)
    loop = batch["loop"].to(device)
    dist = batch["dist"].to(device)

    # Forward pass
    output = model(seq, loop, dist)

    # Check Output Shape: (Batch, Seq_Len, Num_Targets)
    expected_shape = (2, Config.SEQ_LEN, 3)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Got {output.shape}, expected {expected_shape}"

    print("Model forward pass verified.")


def verify_loss_and_metric():
    """
    Verifies the MaskedMSELoss and MCRMSE metric calculations.
    """
    print("Verifying loss and metric...")

    # 1. Test MaskedMSELoss
    # It should only consider the first 68 positions (Config.PRED_LEN)
    criterion = MaskedMSELoss()

    # Create dummy inputs
    # Batch=1, Len=107, Targets=3
    inputs = torch.zeros((1, 107, 3), dtype=torch.float32)
    targets = torch.zeros((1, 107, 3), dtype=torch.float32)

    # Introduce error at position 0 (scored)
    inputs[0, 0, 0] = 1.0  # Error = 1.0, Squared Error = 1.0
    # Introduce error at position 100 (unscored, > 68)
    inputs[0, 100, 0] = 10.0  # Should be ignored

    # Calculate expected MSE
    # We have 68 positions * 3 targets = 204 elements considered.
    # Only 1 element has error 1.0.
    # MSE = 1.0 / 204
    expected_loss = 1.0 / (68 * 3)

    loss = criterion(inputs, targets)

    assert (
        abs(loss.item() - expected_loss) < 1e-6
    ), f"MaskedMSELoss failed. Got {loss.item()}, expected {expected_loss}"

    # 2. Test MCRMSE Metric
    # Metric: Average of RMSE per column
    y_true = np.zeros((1, 3))
    y_pred = np.zeros((1, 3))

    # Col 0: Error 1 -> RMSE 1
    y_pred[0, 0] = 1.0
    # Col 1: Error 2 -> RMSE 2
    y_pred[0, 1] = 2.0
    # Col 2: Error 0 -> RMSE 0
    y_pred[0, 2] = 0.0

    # MCRMSE = (1 + 2 + 0) / 3 = 1.0
    metric_val = mcrmse_metric(y_true, y_pred)
    assert (
        abs(metric_val - 1.0) < 1e-6
    ), f"MCRMSE Metric failed. Got {metric_val}, expected 1.0"

    print("Loss and metric verified.")


def run_training_pipeline():
    """
    Runs the library's training function.
    """
    print("\n--- Running Training Pipeline ---")

    # Ensure reproducibility
    set_seed(Config.SEED)

    # Run training
    # We set load_cached_data=False to force processing of our new mini-datasets
    train(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=False)

    # Verify artifact creation
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Training failed to generate model file at {Config.MODEL_PATH}"
        )

    print("Training pipeline completed successfully.")


def run_inference_pipeline():
    """
    Runs the library's inference function.
    """
    print("\n--- Running Inference Pipeline ---")

    # Run submission generation
    generate_submission(load_cached_data=False, batch_size=Config.BATCH_SIZE)

    # Verify artifact creation
    submission_file = "./submission/submission.csv"
    if not os.path.exists(submission_file):
        raise FileNotFoundError(
            f"Inference failed to generate submission file at {submission_file}"
        )

    # Verify content format
    df = pd.read_csv(submission_file)
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]

    # Check columns
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(
            f"Submission file missing required columns. Found: {df.columns}"
        )

    # Check row count
    # 5 test samples * 107 length = 535 rows
    expected_rows = 5 * 107
    if len(df) != expected_rows:
        raise ValueError(
            f"Submission row count mismatch. Expected {expected_rows}, got {len(df)}"
        )

    print("Inference pipeline completed successfully.")


if __name__ == "__main__":
    # 1. Setup Data
    train_p, val_p, test_p = create_demo_data()

    # 2. Configure System
    configure_demo(train_p, val_p, test_p)

    # 3. Verify Components
    verify_structure_parsing()
    dataset = verify_dataset_processing()
    verify_model_forward_pass(dataset)
    verify_loss_and_metric()

    # 4. Execute Pipelines
    run_training_pipeline()
    run_inference_pipeline()

    print("\nAll demonstrations and verifications passed!")
