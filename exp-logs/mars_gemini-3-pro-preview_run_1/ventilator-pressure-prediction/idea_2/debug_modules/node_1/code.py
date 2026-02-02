import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data_processing import DataProcessor
from library.dataset import VentilatorDataset
from library.model import HybridCNNLSTM
from library.training import run_training
from library.inference import generate_predictions


def main():
    print("=== Ventilator Pressure Prediction: Library Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config for speed and resource efficiency
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Small sample for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 2

    # Simplify model for faster forward/backward passes during demo
    Config.LSTM_LAYERS = 2
    Config.LSTM_HIDDEN_SIZE = 64
    Config.CNN_FILTERS = 32
    Config.FC_HIDDEN_SIZE = 32

    # Set up working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean/Create working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)
    print("    Configuration updated. Debug mode enabled.")

    # ---------------------------------------------------------
    # 2. Data Processing Demonstration
    # ---------------------------------------------------------
    print("\n[2] Demonstrating DataProcessor...")
    processor = DataProcessor()

    # Load train data
    print("    Loading training data (subset)...")
    X_train, y_train, u_out_train = processor.load_dataset(
        split="train", load_cached_data=False
    )

    # Assertions to verify data structure
    assert isinstance(X_train, np.ndarray), "X_train should be a numpy array"
    assert isinstance(y_train, np.ndarray), "y_train should be a numpy array"
    assert isinstance(u_out_train, np.ndarray), "u_out_train should be a numpy array"

    # Check shapes: (N_breaths, Seq_Len, Features)
    n_breaths, seq_len, n_features = X_train.shape
    print(
        f"    Data Shapes -> X: {X_train.shape}, y: {y_train.shape}, u_out: {u_out_train.shape}"
    )

    assert (
        seq_len == Config.SEQ_LEN
    ), f"Sequence length mismatch. Expected {Config.SEQ_LEN}, got {seq_len}"
    assert (
        n_features == Config.INPUT_DIM
    ), f"Feature dim mismatch. Expected {Config.INPUT_DIM}, got {n_features}"
    assert y_train.shape == (n_breaths, seq_len), "Target shape mismatch"
    assert u_out_train.shape == (n_breaths, seq_len), "Mask shape mismatch"

    print("    DataProcessor verification passed.")

    # ---------------------------------------------------------
    # 3. Dataset & DataLoader Demonstration
    # ---------------------------------------------------------
    print("\n[3] Demonstrating VentilatorDataset & DataLoader...")
    dataset = VentilatorDataset(X_train, u_out_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    # Fetch a single batch
    x_batch, y_batch, u_out_batch = next(iter(loader))

    print(f"    Batch Shapes -> X: {x_batch.shape}, y: {y_batch.shape}")
    assert x_batch.shape == (4, Config.SEQ_LEN, Config.INPUT_DIM)
    assert y_batch.shape == (4, Config.SEQ_LEN)
    assert x_batch.dtype == torch.float32
    print("    Dataset verification passed.")

    # ---------------------------------------------------------
    # 4. Model Architecture Demonstration
    # ---------------------------------------------------------
    print("\n[4] Demonstrating HybridCNNLSTM Model...")
    device = torch.device(Config.DEVICE)
    model = HybridCNNLSTM().to(device)

    # Forward pass check
    x_input = x_batch.to(device)
    output = model(x_input)

    print(f"    Model Output Shape: {output.shape}")
    assert output.shape == (4, Config.SEQ_LEN), "Model output shape mismatch"

    # Check if parameters require grad
    assert any(
        p.requires_grad for p in model.parameters()
    ), "Model parameters should require gradients"
    print("    Model verification passed.")

    # ---------------------------------------------------------
    # 5. Full Training Pipeline
    # ---------------------------------------------------------
    print("\n[5] Running Full Training Pipeline (run_training)...")
    # run_training handles loading, training loop, validation, and saving model
    run_training()

    # Verify model file was created
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved after training"
    print(f"    Training complete. Model saved to {Config.MODEL_PATH}")

    # ---------------------------------------------------------
    # 6. Inference Demonstration
    # ---------------------------------------------------------
    print("\n[6] Demonstrating Inference (generate_predictions)...")

    # Define a separate output path for this specific inference call
    inference_output = os.path.join(Config.WORKING_DIR, "inference_submission.csv")

    generate_predictions(
        model_path=Config.MODEL_PATH,
        output_path=inference_output,
        sample_submission_path=Config.SAMPLE_SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE,
        device=Config.DEVICE,
        load_cached_data=True,  # Use cached test data generated during run_training if available
    )

    assert os.path.exists(inference_output), "Inference output file not found"
    print(f"    Inference complete. Submission saved to {inference_output}")

    # ---------------------------------------------------------
    # 7. Final Output Validation
    # ---------------------------------------------------------
    print("\n[7] Validating Submission File...")
    sub_df = pd.read_csv(inference_output)

    # Basic checks
    print(f"    Submission Shape: {sub_df.shape}")
    print(f"    Columns: {sub_df.columns.tolist()}")

    assert (
        "id" in sub_df.columns and "pressure" in sub_df.columns
    ), "Missing required columns"
    assert not sub_df["pressure"].isnull().any(), "Submission contains NaN values"

    # Check against sample submission length
    sample_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    if Config.DEBUG:
        print(f"    Debug mode: Skipping strict length check (Got {len(sub_df)} rows).")
    else:
        assert len(sub_df) == len(
            sample_df
        ), f"Length mismatch. Expected {len(sample_df)}, got {len(sub_df)}"

    print("    Submission file is valid.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
