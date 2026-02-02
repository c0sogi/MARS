import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import prepare_data
from library.model import FCPNet
from library.train import masked_mae_loss, train_epoch, validate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_config():
    """
    Overrides the default configuration for a quick demonstration run.
    Uses a separate working directory and reduced model complexity.
    """
    print("1. Setting up demo configuration...")

    # Define a specific directory for this demo to avoid conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_CACHE = os.path.join(demo_dir, "train.parquet")
    Config.VAL_CACHE = os.path.join(demo_dir, "val.parquet")
    Config.TEST_CACHE = os.path.join(demo_dir, "test.parquet")
    Config.MODEL_PATH = os.path.join(demo_dir, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Override Hyperparameters for speed
    Config.BATCH_SIZE = 16
    Config.EPOCHS = 1
    Config.HIDDEN_DIM = 32  # Reduced from 64
    Config.LSTM_LAYERS = 1  # Reduced from 3
    Config.SEQ_LEN = 80  # Fixed by dataset

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print(f"   Working directory set to: {Config.WORKING_DIR}")
    print(f"   Batch size: {Config.BATCH_SIZE}, Hidden Dim: {Config.HIDDEN_DIM}")


def verify_data_pipeline():
    """
    Demonstrates data loading, feature engineering, and loader creation.
    Verifies tensor shapes.
    """
    print("\n2. Verifying Data Pipeline...")

    # prepare_data with debug=True loads a tiny subset (100 train, 50 val, 50 test breaths)
    # load_cached_data=False forces regeneration of cache for this new dir
    train_loader, val_loader, test_loader = prepare_data(
        debug=True, load_cached_data=False
    )

    # Fetch one batch from training loader
    inputs, targets = next(iter(train_loader))

    # Verify shapes
    # Inputs: (Batch, Seq_Len, Input_Dim)
    # Targets: (Batch, Seq_Len)
    print(f"   Input batch shape: {inputs.shape}")
    print(f"   Target batch shape: {targets.shape}")

    expected_seq_len = Config.SEQ_LEN
    assert (
        inputs.shape[1] == expected_seq_len
    ), f"Expected sequence length {expected_seq_len}, got {inputs.shape[1]}"
    assert (
        inputs.shape[2] == Config.INPUT_DIM
    ), f"Expected input dim {Config.INPUT_DIM}, got {inputs.shape[2]}"
    assert (
        targets.shape[1] == expected_seq_len
    ), f"Expected target sequence length {expected_seq_len}, got {targets.shape[1]}"

    print("   Data Pipeline verification passed.")
    return train_loader, val_loader, test_loader


def verify_model_architecture(device):
    """
    Instantiates the model and runs a forward pass check.
    """
    print("\n3. Verifying Model Architecture...")

    model = FCPNet(config=Config).to(device)

    # Create a dummy input tensor: (Batch, Seq, Features)
    dummy_input = torch.randn(Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_DIM).to(
        device
    )

    # Forward pass
    output = model(dummy_input)

    # Verify output shape: (Batch, Seq, 1)
    print(f"   Model output shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        1,
    ), "Model output shape mismatch"

    print("   Model Architecture verification passed.")
    return model


def verify_loss_function(device):
    """
    Unit test for masked_mae_loss.
    Ensures that errors during the expiratory phase (u_out=1) are ignored.
    """
    print("\n4. Verifying Loss Function Logic...")

    # Scenario:
    # 2 time steps.
    # Step 0: u_out=0 (Inspiratory), Prediction=10, True=12 -> Error=2
    # Step 1: u_out=1 (Expiratory), Prediction=100, True=10 -> Error=90 (Should be ignored)

    y_pred = torch.tensor([[10.0], [100.0]]).to(
        device
    )  # Shape (2, 1) - mimicking flattened batch or seq
    y_true = torch.tensor([12.0, 10.0]).to(device)  # Shape (2,)

    # u_out is typically part of the input features, but the loss function takes it as a separate tensor
    # Shape (2,)
    u_out = torch.tensor([0.0, 1.0]).to(device)

    # Calculate loss
    loss = masked_mae_loss(y_pred, y_true, u_out)

    print(f"   Calculated Loss: {loss.item()}")

    # Expected: Mean of valid errors. Valid error is |10 - 12| = 2.
    # The error |100 - 10| = 90 should be masked out.
    # Mean = 2 / 1 = 2.0
    assert (
        abs(loss.item() - 2.0) < 1e-6
    ), f"Loss function failed. Expected 2.0, got {loss.item()}"

    print("   Loss Function verification passed.")


def demonstrate_training_loop(model, train_loader, val_loader, device):
    """
    Runs a minimal training loop (1 epoch) and validation.
    """
    print("\n5. Demonstrating Training Loop...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run 1 Epoch
    print(f"   Running training for {Config.EPOCHS} epoch(s)...")
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"   Train Loss: {train_loss:.4f}")

    # Run Validation
    print("   Running validation...")
    val_loss = validate(model, val_loader, device)
    print(f"   Validation Loss: {val_loss:.4f}")

    # Assert losses are valid numbers
    assert np.isfinite(train_loss), "Training loss is NaN or Infinite"
    assert np.isfinite(val_loss), "Validation loss is NaN or Infinite"

    # Save the model (simulating the checkpointing in train.py)
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"   Model saved to {Config.MODEL_PATH}")


def demonstrate_inference(model, test_loader, device):
    """
    Generates predictions and creates a submission file.
    """
    print("\n6. Demonstrating Inference and Submission...")

    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs)
            # Flatten predictions
            preds = preds.squeeze(-1).flatten().cpu().numpy()
            predictions.extend(preds)

    predictions = np.array(predictions)
    print(f"   Generated {len(predictions)} predictions.")

    # Load test metadata (subset used in debug mode)
    # Note: In a real run, we read the full file. Here, prepare_data(debug=True)
    # filtered the data. We need to manually replicate that filtering to match IDs
    # or just verify the prediction count matches the loader's data count.

    # For this demo, we will verify that the number of predictions matches
    # the number of rows in the cached test dataframe.
    test_df = pd.read_parquet(Config.TEST_CACHE)

    print(f"   Test metadata rows: {len(test_df)}")

    assert len(predictions) == len(
        test_df
    ), f"Mismatch: Predictions ({len(predictions)}) vs Metadata ({len(test_df)})"

    # Create submission dataframe
    # Sorting is handled in prepare_data, so order should align
    test_df = test_df.sort_values(["breath_id", "time_step"])
    submission = pd.DataFrame({"id": test_df["id"], "pressure": predictions})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"   Submission saved to {Config.SUBMISSION_PATH}")

    # Verify file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Peek at submission
    print("   Submission head:")
    print(submission.head(3).to_string(index=False))


if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()
    device = get_device()
    print(f"   Device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader = verify_data_pipeline()

    # 3. Model
    model = verify_model_architecture(device)

    # 4. Loss Logic
    verify_loss_function(device)

    # 5. Training
    demonstrate_training_loop(model, train_loader, val_loader, device)

    # 6. Inference
    demonstrate_inference(model, test_loader, device)

    print("\n=== Demo Completed Successfully ===")
