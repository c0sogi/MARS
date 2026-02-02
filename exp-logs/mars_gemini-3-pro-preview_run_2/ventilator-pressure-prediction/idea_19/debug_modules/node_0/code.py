import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.utils import set_seed, WeightedL1Loss, compute_metric
from library.data_processing import prepare_data, VentilatorDataset
from library.model import CWCDP_BiLSTM
from library.training import Trainer, run_training


def setup_demo_config():
    """
    Overrides Config parameters to ensure the demo runs fast and uses minimal resources.
    """
    print(">>> Setting up Demo Configuration...")

    # Reduce training duration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16

    # Reduce Model Complexity for speed
    Config.LSTM_HIDDEN_SIZE = 32
    Config.LSTM_LAYERS = 2
    Config.GLU_HIDDEN_SIZE = 16

    # Disable multiprocessing for simple script stability
    Config.NUM_WORKERS = 0

    # Clean working directory to ensure a fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print("Configuration updated for speed.")


def demo_data_processing():
    """
    Demonstrates data loading and processing using prepare_data in debug mode.
    """
    print("\n>>> Demonstrating Data Processing...")

    # Use debug=True to load a tiny subset of data
    train_loader, val_loader, test_loader, test_ids = prepare_data(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Force processing from scratch for demo
        debug=True,
    )

    print(f"Test IDs count: {len(test_ids)}")

    # Fetch one batch to validate shapes
    X_batch, u_out_batch, y_batch = next(iter(train_loader))

    print(f"Batch X shape: {X_batch.shape}")  # Expected: (Batch, Seq, Features)
    print(f"Batch u_out shape: {u_out_batch.shape}")  # Expected: (Batch, Seq, 1)
    print(f"Batch y shape: {y_batch.shape}")  # Expected: (Batch, Seq)

    # Assertions
    expected_features = Config.INPUT_DIM
    assert X_batch.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        expected_features,
    ), f"Incorrect X shape. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN, expected_features)}, got {X_batch.shape}"
    assert y_batch.shape == (Config.BATCH_SIZE, Config.SEQ_LEN), "Incorrect y shape."

    return train_loader, val_loader, test_loader


def demo_model_architecture():
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n>>> Demonstrating Model Architecture...")

    model = CWCDP_BiLSTM()
    model.eval()

    # Create dummy inputs
    batch_size = 4
    seq_len = Config.SEQ_LEN
    input_dim = Config.INPUT_DIM

    dummy_x = torch.randn(batch_size, seq_len, input_dim)
    # u_out is passed to forward but not used in logic, just signature
    dummy_u_out = torch.zeros(batch_size, seq_len, 1)

    with torch.no_grad():
        output = model(dummy_x, dummy_u_out)

    print(f"Model Output shape: {output.shape}")

    # Assertions
    # Output should be (Batch, Seq, 1)
    assert output.shape == (
        batch_size,
        seq_len,
        1,
    ), f"Incorrect output shape. Expected {(batch_size, seq_len, 1)}, got {output.shape}"

    return model


def demo_loss_and_metric():
    """
    Demonstrates the custom WeightedL1Loss and Metric computation.
    """
    print("\n>>> Demonstrating Loss and Metric Logic...")

    criterion = WeightedL1Loss()

    # Scenario: Prediction = 10, Target = 12 -> Absolute Error = 2.0
    preds = torch.tensor([10.0, 10.0])
    targets = torch.tensor([12.0, 12.0])

    # Case 1: Inspiratory Phase (u_out = 0)
    # Config.LOSS_INSP_WEIGHT is 1.0
    u_out_insp = torch.tensor([0.0, 0.0])
    loss_insp = criterion(preds, targets, u_out_insp)

    print(f"Inspiratory Loss (Exp: 2.0): {loss_insp.item()}")
    assert abs(loss_insp.item() - 2.0) < 1e-6, "Inspiratory loss calculation incorrect."

    # Case 2: Expiratory Phase (u_out = 1)
    # Config.LOSS_EXP_WEIGHT is 0.1
    u_out_exp = torch.tensor([1.0, 1.0])
    loss_exp = criterion(preds, targets, u_out_exp)

    print(f"Expiratory Loss (Exp: 0.2): {loss_exp.item()}")
    assert abs(loss_exp.item() - 0.2) < 1e-6, "Expiratory loss calculation incorrect."

    # Case 3: Metric Computation (Only counts Inspiration)
    # 1 Insp (Error 2.0), 1 Exp (Error 2.0)
    # Metric should ignore Exp, so result should be 2.0
    u_out_mixed = torch.tensor([0.0, 1.0])
    mae = compute_metric(preds, targets, u_out_mixed)

    print(f"Metric MAE (Exp: 2.0): {mae}")
    assert abs(mae - 2.0) < 1e-6, "Metric calculation incorrect."


def demo_training_execution(model, train_loader, val_loader):
    """
    Demonstrates the Trainer class and fitting process.
    """
    print("\n>>> Demonstrating Training Loop...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = model.to(device)
    trainer = Trainer(model, train_loader, val_loader, device)

    # Run fit (Config.EPOCHS is set to 1)
    trainer.fit(epochs=Config.EPOCHS)

    # Verify Checkpoint
    assert os.path.exists(Config.BEST_MODEL_PATH), "Model checkpoint was not saved."
    print(f"Checkpoint verified at: {Config.BEST_MODEL_PATH}")


def demo_full_pipeline():
    """
    Demonstrates the high-level run_training function which orchestrates everything.
    """
    print("\n>>> Demonstrating Full Pipeline (run_training)...")

    # Execute the full pipeline function provided in library.training
    # This will re-prepare data, train a new model, and generate submission
    run_training(epochs=1, debug=True)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")

    assert len(df_sub) > 0, "Submission file is empty."
    assert list(df_sub.columns) == [
        "id",
        "pressure",
    ], "Submission columns are incorrect."


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # 1. Setup Config
    setup_demo_config()

    # 2. Process Data
    train_loader, val_loader, test_loader = demo_data_processing()

    # 3. Initialize Model
    model = demo_model_architecture()

    # 4. Verify Loss/Metric
    demo_loss_and_metric()

    # 5. Run Training Loop
    demo_training_execution(model, train_loader, val_loader)

    # 6. Run Full Pipeline (End-to-End)
    demo_full_pipeline()

    print("\n>>> All demonstrations completed successfully.")
