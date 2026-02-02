import torch
import numpy as np
import pandas as pd
import os
import random
import torch.optim as optim
from library.config import Config
from library.data_utils import get_dataloaders
from library.model import StructuralBiGRU
from library.loss import MCRMSELoss, mcrmse_metric
from library.train_eval import train_epoch, validate, predict_and_submit


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Initializing demonstration...")

    # 1. Configuration Override for Speed
    # We modify the Config class attributes directly to run a fast demo.
    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples for training/val
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading Verification
    print("Loading data (debug mode)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    # Fetch one batch to verify shapes
    inputs, pair_indices, targets = next(iter(train_loader))

    # Assertions for data shapes
    # Inputs: (Batch, Seq_Len=107, Channels=14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_DIM,
    ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, 107, 14)}, got {inputs.shape}"

    # Pair Indices: (Batch, Seq_Len=107)
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Pair indices shape mismatch. Expected {(Config.BATCH_SIZE, 107)}, got {pair_indices.shape}"

    # Targets: (Batch, Seq_Scored=68, Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_SCORED,
        Config.NUM_TARGETS,
    ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, 68, 5)}, got {targets.shape}"

    print("Data shapes verified successfully.")

    # 3. Model Initialization and Forward Pass
    print("Initializing model...")
    model = StructuralBiGRU().to(device)

    # Move batch to device
    inputs = inputs.to(device)
    pair_indices = pair_indices.to(device)
    targets = targets.to(device)

    print("Executing forward pass...")
    outputs = model(inputs, pair_indices)

    # Assertions for output shapes
    # Model outputs predictions for the full sequence (107), not just the scored part (68)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Output shape mismatch. Expected {(Config.BATCH_SIZE, 107, 5)}, got {outputs.shape}"

    print("Forward pass successful.")

    # 4. Loss Function Verification
    print("Calculating loss...")
    criterion = MCRMSELoss()

    # The loss function should automatically slice the outputs (107) to match targets (68)
    loss = criterion(outputs, targets)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # Verify metric calculation (scored_only=True)
    # scored_only selects columns [0, 1, 3] corresponding to reactivity, deg_Mg_pH10, deg_Mg_50C
    metric_val = mcrmse_metric(outputs, targets, scored_only=True)
    assert isinstance(metric_val, float), "Metric should return a float"

    print(f"Loss: {loss.item():.4f}, Scored MCRMSE: {metric_val:.4f}")

    # 5. Training Loop Demonstration
    print("Running training epoch...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    avg_train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch finished. Avg Train Loss: {avg_train_loss:.4f}")

    print("Running validation...")
    val_score = validate(model, val_loader, device)
    print(f"Validation Score: {val_score:.4f}")

    # Save the model manually to simulate the training process saving the best model
    # This is required for predict_and_submit to work
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    assert os.path.exists(Config.BEST_MODEL_PATH), "Model file was not saved."
    print("Model saved.")

    # 6. Inference and Submission
    print("Generating submission...")
    # predict_and_submit loads the model from Config.BEST_MODEL_PATH
    # It uses the full test set (which is small, 240 samples), so it runs quickly.
    predict_and_submit()

    # Verify submission file
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Expected rows: Number of test samples * Seq Length (107)
    # We need to know the number of test samples.
    # Since we can't easily get len(test_loader.dataset) without reloading,
    # we know from metadata description that test.json has 240 lines.
    expected_rows = 240 * 107

    assert (
        len(submission_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(submission_df)}"

    # Check columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(submission_df.columns) == expected_cols, "Submission columns mismatch."

    print("Submission verified successfully.")
    print("Demonstration complete.")


if __name__ == "__main__":
    main()
