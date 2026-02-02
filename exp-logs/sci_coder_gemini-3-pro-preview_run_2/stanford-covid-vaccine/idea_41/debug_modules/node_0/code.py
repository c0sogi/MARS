import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.data import load_data
from library.model import RCRDN
from library.loss import MCRMSELoss
from library.train import run_training


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configure for Speed
    # We override Config attributes to run a fast debug session
    print("Configuring parameters for fast execution...")
    Config.EPOCHS = 2
    Config.DEBUG = True  # Limits data to 100 samples
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small debug run

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading & Verification
    print("\n[Step 1] Verifying Data Loading...")
    # Load train data (debug mode will load subset)
    train_dataset = load_data("train", load_cached_data=False, debug=True)

    # Check dataset length
    assert (
        len(train_dataset) == Config.DEBUG_SIZE
    ), f"Expected {Config.DEBUG_SIZE} samples in debug mode, got {len(train_dataset)}"

    # Get a single sample
    x, p_idx, p_mask, y = train_dataset[0]

    # Verify shapes
    # Input features: (107, 18)
    assert x.shape == (
        Config.SEQ_LENGTH,
        Config.NUM_NODE_FEATURES,
    ), f"Input shape mismatch. Expected ({Config.SEQ_LENGTH}, {Config.NUM_NODE_FEATURES}), got {x.shape}"

    # Partner indices: (107,)
    assert p_idx.shape == (
        Config.SEQ_LENGTH,
    ), f"Partner index shape mismatch. Expected ({Config.SEQ_LENGTH},), got {p_idx.shape}"

    # Targets: (107, 5)
    assert y.shape == (
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Target shape mismatch. Expected ({Config.SEQ_LENGTH}, {Config.NUM_TARGETS}), got {y.shape}"

    print("Data loading verification passed.")

    # 3. Model Initialization & Forward Pass
    print("\n[Step 2] Verifying Model Architecture...")
    model = RCRDN().to(device)

    # Create a batch
    loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    batch_x, batch_p_idx, batch_p_mask, batch_y = next(iter(loader))

    batch_x = batch_x.to(device)
    batch_p_idx = batch_p_idx.to(device)
    batch_p_mask = batch_p_mask.to(device)
    batch_y = batch_y.to(device)

    # Forward pass
    # Model returns a list of predictions (one per recycling cycle)
    preds_list = model(batch_x, batch_p_idx, batch_p_mask)

    # Verify recycling output
    assert isinstance(
        preds_list, list
    ), "Model output should be a list (recycling iterations)."
    assert (
        len(preds_list) == Config.N_CYCLES
    ), f"Expected {Config.N_CYCLES} recycling outputs, got {len(preds_list)}"

    # Verify shape of final prediction: (B, 5, L) -> Model outputs channels first usually in PyTorch convs,
    # but let's check the specific implementation in library/model.py.
    # In library/model.py: y_next = logits.transpose(1, 2). logits is (B, L, 5).
    # So output is indeed (B, 5, L).
    final_pred = preds_list[-1]
    expected_shape = (2, Config.NUM_TARGETS, Config.SEQ_LENGTH)
    assert (
        final_pred.shape == expected_shape
    ), f"Prediction shape mismatch. Expected {expected_shape}, got {final_pred.shape}"

    print("Model forward pass verification passed.")

    # 4. Loss Function Verification
    print("\n[Step 3] Verifying Loss Function...")
    criterion = MCRMSELoss(weights=[0.5, 1.0])

    # Compute loss
    loss = criterion(preds_list, batch_y)

    # Check validity
    assert torch.is_tensor(loss), "Loss should be a tensor."
    assert loss.item() >= 0, "Loss should be non-negative."
    assert not torch.isnan(loss), "Loss should not be NaN."

    print(f"Loss verification passed. Loss value: {loss.item():.4f}")

    # 5. Full Pipeline Execution
    print("\n[Step 4] Running Training Pipeline (2 Epochs)...")
    # run_training handles loading data, training loop, validation, and submission generation
    # We pass debug=True to ensure it uses the small dataset
    run_training(debug=True, epochs=Config.EPOCHS)

    print("Training pipeline execution completed.")

    # 6. Submission Validation
    print("\n[Step 5] Validating Submission File...")
    submission_path = Config.SUBMISSION_PATH

    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        print(f"Submission file found at {submission_path}")
        print(f"Shape: {sub_df.shape}")

        # Expected rows: 240 test samples * 107 positions = 25680
        # Expected columns: id_seqpos + 5 targets = 6 columns
        expected_rows = 240 * 107
        expected_cols = 6

        assert (
            sub_df.shape[0] == expected_rows
        ), f"Submission row count mismatch. Expected {expected_rows}, got {sub_df.shape[0]}"
        assert (
            sub_df.shape[1] == expected_cols
        ), f"Submission column count mismatch. Expected {expected_cols}, got {sub_df.shape[1]}"

        # Check column names
        expected_names = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        assert (
            list(sub_df.columns) == expected_names
        ), f"Submission columns mismatch. Expected {expected_names}, got {list(sub_df.columns)}"

        print("Submission file validation passed.")
    else:
        raise FileNotFoundError(
            f"Submission file was not generated at {submission_path}"
        )

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
