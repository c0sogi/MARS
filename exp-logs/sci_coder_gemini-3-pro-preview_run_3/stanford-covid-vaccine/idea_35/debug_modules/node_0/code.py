import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import provided library components
from library.config import Config
from library.data_utils import get_dataloaders, one_hot_encode, get_structure_adj
from library.model import DDCGBiGRU
from library.loss_metric import MCRMSELoss, competition_metric
from library.train_utils import train_model, generate_submission, set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demonstration():
    print("==== RNA Degradation Prediction Pipeline Demonstration ====")

    # 1. Configuration Override for Speed and Demo
    # We modify the Config class attributes directly to run a lightweight version
    print("\n[1] Configuring environment for fast demonstration...")
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.DEBUG_SAMPLES = 32  # Only use 32 samples from each dataset
    Config.NUM_WORKERS = 0  # Use main process for data loading to avoid overhead
    Config.PATIENCE = 2  # Short patience
    Config.HIDDEN_SIZE = 64  # Smaller model for speed
    Config.CONV_FILTERS = 32  # Smaller model for speed
    Config.NUM_LAYERS = 2  # Fewer layers

    # Ensure reproducibility
    set_seed(Config.SEED)
    print("Configuration updated for debug mode.")

    # 2. Verify Data Processing Logic
    print("\n[2] Verifying Data Processing Logic...")

    # Test one_hot_encode
    seq = "AGCU"
    struct = ".(.)"
    loop = "ESHE"
    # Expected length 4, Input Dim 14
    encoded = one_hot_encode(seq, struct, loop)
    assert encoded.shape == (4, 14), f"Encoding shape mismatch: {encoded.shape}"
    # Check A (index 0)
    assert encoded[0, 0] == 1.0, "Sequence encoding incorrect"
    # Check . (index 4+0=4)
    assert encoded[0, 4] == 1.0, "Structure encoding incorrect"
    print("Feature encoding verified.")

    # Test get_structure_adj
    test_struct = "((..))"
    adj, mask = get_structure_adj(test_struct)
    # Indices: 0 paired with 5, 1 paired with 4. 2,3 unpaired.
    assert adj[0] == 5 and adj[5] == 0, "Adjacency index logic incorrect"
    assert adj[1] == 4 and adj[4] == 1, "Adjacency index logic incorrect"
    assert mask[0] == 1.0 and mask[2] == 0.0, "Pair mask logic incorrect"
    print("Structure adjacency logic verified.")

    # 3. Verify Data Loaders
    print("\n[3] Verifying Data Loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,  # Force reprocessing to test logic
        debug=True,
    )

    # Check batch structure
    batch = next(iter(train_loader))
    features = batch["features"]
    adj_indices = batch["adj_indices"]
    pair_mask = batch["pair_mask"]
    targets = batch["targets"]

    # Shapes: (B, 107, 14), (B, 107), (B, 107, 1), (B, 68, 5)
    assert features.shape == (
        Config.BATCH_SIZE,
        107,
        14,
    ), f"Feature shape error: {features.shape}"
    assert adj_indices.shape == (
        Config.BATCH_SIZE,
        107,
    ), f"Adj shape error: {adj_indices.shape}"
    assert pair_mask.shape == (
        Config.BATCH_SIZE,
        107,
        1,
    ), f"Mask shape error: {pair_mask.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        68,
        5,
    ), f"Target shape error: {targets.shape}"
    print("DataLoaders and batch shapes verified.")

    # 4. Verify Model Architecture
    print("\n[4] Verifying Model Architecture...")
    device = torch.device("cpu")  # Use CPU for shape check
    model = DDCGBiGRU().to(device)

    # Forward pass
    with torch.no_grad():
        output = model(features, adj_indices, pair_mask)

    # Output shape should be (B, 107, 5) - predicting 5 targets for all 107 positions
    assert output.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Model output shape error: {output.shape}"
    print("Model forward pass verified.")

    # 5. Verify Loss Calculation
    print("\n[5] Verifying Loss Function (MCRMSE)...")
    criterion = MCRMSELoss()

    # Loss expects output (B, 107, 5) and targets (B, 68, 5)
    # It should slice output internally
    loss = criterion(output, targets)

    assert isinstance(loss.item(), float), "Loss did not return a scalar float"
    assert loss.item() >= 0, "Loss value is negative"
    print(f"Loss calculation verified. Initial Loss: {loss.item():.4f}")

    # 6. Execute Training Loop
    print("\n[6] Executing Training Loop (Debug Mode)...")
    # This calls train_model from library.train_utils which handles the loop, validation, and saving
    train_model(debug=True)

    # Check if model was saved
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model file was not created after training.")

    # 7. Execute Submission Generation
    print("\n[7] Generating Submission...")
    generate_submission(debug=True)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated at {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {df_sub.shape}")

        # Expected rows: Num_Test_Samples (debug=32) * Seq_Len (107) = 3424
        expected_rows = Config.DEBUG_SAMPLES * Config.SEQ_LEN
        assert (
            len(df_sub) == expected_rows
        ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

        # Check columns
        expected_cols = ["id_seqpos"] + Config.TARGET_COLS
        assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

        print("Submission file verified.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    run_demonstration()
