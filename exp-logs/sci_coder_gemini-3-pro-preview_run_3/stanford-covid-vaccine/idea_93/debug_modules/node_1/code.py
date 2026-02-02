import os
import shutil
import torch
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_structure_indices, MCRMSELoss
from library.data import get_dataset, RNADataset
from library.layers import StabilizedGLUInteraction
from library.model import HighCapacityBiGRU
from library.train import train_model, generate_submission


def run_demo():
    # =========================================================================
    # 1. Setup Configuration for Demo
    # =========================================================================
    print("--- 1. Setting up Configuration ---")

    # Override Config to use a separate demo directory and run faster
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = Config.WORKING_DIR

    # Reduce hyperparameters for speed
    Config.CONV_FILTERS = 32
    Config.GRU_HIDDEN = 64
    Config.NUM_LAYERS = 2
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8

    # Ensure clean state
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated: Reduced model size and epochs for demo.")

    # =========================================================================
    # 2. Demonstrate Utils
    # =========================================================================
    print("\n--- 2. Demonstrating Utils ---")

    # Test Structure Parsing
    structure_str = "((..))"
    indices = get_structure_indices(structure_str)
    print(f"Structure: {structure_str}")
    print(f"Indices: {indices}")

    # Verification:
    # Index 0 '(' pairs with 5 ')'
    # Index 1 '(' pairs with 4 ')'
    # Indices 2, 3 '.' are unpaired (-1)
    expected_indices = np.array([5, 4, -1, -1, 1, 0], dtype=np.int32)
    np.testing.assert_array_equal(indices, expected_indices)
    print("Utils: get_structure_indices logic verified.")

    # Test Loss Function
    criterion = MCRMSELoss()
    # Dummy preds and targets: (Batch=2, Length=3, Channels=2)
    # Case: Col 0 perfect match, Col 1 error of 1.0
    preds = torch.tensor(
        [[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]]
    )
    targets = torch.tensor(
        [[[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]], [[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]]]
    )

    loss = criterion(preds, targets)
    print(f"MCRMSE Loss: {loss.item()}")
    # RMSE(Col0)=0, RMSE(Col1)=1. Mean=0.5
    assert abs(loss.item() - 0.5) < 1e-6
    print("Utils: MCRMSELoss logic verified.")

    # =========================================================================
    # 3. Demonstrate Data Processing
    # =========================================================================
    print("\n--- 3. Demonstrating Data Processing ---")

    # Load validation data (smaller than train).
    # load_cached_data=False forces processing from metadata parquet.
    inputs, targets = get_dataset("val", load_cached_data=False)

    print(f"Keys in inputs: {list(inputs.keys())}")
    print(f"Feature Tensor (X) Shape: {inputs['X'].shape}")
    print(f"Target Tensor Shape: {targets.shape}")

    # Verify Shapes
    # X: (N_samples, 107, 14)
    assert inputs["X"].ndim == 3
    assert inputs["X"].shape[1] == Config.SEQ_LEN
    assert inputs["X"].shape[2] == Config.INPUT_CHANNELS

    # Targets: (N_samples, 68, 5)
    assert targets.ndim == 3
    assert targets.shape[1] == Config.SEQ_SCORED
    assert targets.shape[2] == 5

    # Verify Dataset Wrapper
    ds = RNADataset(inputs, targets)
    sample_item, sample_y = ds[0]
    assert torch.is_tensor(sample_item["X"])
    assert torch.is_tensor(sample_y)
    print("Data: Dataset loading and processing verified.")

    # =========================================================================
    # 4. Demonstrate Layers
    # =========================================================================
    print("\n--- 4. Demonstrating Layers ---")

    # Initialize Interaction Module
    # Hidden dim in model is GRU_HIDDEN * 2 (bidirectional)
    hidden_dim = Config.GRU_HIDDEN * 2
    layer = StabilizedGLUInteraction(hidden_dim)

    # Create dummy inputs
    B, L = 4, Config.SEQ_LEN
    h_dummy = torch.randn(B, L, hidden_dim)
    adj_dummy = torch.randint(0, L, (B, L))
    mask_dummy = torch.ones(B, L)

    # Forward pass
    out = layer(h_dummy, adj_dummy, mask_dummy)
    print(f"Layer Input Shape: {h_dummy.shape}")
    print(f"Layer Output Shape: {out.shape}")

    assert out.shape == h_dummy.shape
    print("Layers: StabilizedGLUInteraction forward pass verified.")

    # =========================================================================
    # 5. Demonstrate Model
    # =========================================================================
    print("\n--- 5. Demonstrating Model ---")

    model = HighCapacityBiGRU(Config)

    # Dummy input features (B, L, 14)
    x_dummy = torch.randn(B, L, Config.INPUT_CHANNELS)

    preds = model(x_dummy, adj_dummy, mask_dummy)
    print(f"Model Output Shape: {preds.shape}")

    # Expected output: (B, L, 5)
    assert preds.shape == (B, L, 5)
    print("Model: HighCapacityBiGRU forward pass verified.")

    # =========================================================================
    # 6. Demonstrate Training Pipeline
    # =========================================================================
    print("\n--- 6. Demonstrating Training Pipeline ---")

    # Run training with debug=True.
    # This limits the dataset to 100 samples for quick execution.
    print("Starting training loop (debug mode)...")
    best_score = train_model(debug=True, epochs=Config.EPOCHS, patience=1)

    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Training successful. Model saved to: {best_model_path}")
    else:
        raise FileNotFoundError("Training failed to save best_model.pth")

    # =========================================================================
    # 7. Demonstrate Submission Generation
    # =========================================================================
    print("\n--- 7. Demonstrating Submission Generation ---")

    # Generate submission using the trained model
    generate_submission()

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(sub_path):
        print(f"Submission generated at: {sub_path}")

        # Verify submission content
        df_sub = pd.read_csv(sub_path)
        print(f"Submission DataFrame Shape: {df_sub.shape}")

        # Check columns
        expected_cols = ["id_seqpos"] + Config.TARGET_COLS
        assert list(df_sub.columns) == expected_cols

        # Check rows: 240 test samples * 107 positions = 25680
        expected_rows = 240 * 107
        assert (
            len(df_sub) == expected_rows
        ), f"Expected {expected_rows} rows, got {len(df_sub)}"

        print("Submission: Format and dimensions verified.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
