import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import from the provided library
import library.config as config_module
from library.config import ModelConfig
from library.utils import set_seed, mcrmse_loss, get_device
from library.dataset import process_data, RNADataset, load_or_process_data
from library.model import RNARegressor
from library.train import run_training
from library.predict import run_inference


def create_dummy_data(base_dir):
    """
    Creates dummy parquet files mimicking the competition dataset structure.
    """
    os.makedirs(base_dir, exist_ok=True)

    seq_len = 107
    scored_len = 68
    n_samples = 20

    # Generate random sequences and structures
    ids = [f"id_{i:05d}" for i in range(n_samples)]
    sequences = [
        "".join(np.random.choice(list("AGUC"), size=seq_len)) for _ in range(n_samples)
    ]
    # Simple dummy structure: all dots
    structures = ["." * seq_len for _ in range(n_samples)]
    # Simple dummy loops: all External
    loops = ["E" * seq_len for _ in range(n_samples)]

    # Generate random targets
    # Targets are lists of floats in the parquet file
    reactivity = [np.random.rand(scored_len).tolist() for _ in range(n_samples)]
    deg_Mg_pH10 = [np.random.rand(scored_len).tolist() for _ in range(n_samples)]
    deg_Mg_50C = [np.random.rand(scored_len).tolist() for _ in range(n_samples)]

    # Create DataFrames
    df_train = pd.DataFrame(
        {
            "id": ids,
            "sequence": sequences,
            "structure": structures,
            "predicted_loop_type": loops,
            "reactivity": reactivity,
            "deg_Mg_pH10": deg_Mg_pH10,
            "deg_Mg_50C": deg_Mg_50C,
        }
    )

    df_test = pd.DataFrame(
        {
            "id": ids,
            "sequence": sequences,
            "structure": structures,
            "predicted_loop_type": loops,
        }
    )

    # Save to parquet
    train_path = os.path.join(base_dir, "train_dummy.parquet")
    val_path = os.path.join(base_dir, "val_dummy.parquet")
    test_path = os.path.join(base_dir, "test_dummy.parquet")

    df_train.to_parquet(train_path, index=False)
    df_train.to_parquet(val_path, index=False)  # Use same for val
    df_test.to_parquet(test_path, index=False)

    return train_path, val_path, test_path


def verify_data_processing(df_path):
    """
    Verifies the data processing logic in library.dataset.
    """
    print("\n[Verification] Data Processing...")
    df = pd.read_parquet(df_path)

    # Test process_data function
    data_dict = process_data(df, mode="train")

    # Check keys
    expected_keys = ["seq", "loop", "dist", "mask", "id", "target"]
    for k in expected_keys:
        assert k in data_dict, f"Missing key {k} in processed data"

    # Check shapes
    n_samples = len(df)
    assert data_dict["seq"].shape == (n_samples, 107)
    assert data_dict["target"].shape == (n_samples, 68, 3)
    assert data_dict["mask"].shape == (n_samples, 107)

    # Test Dataset Class
    ds = RNADataset(data_dict, mode="train")
    item = ds[0]

    assert isinstance(item["seq"], torch.Tensor)
    assert item["seq"].shape == (107,)
    assert item["target"].shape == (68, 3)

    print("Data processing verified successfully.")


def verify_model_architecture(config):
    """
    Verifies the RNARegressor model forward and backward pass.
    """
    print("\n[Verification] Model Architecture...")
    device = get_device()
    model = RNARegressor(config).to(device)

    # Create dummy batch
    batch_size = 4
    seq_len = 107

    seq = torch.randint(0, 4, (batch_size, seq_len)).to(device)
    loop = torch.randint(0, 7, (batch_size, seq_len)).to(device)
    dist = torch.randn(batch_size, seq_len).to(device)
    mask = torch.ones(batch_size, seq_len).to(device)

    # Forward pass
    output = model(seq, loop, dist, mask)

    # Check output shape: (B, 107, 3)
    assert output.shape == (
        batch_size,
        seq_len,
        3,
    ), f"Output shape mismatch: {output.shape}"

    # Backward pass check
    target = torch.randn(batch_size, 68, 3).to(device)
    scored_output = output[:, :68, :]
    loss = mcrmse_loss(target, scored_output)

    loss.backward()

    # Check if gradients are populated
    assert model.head.weight.grad is not None, "Gradients not computed for head layer"

    print("Model architecture verified successfully.")


def main():
    # 1. Setup
    set_seed(42)
    working_dir = "./working/demo_run"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir)

    print(f"Running demo in {working_dir}")

    # 2. Create Dummy Data
    train_path, val_path, test_path = create_dummy_data(working_dir)

    # 3. Modify Config for Demo (Speed Optimization)
    # We modify the class attributes directly since the library modules use the class directly
    ModelConfig.train_file = train_path
    ModelConfig.val_file = val_path
    ModelConfig.test_file = test_path
    ModelConfig.output_dir = os.path.join(working_dir, "output")
    ModelConfig.submission_file = os.path.join(working_dir, "submission.csv")

    # Reduce model size and training duration for speed
    ModelConfig.hidden_dim = 32
    ModelConfig.num_layers = 2
    ModelConfig.num_rbf = 16
    ModelConfig.batch_size = 4
    ModelConfig.num_epochs = 2
    ModelConfig.learning_rate = 1e-3

    # 4. Verify Components
    verify_data_processing(train_path)
    verify_model_architecture(ModelConfig)

    # 5. Run Training Pipeline
    print("\n[Execution] Running Training Pipeline...")
    # This calls library.train.run_training which uses the modified ModelConfig
    run_training()

    # Verify training artifacts
    best_model_path = os.path.join(ModelConfig.output_dir, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."
    print("Training pipeline completed. Checkpoint saved.")

    # 6. Run Inference Pipeline
    print("\n[Execution] Running Inference Pipeline...")
    # This calls library.predict.run_inference
    run_inference()

    # Verify submission file
    assert os.path.exists(
        ModelConfig.submission_file
    ), "Submission file was not generated."

    # Check submission content format
    sub_df = pd.read_csv(ModelConfig.submission_file)
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    # Check row count: 20 samples * 107 length = 2140 rows
    assert len(sub_df) == 20 * 107, f"Expected 2140 rows, got {len(sub_df)}"

    print("Inference pipeline completed. Submission generated.")
    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    main()
