import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import library modules
from library.config import Config
from library.data_utils import parse_structure_pairs
from library.dataset import RNADataset
from library.model import LatentSpatialBiGRU
from library.loss import MCRMSELoss
from library.trainer import Trainer, set_seed

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("Starting RNA Degradation Prediction Demo...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Setting up Configuration...")

    class DemoConfig(Config):
        # Override settings for fast execution
        working_dir = "./working/demo_execution"
        model_save_path = os.path.join(working_dir, "best_model.pth")
        submission_path = os.path.join(working_dir, "submission.csv")

        # Training params
        epochs = 2
        batch_size = 16
        num_workers = 0  # Avoid multiprocessing overhead for small demo

        # Ensure directories exist
        def __init__(self):
            super().__init__()
            os.makedirs(self.working_dir, exist_ok=True)

    config = DemoConfig()
    set_seed(config.seed)
    print(f"    Working Directory: {config.working_dir}")
    print(f"    Device: {config.device}")
    print(f"    Epochs: {config.epochs}")

    # ==========================================
    # 2. Verify Data Utilities
    # ==========================================
    print("\n[2] Verifying Data Utilities...")

    # Test structure parsing
    # Structure: ((..)) -> Indices: 012345
    # Pairs: 0-5, 1-4, 2-2, 3-3
    test_struct = "((..))"
    expected_pairs = np.array([5, 4, 2, 3, 1, 0])
    calculated_pairs = parse_structure_pairs(test_struct)

    assert np.array_equal(
        calculated_pairs, expected_pairs
    ), f"Structure parsing failed. Expected {expected_pairs}, got {calculated_pairs}"
    print("    parse_structure_pairs(): Passed")

    # ==========================================
    # 3. Verify Dataset
    # ==========================================
    print("\n[3] Verifying Dataset...")

    # Load training dataset
    train_dataset = RNADataset(split="train", load_cached_data=True)
    print(f"    Train Dataset Size: {len(train_dataset)}")

    # Fetch one sample
    sample = train_dataset[0]
    input_tensor = sample["input"]
    pair_index = sample["pair_index"]
    target_tensor = sample["target"]

    # Verify Shapes
    # Input: (SeqLen=107, Channels=14)
    assert input_tensor.shape == (
        107,
        14,
    ), f"Input shape mismatch: {input_tensor.shape}"
    # Pair Index: (SeqLen=107)
    assert pair_index.shape == (107,), f"Pair index shape mismatch: {pair_index.shape}"
    # Target: (PredLen=68, OutputDim=5)
    assert target_tensor.shape == (
        68,
        5,
    ), f"Target shape mismatch: {target_tensor.shape}"

    print("    Sample shapes verified successfully.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n[4] Verifying Model Architecture...")

    model = LatentSpatialBiGRU(config).to(config.device)

    # Create dummy batch (BatchSize=2)
    dummy_input = input_tensor.unsqueeze(0).repeat(2, 1, 1).to(config.device)
    dummy_pairs = pair_index.unsqueeze(0).repeat(2, 1).to(config.device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input, dummy_pairs)

    # Output shape should be (Batch, SeqLen=107, OutputDim=5)
    expected_shape = (2, 107, 5)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print(f"    Forward pass successful. Output shape: {output.shape}")

    # ==========================================
    # 5. Verify Loss Function
    # ==========================================
    print("\n[5] Verifying Loss Function...")

    criterion = MCRMSELoss()

    # Create dummy targets (Batch=2, SeqLen=68, Dim=5)
    # Model output is (2, 107, 5), Loss should slice it to (2, 68, 5)
    dummy_target = torch.zeros((2, 68, 5)).to(config.device)

    # Case 1: Perfect prediction (first 68 match)
    perfect_pred = torch.zeros((2, 107, 5)).to(config.device)
    loss_zero = criterion(perfect_pred, dummy_target)
    assert torch.isclose(
        loss_zero, torch.tensor(0.0).to(config.device)
    ), f"Loss should be 0 for perfect prediction, got {loss_zero}"

    # Case 2: Constant error
    # Set predictions to 1.0, targets are 0.0. Squared diff is 1.0. RMSE is 1.0. Mean RMSE is 1.0.
    ones_pred = torch.ones((2, 107, 5)).to(config.device)
    loss_one = criterion(ones_pred, dummy_target)
    assert torch.isclose(
        loss_one, torch.tensor(1.0).to(config.device), atol=1e-5
    ), f"Loss should be 1.0 for constant error, got {loss_one}"

    print("    MCRMSELoss verification passed.")

    # ==========================================
    # 6. Run Training Loop
    # ==========================================
    print("\n[6] Running Training Loop (Demo)...")

    trainer = Trainer(config)
    trainer.fit(load_cached_data=True)

    print("    Training execution completed.")

    # ==========================================
    # 7. Inference & Submission Generation
    # ==========================================
    print("\n[7] Generating Submission...")

    # Load best model
    model.load_state_dict(
        torch.load(config.model_save_path, map_location=config.device)
    )
    model.eval()

    # Load Test Data
    test_dataset = RNADataset(split="test", load_cached_data=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    predictions = []
    ids = []

    print(f"    Predicting on {len(test_dataset)} test samples...")

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(config.device)
            pair_indices = batch["pair_index"].to(config.device)
            batch_ids = batch["id"]

            # Forward
            preds = model(inputs, pair_indices)  # (B, 107, 5)
            preds = preds.cpu().numpy()

            predictions.append(preds)
            ids.extend(batch_ids)

    predictions = np.concatenate(predictions, axis=0)  # (N, 107, 5)

    # Format for submission
    # We need to flatten: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Columns in model output order: ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    submission_rows = []
    target_cols = config.target_cols

    for i, sample_id in enumerate(ids):
        sample_preds = predictions[i]  # (107, 5)

        for seqpos in range(sample_preds.shape[0]):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    # Save
    submission_df.to_csv(config.submission_path, index=False)
    print(f"    Submission saved to {config.submission_path}")
    print(f"    Submission shape: {submission_df.shape}")

    # Verify submission structure against sample
    sample_sub = pd.read_csv(config.sample_submission_path)
    # Note: Our generated submission might have different row count if test set size differs from sample_submission
    # or if sample_submission covers different IDs.
    # The prompt says test.json has 240 lines. 240 * 107 = 25680 rows.
    # sample_submission.csv has 25680 rows.

    assert (
        submission_df.shape[1] == sample_sub.shape[1]
    ), "Column count mismatch with sample submission"
    assert list(submission_df.columns) == list(
        sample_sub.columns
    ), "Column names mismatch"

    print("    Submission format verified.")
    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
