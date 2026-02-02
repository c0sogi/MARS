import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, MCRMSELoss, mcrmse_loss
from library.data import process_data, RNADataset, get_dataloaders
from library.model import RNAModel
from library.train import train_epoch, validate


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> Setting up demonstration configuration...")

    # Override Config for speed and demo purposes
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Reduce model complexity for fast CPU execution
    Config.HIDDEN_DIM = 32  # Reduced from 384
    Config.NUM_LAYERS = 2  # Reduced from 4
    Config.CONV_FILTERS = 16  # Reduced from 256
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead

    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = torch.device("cpu")  # Use CPU for this lightweight demo

    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Processing
    # ==========================================
    print(">>> Processing Data...")

    # We use the provided process_data function.
    # It reads from metadata/train.parquet and saves cache to working/demo_execution.
    # Note: We set load_cached_data=False to force processing for the demo.
    train_inputs, train_targets, train_pairs, train_ids = process_data(
        Config.TRAIN_DATA_PATH, "train_demo", load_cached_data=False
    )

    # Verify Data Shapes
    # Inputs: (N, 107, 14) -> Sequence(4) + Structure(3) + LoopType(7)
    assert train_inputs.ndim == 3
    assert train_inputs.shape[1] == 107
    assert train_inputs.shape[2] == 14

    # Targets: (N, 107, 5)
    assert train_targets.ndim == 3
    assert train_targets.shape[1] == 107
    assert train_targets.shape[2] == 5

    # Pair Indices: (N, 107)
    assert train_pairs.ndim == 2
    assert train_pairs.shape[1] == 107

    print(f"    Train Data Loaded. Samples: {len(train_inputs)}")

    # ==========================================
    # 3. Dataset & DataLoader
    # ==========================================
    print(">>> Initializing Dataset and DataLoader...")

    # Create a small subset for the demo to run instantly
    subset_size = 32
    demo_ds = RNADataset(
        train_inputs[:subset_size],
        train_targets[:subset_size],
        train_pairs[:subset_size],
        train_ids[:subset_size],
    )

    demo_loader = DataLoader(demo_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Fetch one batch to verify structure
    batch = next(iter(demo_loader))
    inputs = batch["inputs"]
    targets = batch["targets"]
    pair_index = batch["pair_index"]
    pair_mask = batch["pair_mask"]

    assert inputs.shape == (Config.BATCH_SIZE, 107, 14)
    assert targets.shape == (Config.BATCH_SIZE, 107, 5)
    assert pair_index.shape == (Config.BATCH_SIZE, 107)
    assert pair_mask.shape == (Config.BATCH_SIZE, 107)

    print("    Batch shapes verified.")

    # ==========================================
    # 4. Model Initialization & Forward Pass
    # ==========================================
    print(">>> Initializing Model...")

    model = RNAModel().to(device)

    print(">>> Running Forward Pass...")
    # Move batch to device
    b_inputs = inputs.to(device)
    b_pair_index = pair_index.to(device)
    b_pair_mask = pair_mask.to(device)

    # Forward
    outputs = model(b_inputs, pair_index=b_pair_index, pair_mask=b_pair_mask)

    # Verify Output Shape: (Batch, SeqLen, OutputDim)
    assert outputs.shape == (Config.BATCH_SIZE, 107, 5)
    assert torch.isfinite(outputs).all(), "Model output contains NaNs or Infs"

    print("    Forward pass successful.")

    # ==========================================
    # 5. Loss Calculation Verification
    # ==========================================
    print(">>> Verifying Loss Function...")

    criterion = MCRMSELoss()
    loss = criterion(outputs, targets.to(device))

    print(f"    Calculated Loss: {loss.item():.4f}")

    # Manual verification of MCRMSE logic
    # Create dummy data
    y_true = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    y_pred = torch.tensor([[[1.5, 2.0], [2.0, 5.0]]])  # (1, 2, 2)
    # Col 0: (1-1.5)^2 + (3-2)^2 = 0.25 + 1 = 1.25 -> Mean=0.625 -> RMSE=sqrt(0.625) ~= 0.7905
    # Col 1: (2-2)^2 + (4-5)^2 = 0 + 1 = 1 -> Mean=0.5 -> RMSE=sqrt(0.5) ~= 0.7071
    # MCRMSE = (0.7905 + 0.7071) / 2 = 0.7488

    manual_loss = criterion(y_pred, y_true).item()
    expected_loss = (np.sqrt(0.625) + np.sqrt(0.5)) / 2

    assert np.isclose(manual_loss, expected_loss, atol=1e-4)
    print("    Loss function logic verified.")

    # ==========================================
    # 6. Training Loop Demo
    # ==========================================
    print(">>> Running Training Epoch...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train for 1 epoch on the subset
    epoch_loss = train_epoch(model, demo_loader, optimizer, criterion, device)
    print(f"    Epoch Loss: {epoch_loss:.4f}")

    # Validate
    # Note: validate() calculates MCRMSE only on specific scored columns (0, 1, 3)
    # and only on the first 68 positions.
    print(">>> Running Validation...")
    val_score = validate(model, demo_loader, criterion, device)
    print(f"    Validation Score (Scored Targets Only): {val_score:.4f}")

    # ==========================================
    # 7. Inference & Submission Generation
    # ==========================================
    print(">>> Generating Submission Demo...")

    # Load Test Data (No targets)
    test_inputs, _, test_pairs, test_ids = process_data(
        Config.TEST_DATA_PATH, "test_demo", load_cached_data=False
    )

    # Create Test Dataset (Subset for demo)
    test_subset_size = 10
    test_ds = RNADataset(
        test_inputs[:test_subset_size],
        None,
        test_pairs[:test_subset_size],
        test_ids[:test_subset_size],
    )
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            t_inputs = batch["inputs"].to(device)
            t_pair_index = batch["pair_index"].to(device)
            t_pair_mask = batch["pair_mask"].to(device)

            # Forward
            t_outputs = model(t_inputs, pair_index=t_pair_index, pair_mask=t_pair_mask)

            all_preds.append(t_outputs.cpu().numpy())
            all_ids.extend(batch["id"])

    # Concatenate predictions: (N, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Flatten for submission format
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(all_ids):
        pred_matrix = all_preds[i]  # (107, 5)
        seq_len = pred_matrix.shape[0]

        for seqpos in range(seq_len):
            row_id = f"{sample_id}_{seqpos}"
            row_values = pred_matrix[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_values[col_idx])

            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    # Verify Submission Format
    print(f"    Generated Submission Shape: {submission_df.shape}")
    expected_rows = test_subset_size * 107
    assert len(submission_df) == expected_rows
    assert list(submission_df.columns) == ["id_seqpos"] + target_cols

    # Save demo submission
    sub_path = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"    Submission saved to {sub_path}")

    print(">>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
