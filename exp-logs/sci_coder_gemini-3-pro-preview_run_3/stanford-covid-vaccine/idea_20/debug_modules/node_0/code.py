import os
import torch
import numpy as np
import pandas as pd
import warnings
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, MCRMSELoss
from library.data import get_loaders
from library.model import CGSRBiGRU
from library.train import train_epoch, validate


def run_demo():
    # 1. Setup & Configuration
    # ------------------------
    warnings.filterwarnings("ignore")
    seed_everything(42)
    print("Initializing Demo...")

    # Override Config for a fast, debug execution
    Config.debug = True  # Use only 100 samples
    Config.epochs = 2  # Run only 2 epochs
    Config.batch_size = 8
    Config.working_dir = "./working/demo_execution"

    # Update paths to point to the demo working directory
    os.makedirs(Config.working_dir, exist_ok=True)
    Config.train_cache_path = os.path.join(Config.working_dir, "train_cache_debug.npz")
    Config.val_cache_path = os.path.join(Config.working_dir, "val_cache_debug.npz")
    Config.test_cache_path = os.path.join(Config.working_dir, "test_cache_debug.npz")
    Config.model_save_path = os.path.join(Config.working_dir, "best_model.pth")
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Pipeline Verification
    # -----------------------------
    print("\n[Data] Loading and processing data (Debug Mode)...")
    # load_cached_data=False forces reprocessing of the debug subset
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Fetch a single batch to verify shapes and content
    features, adjacency, targets, masks = next(iter(train_loader))

    print(
        f"Batch Shapes -> Features: {features.shape}, Adjacency: {adjacency.shape}, "
        f"Targets: {targets.shape}, Masks: {masks.shape}"
    )

    # Assertions for Data Integrity
    # Features: (Batch, Seq_Len=107, Channels=14)
    assert features.shape == (Config.batch_size, 107, 14), "Incorrect feature shape"
    # Adjacency: (Batch, Seq_Len=107)
    assert adjacency.shape == (Config.batch_size, 107), "Incorrect adjacency shape"
    # Targets: (Batch, Seq_Len=107, Targets=5)
    assert targets.shape == (Config.batch_size, 107, 5), "Incorrect target shape"

    # Verify Masks: seq_scored is 68. First 68 should be 1, rest 0.
    # Note: We check the first sample in the batch
    assert torch.all(
        masks[0, :68] == 1.0
    ), "Masks should be 1.0 for scored positions (0-67)"
    assert torch.all(
        masks[0, 68:] == 0.0
    ), "Masks should be 0.0 for unscored positions (68-106)"
    print("[Data] Verification Passed.")

    # 3. Model Verification
    # ---------------------
    print("\n[Model] Initializing CGSRBiGRU...")
    model = CGSRBiGRU().to(device)

    # Move batch to device
    features = features.to(device)
    adjacency = adjacency.to(device)

    # Forward Pass
    preds = model(features, adjacency)
    print(f"Prediction Shape: {preds.shape}")

    # Assertions for Model Output
    assert preds.shape == (Config.batch_size, 107, 5), "Model output shape mismatch"
    assert not torch.isnan(preds).any(), "Model produced NaN predictions"
    print("[Model] Verification Passed.")

    # 4. Loss Function Verification
    # -----------------------------
    print("\n[Loss] Verifying MCRMSELoss...")
    criterion = MCRMSELoss()

    # Case A: Perfect predictions (Loss should be 0)
    loss_zero = criterion(preds, preds)
    assert torch.isclose(
        loss_zero, torch.tensor(0.0, device=device), atol=1e-6
    ), f"Loss should be 0 for identical inputs, got {loss_zero.item()}"

    # Case B: Known error
    # Preds = 1.0, Targets = 0.0 -> Diff=1 -> Sq=1 -> Mean=1 -> Sqrt=1 -> Mean=1
    ones = torch.ones_like(preds)
    zeros = torch.zeros_like(preds)
    loss_one = criterion(ones, zeros)
    assert torch.isclose(
        loss_one, torch.tensor(1.0, device=device), atol=1e-6
    ), f"Loss should be 1.0 for |1-0| errors, got {loss_one.item()}"
    print("[Loss] Verification Passed.")

    # 5. Training Loop Execution
    # --------------------------
    print("\n[Train] Executing Training Loop...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)

    # Run 1 Epoch
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")

    # Run Validation
    val_score = validate(model, val_loader, device)
    print(f"Validation MCRMSE: {val_score:.4f}")

    assert train_loss > 0, "Training loss should be positive"
    assert val_score > 0, "Validation score should be positive"
    print("[Train] Loop Execution Passed.")

    # 6. Submission Generation
    # ------------------------
    print("\n[Submission] Generating Submission File...")
    model.eval()
    ids_list = []
    preds_list = []

    # Generate predictions on Test Set
    with torch.no_grad():
        for batch in test_loader:
            feat, adj, batch_ids = batch
            feat = feat.to(device)
            adj = adj.to(device)

            p = model(feat, adj)
            preds_list.append(p.cpu().numpy())
            ids_list.extend(batch_ids)

    all_preds = np.concatenate(preds_list, axis=0)

    # Format for CSV
    submission_data = []
    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 5)
        for seq_pos in range(107):
            row_id = f"{sample_id}_{seq_pos}"
            row_vals = sample_preds[seq_pos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(Config.target_cols):
                row_dict[col_name] = row_vals[col_idx]
            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Verify Submission
    # Number of rows = Num_Test_Samples * Seq_Len
    expected_rows = len(ids_list) * 107
    assert (
        len(submission_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(submission_df)}"

    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
    print(f"Submission Head:\n{submission_df.head(2)}")
    print("[Submission] Verification Passed.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
