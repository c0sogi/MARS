import os
import torch
import numpy as np
import pandas as pd
import shutil
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data import get_loader
from library.model import SR_DCN
from library.train import masked_mcrmse, train_epoch, validate


def run_demo():
    # ------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # ------------------------------------------------------------------------
    print("1. Setting up configuration for demo...")

    # Set a fixed seed for reproducibility
    seed_everything(42)

    # Define a temporary working directory for this demo
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config parameters to ensure speed and isolation
    Config.WORKING_DIR = demo_dir
    Config.INPUT_DIR = "./input"
    Config.METADATA_DIR = "./metadata"

    # Point caches to the demo directory so we don't overwrite real training data
    Config.TRAIN_CACHE = os.path.join(demo_dir, "train_data.npz")
    Config.VAL_CACHE = os.path.join(demo_dir, "val_data.npz")
    Config.TEST_CACHE = os.path.join(demo_dir, "test_data.npz")
    Config.BEST_MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Enable DEBUG mode to load only 50 samples
    Config.DEBUG = True
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    print(f"   Working Directory: {Config.WORKING_DIR}")

    # ------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # ------------------------------------------------------------------------
    print("\n2. Demonstrating Data Loading...")

    # Load training data
    # This will trigger processing of the first 50 rows of train.csv and cache it
    train_loader = get_loader("train", shuffle=True, load_cached_data=False)
    val_loader = get_loader("val", shuffle=False, load_cached_data=False)

    # Fetch a single batch to verify shapes
    x, y, p_idx, mask, ids = next(iter(train_loader))

    print(f"   Batch shapes:")
    print(
        f"   Inputs (x): {x.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LENGTH}, 18])"
    )
    print(
        f"   Targets (y): {y.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LENGTH}, 5])"
    )
    print(
        f"   Partner Indices: {p_idx.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LENGTH}])"
    )
    print(
        f"   Mask: {mask.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LENGTH}])"
    )

    # Assertions to ensure data integrity
    assert x.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, 18), "Input shape mismatch"
    assert y.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, 5), "Target shape mismatch"
    assert p_idx.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), "Partner index shape mismatch"

    # ------------------------------------------------------------------------
    # 3. Model Instantiation and Forward Pass
    # ------------------------------------------------------------------------
    print("\n3. Demonstrating Model Forward Pass...")

    model = SR_DCN().to(device)

    # Move batch to device
    x = x.to(device)
    p_idx = p_idx.to(device)

    # Create initial recycling tensor (zeros)
    # Shape: (Batch, Seq_Len, 5)
    recycling = torch.zeros((x.size(0), x.size(1), 5), device=device)

    # Perform forward pass
    # The model expects (x, recycling, partner_indices)
    preds = model(x, recycling, p_idx)

    print(
        f"   Output shape: {preds.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LENGTH}, 5])"
    )

    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        5,
    ), "Model output shape mismatch"

    # ------------------------------------------------------------------------
    # 4. Loss Function Verification
    # ------------------------------------------------------------------------
    print("\n4. Verifying Loss Function Logic...")

    # Create synthetic data for loss verification
    # Scored indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    # Case 1: Perfect prediction -> Loss should be 0
    syn_preds = torch.tensor([[[0.5, 0.5, 0.0, 0.5, 0.0]]], device=device)
    syn_targets = torch.tensor([[[0.5, 0.5, 0.0, 0.5, 0.0]]], device=device)
    syn_mask = torch.tensor([[1.0]], device=device)

    loss_zero = masked_mcrmse(syn_preds, syn_targets, syn_mask, scored_indices)
    assert torch.isclose(
        loss_zero, torch.tensor(0.0, device=device), atol=1e-4
    ), f"Loss should be 0, got {loss_zero}"

    # Case 2: Known error
    # Target: [1.0, 1.0, ..., 1.0] at scored indices
    # Pred:   [0.0, 0.0, ..., 0.0] at scored indices
    # MSE per col = 1.0, RMSE per col = 1.0, Mean RMSE = 1.0
    syn_preds_err = torch.zeros((1, 1, 5), device=device)
    syn_targets_err = torch.tensor([[[1.0, 1.0, 0.0, 1.0, 0.0]]], device=device)

    loss_one = masked_mcrmse(syn_preds_err, syn_targets_err, syn_mask, scored_indices)
    # Note: The function adds 1e-8 epsilon, so sqrt(1 + 1e-8) is approx 1.0
    print(f"   Calculated Loss for unit error: {loss_one.item():.4f}")
    assert torch.isclose(
        loss_one, torch.tensor(1.0, device=device), atol=1e-3
    ), "Loss calculation incorrect"

    # ------------------------------------------------------------------------
    # 5. Training Loop Execution
    # ------------------------------------------------------------------------
    print("\n5. Running Training Epoch...")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one training epoch using the library function
    # train_epoch handles the recycling passes (Cold Start + Refinement) internally
    avg_train_loss = train_epoch(model, train_loader, optimizer, device, scored_indices)
    print(f"   Training Epoch Completed. Average Loss: {avg_train_loss:.4f}")

    # Run validation
    val_loss = validate(model, val_loader, device, scored_indices)
    print(f"   Validation Completed. MCRMSE: {val_loss:.4f}")

    # Save the model (simulating the checkpointing)
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    assert os.path.exists(Config.BEST_MODEL_PATH), "Model checkpoint not saved"

    # ------------------------------------------------------------------------
    # 6. Inference and Submission Generation
    # ------------------------------------------------------------------------
    print("\n6. Generating Submission...")

    # Load test loader (Debug mode limits this to 50 samples)
    test_loader = get_loader("test", shuffle=False, load_cached_data=False)

    model.eval()
    results = []
    target_cols = Config.TARGET_COLS

    with torch.no_grad():
        for x_test, _, p_idx_test, _, ids_test in test_loader:
            x_test = x_test.to(device)
            p_idx_test = p_idx_test.to(device)

            # Pass 1: Cold Start
            b, l, _ = x_test.shape
            recycling_zero = torch.zeros((b, l, 5), device=device)
            pred1 = model(x_test, recycling_zero, p_idx_test)

            # Pass 2: Refinement
            pred2 = model(x_test, pred1, p_idx_test)

            preds_np = pred2.cpu().numpy()

            for i, sample_id in enumerate(ids_test):
                sample_preds = preds_np[i]  # (107, 5)
                for seqpos in range(Config.SEQ_LENGTH):
                    row_id = f"{sample_id}_{seqpos}"
                    vals = sample_preds[seqpos]

                    row_dict = {"id_seqpos": row_id}
                    for k, col_name in enumerate(target_cols):
                        row_dict[col_name] = float(vals[k])
                    results.append(row_dict)

    submission_df = pd.DataFrame(results)

    # Reorder columns
    cols_order = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols_order]

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"   Submission saved to {Config.SUBMISSION_PATH}")
    print(f"   Submission shape: {submission_df.shape}")

    # Verify submission integrity
    assert not submission_df.isnull().values.any(), "Submission contains NaNs"
    assert len(submission_df) > 0, "Submission is empty"

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
