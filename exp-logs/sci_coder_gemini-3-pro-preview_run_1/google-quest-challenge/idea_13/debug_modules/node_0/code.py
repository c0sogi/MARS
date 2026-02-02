import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.dataset import get_dataloaders
from library.model import TripleBranchDistilRoBERTa
from library.engine import train_fn, eval_fn, get_optimizer_params


def main():
    print("=== Starting Library Demonstration ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config for speed and isolation
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Create working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # --------------------------------------------------------------------------
    # 2. Utilities Verification
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test Seeding
    seed_everything(Config.SEED)
    rn1 = np.random.rand()
    seed_everything(Config.SEED)
    rn2 = np.random.rand()
    assert rn1 == rn2, "seed_everything failed to produce reproducible numpy results."
    print("    seed_everything: Verified.")

    # Test Spearman Correlation
    # Case 1: Perfect positive correlation
    t1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    p1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    score_perfect = compute_spearmanr(p1, t1)
    assert np.isclose(score_perfect, 1.0), f"Expected 1.0, got {score_perfect}"

    # Case 2: Perfect negative correlation
    p2 = np.array([[0.5, 0.6], [0.3, 0.4], [0.1, 0.2]])
    score_neg = compute_spearmanr(p2, t1)
    assert np.isclose(score_neg, -1.0), f"Expected -1.0, got {score_neg}"

    print("    compute_spearmanr: Verified.")

    # --------------------------------------------------------------------------
    # 3. Dataset & DataLoader
    # --------------------------------------------------------------------------
    print("\n[3] Loading Data (Debug Mode)...")

    # Use debug=True to load only 100 rows
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    required_keys = [
        "title_input_ids",
        "title_attention_mask",
        "body_input_ids",
        "body_attention_mask",
        "answer_input_ids",
        "answer_attention_mask",
        "targets",
    ]

    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Verify shapes
    # Input IDs: (Batch, SeqLen)
    assert batch["title_input_ids"].shape[0] == Config.TRAIN_BATCH_SIZE
    # Targets: (Batch, 30)
    assert batch["targets"].shape == (Config.TRAIN_BATCH_SIZE, 30)

    print("    Batch structure and shapes: Verified.")

    # --------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # --------------------------------------------------------------------------
    print("\n[4] Initializing Model...")

    model = TripleBranchDistilRoBERTa()
    model.to(Config.DEVICE)

    print("    Model instantiated successfully.")

    # Perform forward pass with the batch retrieved earlier
    print("    Running forward pass check...")

    # Move batch to device
    inputs = {
        k: v.to(Config.DEVICE)
        for k, v in batch.items()
        if k != "qa_ids" and k != "targets"
    }

    with torch.no_grad():
        logits = model(**inputs)

    # Verify output shape (Batch, 30)
    assert logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        30,
    ), f"Output shape mismatch. Expected {(Config.TRAIN_BATCH_SIZE, 30)}, got {logits.shape}"

    # Verify values are finite
    assert torch.all(torch.isfinite(logits)), "Model produced NaN or Inf logits."

    print("    Forward pass: Verified.")

    # --------------------------------------------------------------------------
    # 5. Engine / Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[5] Demonstrating Training Loop (Engine)...")

    # Setup Optimizer using library function
    optimizer_params = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_params)

    # Setup Scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=len(train_loader)
    )

    print("    Running 1 Epoch of Training...")

    # Run Train Function
    train_loss = train_fn(model, train_loader, optimizer, Config.DEVICE, scheduler)
    print(f"    Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Train loss should be positive."

    print("    Running Validation...")

    # Run Eval Function
    val_loss, val_score = eval_fn(model, val_loader, Config.DEVICE)
    print(f"    Val Loss: {val_loss:.4f}")
    print(f"    Val Spearman: {val_score:.4f}")

    # Basic sanity checks on metrics
    assert val_loss > 0, "Validation loss should be positive."
    assert -1.0 <= val_score <= 1.0, "Spearman score out of range [-1, 1]."

    # Save model check
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model failed to save."
    print("    Model save: Verified.")

    # --------------------------------------------------------------------------
    # 6. Inference Demonstration
    # --------------------------------------------------------------------------
    print("\n[6] Demonstrating Inference...")

    model.eval()
    test_preds = []
    test_ids = []

    # Run inference on a few test batches
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Limit to 2 batches for speed
            if i >= 2:
                break

            inputs = {
                k: v.to(Config.DEVICE)
                for k, v in batch.items()
                if k != "qa_ids" and k != "targets"
            }

            logits = model(**inputs)
            preds = torch.sigmoid(logits).cpu().numpy()

            test_preds.append(preds)
            if "qa_ids" in batch:
                test_ids.extend(batch["qa_ids"])

    if test_preds:
        final_preds = np.vstack(test_preds)
        print(f"    Generated predictions shape: {final_preds.shape}")

        # Create submission dataframe
        sub_df = pd.DataFrame(final_preds, columns=Config.TARGET_COLS)
        if test_ids:
            sub_df.insert(0, "qa_id", test_ids)

        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."
        print(f"    Submission saved to: {Config.SUBMISSION_PATH}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
